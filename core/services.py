from datetime import datetime, timedelta

from django.core.cache import cache
from django.db.models import (
    Count,
    DurationField,
    ExpressionWrapper,
    F,
    Q,
    Sum,
)
from django.db.models.functions import TruncDate
from django.utils import timezone as dj_timezone
from django.utils.crypto import get_random_string
from rest_framework import status
from rest_framework.response import Response
from rest_framework.serializers import ValidationError
from rest_framework_simplejwt.tokens import RefreshToken

from .models import BloodSample, CustomUser, DialysisMachine, DialysisSession, Patient


# ---------------------------------------------------------------------------
# Authentication services
# ---------------------------------------------------------------------------

def login_user(data):
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return Response({'detail': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    user = CustomUser.objects.filter(username=username).first()
    if user is None or not user.check_password(password):
        return Response({'detail': 'Invalid username or password.'}, status=status.HTTP_401_UNAUTHORIZED)

    if not user.is_active:
        return Response({'detail': 'This account is disabled.'}, status=status.HTTP_401_UNAUTHORIZED)

    refresh = RefreshToken.for_user(user)
    return Response(
        {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'role': user.role,
                'department': user.department_id,
                'department_name': user.department.name if user.department else None,
                'phone_number': user.phone_number,
            },
        },
        status=status.HTTP_200_OK,
    )


def refresh_token(data):
    refresh_token_value = data.get('refresh')
    if not refresh_token_value:
        return Response({'detail': 'Refresh token is required.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        token = RefreshToken(refresh_token_value)
        return Response({'access': str(token.access_token)}, status=status.HTTP_200_OK)
    except Exception:
        return Response({'detail': 'Invalid or expired refresh token.'}, status=status.HTTP_401_UNAUTHORIZED)


def logout_user():
    # Stateless JWT: no blacklist is stored in Phase 1 (see api-auth-spec.md).
    return Response({'detail': 'Logged out successfully.'}, status=status.HTTP_200_OK)


def forgot_password(email):
    if not email:
        return Response({'detail': 'Email is required.'}, status=status.HTTP_400_BAD_REQUEST)

    user = CustomUser.objects.filter(email=email).first()
    if user is None:
        return Response(
            {'detail': 'If the email exists in the system, an OTP has been generated.'},
            status=status.HTTP_200_OK,
        )

    otp = get_random_string(length=6, allowed_chars='0123456789')
    cache.set(f'otp_{email}', otp, timeout=300)
    user.email_user(
        subject='Dialysis Center password reset OTP',
        message=f'Your OTP is: {otp}',
        from_email='no-reply@dialysis.local',
    )
    return Response({'detail': 'OTP has been generated and sent via email simulation.', 'otp': otp}, status=status.HTTP_200_OK)


def reset_password(email, otp, password):
    if not email or not otp or not password:
        return Response({'detail': 'Email, OTP and new password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    cached_otp = cache.get(f'otp_{email}')
    if cached_otp is None or str(cached_otp) != str(otp):
        return Response({'detail': 'Invalid or expired OTP.'}, status=status.HTTP_400_BAD_REQUEST)

    user = CustomUser.objects.filter(email=email).first()
    if user is None:
        return Response({'detail': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)

    user.set_password(password)
    user.save()
    cache.delete(f'otp_{email}')
    return Response({'detail': 'Password reset successfully.'}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Entity validation services (business rules per api-auth-spec.md)
# ---------------------------------------------------------------------------

def _merged(attrs, instance, field):
    if field in attrs:
        return attrs[field]
    return getattr(instance, field, None) if instance else None


def validate_patient_data(attrs, instance=None):
    errors = {}

    date_of_birth = _merged(attrs, instance, 'date_of_birth')
    if date_of_birth and date_of_birth > dj_timezone.localdate():
        errors['date_of_birth'] = ['Date of birth cannot be in the future.']

    dry_weight = _merged(attrs, instance, 'dry_weight')
    if dry_weight is not None and dry_weight <= 0:
        errors['dry_weight'] = ['Dry weight must be positive.']

    if errors:
        raise ValidationError(errors)


def validate_machine_data(attrs, instance=None):
    errors = {}

    last_maintenance_date = _merged(attrs, instance, 'last_maintenance_date')
    if last_maintenance_date and last_maintenance_date > dj_timezone.localdate():
        errors['last_maintenance_date'] = ['Last maintenance date cannot be in the future.']

    if errors:
        raise ValidationError(errors)


def validate_blood_sample_data(attrs, instance=None):
    errors = {}

    collection_date = _merged(attrs, instance, 'collection_date')
    if collection_date and collection_date > dj_timezone.now():
        errors['collection_date'] = ['Collection date cannot be in the future.']

    hemoglobin_level = _merged(attrs, instance, 'hemoglobin_level')
    if hemoglobin_level is not None and hemoglobin_level <= 0:
        errors['hemoglobin_level'] = ['Hemoglobin level must be a positive number.']

    potassium_level = _merged(attrs, instance, 'potassium_level')
    if potassium_level is not None and potassium_level <= 0:
        errors['potassium_level'] = ['Potassium level must be a positive number.']

    if errors:
        raise ValidationError(errors)


def validate_session_data(attrs, instance=None):
    errors = {}

    scheduled_start = _merged(attrs, instance, 'scheduled_start')
    scheduled_end = _merged(attrs, instance, 'scheduled_end')
    if scheduled_start and scheduled_end and scheduled_end <= scheduled_start:
        errors['scheduled_end'] = ['scheduled_end must be after scheduled_start.']

    assigned_nurse = _merged(attrs, instance, 'assigned_nurse')
    if assigned_nurse is not None and assigned_nurse.role != CustomUser.Role.NURSE:
        errors['assigned_nurse'] = ['assigned_nurse must be a user with role NURSE.']

    machine = _merged(attrs, instance, 'machine')
    if machine and scheduled_start and scheduled_end and scheduled_end > scheduled_start:
        overlapping = DialysisSession.objects.filter(
            machine=machine,
            scheduled_start__lt=scheduled_end,
            scheduled_end__gt=scheduled_start,
        )
        if instance and instance.pk:
            overlapping = overlapping.exclude(pk=instance.pk)
        if overlapping.exists():
            errors['machine'] = ['This machine already has a session overlapping the selected time window.']

    # State machine transition protection
    if instance and 'status' in attrs:
        old_status = instance.status
        new_status = attrs['status']
        if old_status != new_status:
            ALLOWED_TRANSITIONS = {
                DialysisSession.Status.SCHEDULED: {DialysisSession.Status.IN_PROGRESS, DialysisSession.Status.CANCELLED},
                DialysisSession.Status.IN_PROGRESS: {DialysisSession.Status.COMPLETED, DialysisSession.Status.CANCELLED},
                DialysisSession.Status.COMPLETED: set(),
                DialysisSession.Status.CANCELLED: set(),
            }
            allowed = ALLOWED_TRANSITIONS.get(old_status, set())
            if new_status not in allowed:
                errors['status'] = [f'Invalid status transition from {old_status} to {new_status}.']

    # Auto-stamp actual start/end on state transition if not provided
    if attrs.get('status') == DialysisSession.Status.IN_PROGRESS:
        if not attrs.get('actual_start') and (not instance or not instance.actual_start):
            attrs['actual_start'] = dj_timezone.now()
    elif attrs.get('status') == DialysisSession.Status.COMPLETED:
        if not attrs.get('actual_end') and (not instance or not instance.actual_end):
            attrs['actual_end'] = dj_timezone.now()

    # Clinical fields validation (UF in Liters, weight in kg)
    pre_weight = _merged(attrs, instance, 'pre_weight')
    if pre_weight is not None and (pre_weight <= 0 or pre_weight > 300):
        errors['pre_weight'] = ['Pre-dialysis weight must be between 0 and 300 kg.']

    post_weight = _merged(attrs, instance, 'post_weight')
    if post_weight is not None and (post_weight <= 0 or post_weight > 300):
        errors['post_weight'] = ['Post-dialysis weight must be between 0 and 300 kg.']

    uf_target = _merged(attrs, instance, 'uf_target')
    if uf_target is not None and (uf_target < 0 or uf_target > 10):
        errors['uf_target'] = ['UF target must be between 0 and 10 Liters.']

    uf_actual = _merged(attrs, instance, 'uf_actual')
    if uf_actual is not None and (uf_actual < 0 or uf_actual > 10):
        errors['uf_actual'] = ['UF actual must be between 0 and 10 Liters.']

    actual_start = _merged(attrs, instance, 'actual_start')
    actual_end = _merged(attrs, instance, 'actual_end')
    if actual_start and actual_end and actual_end <= actual_start:
        errors['actual_end'] = ['actual_end must be after actual_start.']

    if errors:
        raise ValidationError(errors)


def validate_vital_sign_data(attrs):
    errors = {}

    systolic_bp = attrs.get('systolic_bp')
    if systolic_bp is not None and systolic_bp <= 0:
        errors['systolic_bp'] = ['Systolic blood pressure must be positive.']

    diastolic_bp = attrs.get('diastolic_bp')
    if diastolic_bp is not None and diastolic_bp <= 0:
        errors['diastolic_bp'] = ['Diastolic blood pressure must be positive.']

    heart_rate = attrs.get('heart_rate')
    if heart_rate is not None and not (0 < heart_rate < 300):
        errors['heart_rate'] = ['Heart rate must be between 1 and 299.']

    spo2 = attrs.get('spo2')
    if spo2 is not None and not (0 <= spo2 <= 100):
        errors['spo2'] = ['SpO2 must be between 0 and 100.']

    temperature = attrs.get('temperature')
    if temperature is not None and not (30 <= temperature <= 45):
        errors['temperature'] = ['Temperature must be in a medically plausible range (30-45 °C).']

    if errors:
        raise ValidationError(errors)


# ---------------------------------------------------------------------------
# Dashboard services (basic operational stats only, Phase 1)
# ---------------------------------------------------------------------------

def dashboard_summary():
    today = dj_timezone.localdate()
    return {
        'total_patients': Patient.objects.count(),
        'total_staff': CustomUser.objects.filter(is_active=True).count(),
        'total_sessions': DialysisSession.objects.count(),
        'total_sessions_today': DialysisSession.objects.filter(scheduled_start__date=today).count(),
        'active_machines': DialysisMachine.objects.filter(status__in=['AVAILABLE', 'IN_USE']).count(),
    }


def parse_date_range(request, default_days=30):
    date_from_raw = request.query_params.get('date_from')
    date_to_raw = request.query_params.get('date_to')
    today = dj_timezone.localdate()

    date_from = None
    date_to = None

    if date_from_raw is not None and str(date_from_raw).strip() != '':
        try:
            date_from = datetime.strptime(str(date_from_raw).strip(), '%Y-%m-%d').date()
        except ValueError:
            return None, None, Response(
                {'detail': 'Invalid date format for date_from. Expected YYYY-MM-DD.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    if date_to_raw is not None and str(date_to_raw).strip() != '':
        try:
            date_to = datetime.strptime(str(date_to_raw).strip(), '%Y-%m-%d').date()
        except ValueError:
            return None, None, Response(
                {'detail': 'Invalid date format for date_to. Expected YYYY-MM-DD.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    if date_from is None and date_to is None:
        date_to = today
        date_from = today - timedelta(days=default_days - 1)
    elif date_from is None:
        date_from = date_to - timedelta(days=default_days - 1)
    elif date_to is None:
        date_to = today

    if date_from > date_to:
        return None, None, Response(
            {'detail': 'date_from cannot be greater than date_to.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    return date_from, date_to, None


def dashboard_dialysis_stats(days=7, date_from=None, date_to=None):
    if date_from is not None and date_to is not None:
        start_date = date_from
        end_date = date_to
    else:
        end_date = dj_timezone.localdate()
        start_date = end_date - timedelta(days=days - 1)

    aggregated = (
        DialysisSession.objects.filter(
            scheduled_start__date__gte=start_date,
            scheduled_start__date__lte=end_date,
        )
        .annotate(day=TruncDate('scheduled_start'))
        .values('day')
        .annotate(count=Count('session_id'))
    )
    counts = {row['day']: row['count'] for row in aggregated if row['day'] is not None}

    total_days = (end_date - start_date).days + 1
    return [
        {
            'date': (start_date + timedelta(days=i)).isoformat(),
            'count': counts.get(start_date + timedelta(days=i), 0),
        }
        for i in range(total_days)
    ]


def dashboard_machine_stats():
    rows = (
        DialysisMachine.objects.values('status')
        .annotate(count=Count('machine_id'))
        .order_by('status')
    )
    return [{'status': row['status'], 'count': row['count']} for row in rows]


# ---------------------------------------------------------------------------
# Reports services (Milestone 7: Reports Aggregation)
# ---------------------------------------------------------------------------

def report_operational_summary(date_from, date_to):
    sessions_qs = DialysisSession.objects.filter(
        scheduled_start__date__gte=date_from,
        scheduled_start__date__lte=date_to,
    )

    agg = sessions_qs.aggregate(
        total_sessions=Count('session_id'),
        completed_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.COMPLETED)),
        cancelled_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.CANCELLED)),
        scheduled_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.SCHEDULED)),
        in_progress_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.IN_PROGRESS)),
        total_uf_target=Sum('uf_target'),
        total_uf_actual=Sum('uf_actual'),
    )

    total = agg['total_sessions'] or 0
    completed = agg['completed_sessions'] or 0
    cancelled = agg['cancelled_sessions'] or 0
    scheduled = agg['scheduled_sessions'] or 0
    in_progress = agg['in_progress_sessions'] or 0
    completion_rate = round((completed / total) * 100.0, 2) if total > 0 else 0.0

    raw_uf_target = agg['total_uf_target']
    raw_uf_actual = agg['total_uf_actual']
    total_uf_target = round(float(raw_uf_target), 2) if raw_uf_target is not None else 0.0
    total_uf_actual = round(float(raw_uf_actual), 2) if raw_uf_actual is not None else 0.0

    daily_rows = (
        sessions_qs.annotate(date=TruncDate('scheduled_start'))
        .values('date')
        .annotate(
            total=Count('session_id'),
            completed=Count('session_id', filter=Q(status=DialysisSession.Status.COMPLETED)),
            cancelled=Count('session_id', filter=Q(status=DialysisSession.Status.CANCELLED)),
            scheduled=Count('session_id', filter=Q(status=DialysisSession.Status.SCHEDULED)),
            in_progress=Count('session_id', filter=Q(status=DialysisSession.Status.IN_PROGRESS)),
        )
        .order_by('date')
    )
    daily_map = {row['date']: row for row in daily_rows if row['date'] is not None}
    total_days = (date_to - date_from).days + 1
    daily_trends = [
        {
            'date': (date_from + timedelta(days=i)).isoformat(),
            'total': daily_map.get(date_from + timedelta(days=i), {}).get('total', 0),
            'completed': daily_map.get(date_from + timedelta(days=i), {}).get('completed', 0),
            'cancelled': daily_map.get(date_from + timedelta(days=i), {}).get('cancelled', 0),
            'scheduled': daily_map.get(date_from + timedelta(days=i), {}).get('scheduled', 0),
            'in_progress': daily_map.get(date_from + timedelta(days=i), {}).get('in_progress', 0),
        }
        for i in range(total_days)
    ]

    return {
        'date_from': date_from.isoformat(),
        'date_to': date_to.isoformat(),
        'total_sessions': total,
        'completed_sessions': completed,
        'cancelled_sessions': cancelled,
        'scheduled_sessions': scheduled,
        'in_progress_sessions': in_progress,
        'completion_rate': completion_rate,
        'total_uf_target': total_uf_target,
        'total_uf_actual': total_uf_actual,
        'daily_trends': daily_trends,
    }


def report_machine_utilization(date_from, date_to):
    machines_qs = (
        DialysisMachine.objects.select_related('department')
        .annotate(
            session_count=Count(
                'dialysis_sessions',
                filter=Q(
                    dialysis_sessions__scheduled_start__date__gte=date_from,
                    dialysis_sessions__scheduled_start__date__lte=date_to,
                ),
            ),
            completed_count=Count(
                'dialysis_sessions',
                filter=Q(
                    dialysis_sessions__scheduled_start__date__gte=date_from,
                    dialysis_sessions__scheduled_start__date__lte=date_to,
                    dialysis_sessions__status=DialysisSession.Status.COMPLETED,
                ),
            ),
            cancelled_count=Count(
                'dialysis_sessions',
                filter=Q(
                    dialysis_sessions__scheduled_start__date__gte=date_from,
                    dialysis_sessions__scheduled_start__date__lte=date_to,
                    dialysis_sessions__status=DialysisSession.Status.CANCELLED,
                ),
            ),
        )
        .order_by('machine_id')
    )

    runtime_qs = (
        DialysisSession.objects.filter(
            scheduled_start__date__gte=date_from,
            scheduled_start__date__lte=date_to,
            status=DialysisSession.Status.COMPLETED,
            actual_start__isnull=False,
            actual_end__isnull=False,
        )
        .annotate(
            duration=ExpressionWrapper(
                F('actual_end') - F('actual_start'),
                output_field=DurationField(),
            )
        )
        .values('machine_id')
        .annotate(total_duration=Sum('duration'))
    )
    runtime_map = {row['machine_id']: row['total_duration'] for row in runtime_qs}

    machine_list = []
    total_operating_seconds = 0.0
    total_completed = 0

    for m in machines_qs:
        duration_td = runtime_map.get(m.machine_id)
        seconds = duration_td.total_seconds() if duration_td is not None else 0.0
        seconds = max(0.0, seconds)
        total_operating_seconds += seconds
        total_completed += m.completed_count

        machine_list.append({
            'machine_id': m.machine_id,
            'name': m.name,
            'status': m.status,
            'department_id': m.department_id,
            'department_name': m.department.name if m.department else None,
            'last_maintenance_date': m.last_maintenance_date.isoformat() if m.last_maintenance_date else None,
            'session_count': m.session_count,
            'completed_count': m.completed_count,
            'cancelled_count': m.cancelled_count,
            'actual_runtime_hours': round(seconds / 3600.0, 2),
            'actual_runtime_minutes': round(seconds / 60.0, 1),
        })

    return {
        'date_from': date_from.isoformat(),
        'date_to': date_to.isoformat(),
        'limitation_note': 'Machine historical uptime/downtime percentage is not computed because status transition history logs are not tracked in the current database schema.',
        'total_machines': len(machine_list),
        'total_completed_sessions': total_completed,
        'total_runtime_hours': round(total_operating_seconds / 3600.0, 2),
        'machines': machine_list,
    }
