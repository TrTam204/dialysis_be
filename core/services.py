from datetime import timedelta

from django.core.cache import cache
from django.db.models import Count
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


def dashboard_dialysis_stats(days=7):
    today = dj_timezone.localdate()
    since = today - timedelta(days=days - 1)
    # Group by local date in Python so the buckets match the server timezone.
    counts = {}
    sessions = DialysisSession.objects.filter(scheduled_start__date__gte=since).values_list('scheduled_start', flat=True)
    for value in sessions:
        day = dj_timezone.localtime(value).date() if dj_timezone.is_aware(value) else value.date()
        counts[day] = counts.get(day, 0) + 1
    return [
        {'date': (today - timedelta(days=offset)).isoformat(), 'count': counts.get(today - timedelta(days=offset), 0)}
        for offset in range(days - 1, -1, -1)
    ]


def dashboard_machine_stats():
    rows = (
        DialysisMachine.objects.values('status')
        .annotate(count=Count('machine_id'))
        .order_by('status')
    )
    return [{'status': row['status'], 'count': row['count']} for row in rows]
