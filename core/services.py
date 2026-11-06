from collections import defaultdict
from datetime import datetime, timedelta

from django.core.cache import cache
from django.core.mail import send_mail
from django.conf import settings
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

from .models import (
    AuditLog,
    BloodSample,
    CustomUser,
    DialysisMachine,
    DialysisSession,
    Patient,
    ScheduleAssignment,
    SchedulePlan,
    TreatmentPattern,
)
from .audit import record_audit_log
from django.db import transaction


# ---------------------------------------------------------------------------
# Authentication services
# ---------------------------------------------------------------------------

LOGIN_LOCKOUT_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_TTL = 15 * 60  # 15 minutes


def _get_client_ip(request):
    if not request:
        return '127.0.0.1'
    x_forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded:
        return x_forwarded.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '127.0.0.1')


def login_user(data, request=None):
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return Response({'detail': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    cache_key = f'login_attempts_{username}'
    attempts = cache.get(cache_key, 0)
    lockout_msg = 'Tài khoản đã bị tạm khóa 15 phút do nhập sai mật khẩu 5 lần liên tiếp. Email hướng dẫn khôi phục đã được gửi.'

    if attempts >= LOGIN_LOCKOUT_MAX_ATTEMPTS:
        return Response({
            'detail': lockout_msg,
            'remaining_attempts': 0,
            'locked': True
        }, status=status.HTTP_403_FORBIDDEN)

    user = CustomUser.objects.filter(username=username).first()
    if user is None or not user.check_password(password):
        new_attempts = attempts + 1
        cache.set(cache_key, new_attempts, timeout=LOGIN_LOCKOUT_TTL)

        if new_attempts >= LOGIN_LOCKOUT_MAX_ATTEMPTS:
            client_ip = _get_client_ip(request)
            now_local = dj_timezone.localtime(dj_timezone.now())
            now_str = now_local.strftime('%H:%M:%S • %d/%m/%Y')

            if user and user.email:
                subject = '[Trung Tâm Lọc Máu LocMau] Cảnh báo bảo mật: Phát hiện 5 lần đăng nhập không thành công'
                body = (
                    f"Kính gửi {user.last_name} {user.first_name} ({user.username}),\n\n"
                    f"Hệ thống quản lý Trung tâm Lọc máu LocMau ghi nhận 5 lần đăng nhập không thành công liên tiếp vào tài khoản của bạn.\n\n"
                    f"--- THÔNG TIN SỰ KIỆN ---\n"
                    f"• Tên tài khoản: {user.username}\n"
                    f"• Vai trò: {user.role}\n"
                    f"• Thời gian vi phạm: {now_str} (Giờ địa phương Asia/Ho_Chi_Minh)\n"
                    f"• Địa chỉ IP client: {client_ip}\n\n"
                    f"--- HƯỚNG DẪN XỬ LÝ ---\n"
                    f"1. Nếu chính bạn quên mật khẩu: Vui lòng truy cập chức năng 'Quên mật khẩu' trên website hoặc liên hệ Quản trị viên (Admin) trung tâm để được cấp lại.\n"
                    f"2. Nếu KHÔNG phải bạn thực hiện: Đề nghị đổi mật khẩu ngay lập tức hoặc báo cho Admin khoa/phòng ban để bảo vệ hồ sơ bệnh án người bệnh.\n\n"
                    f"Tài khoản hiện đang tạm khóa đăng nhập trong 15 phút.\n\n"
                    f"Trân trọng,\nBan Quản trị An toàn Thông tin - Trung Tâm Lọc Máu LocMau"
                )
                from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@dialysis.local')
                try:
                    send_mail(subject, body, from_email, [user.email], fail_silently=True)
                except Exception:
                    pass

                record_audit_log(
                    actor=user,
                    action=AuditLog.Action.UPDATE,
                    instance=user,
                    changes={
                        'security_alert': 'LOGIN_LOCKOUT_5_FAILED_ATTEMPTS',
                        'attempts': new_attempts,
                        'client_ip': client_ip,
                        'timestamp': now_local.isoformat(),
                    },
                )
            return Response({
                'detail': lockout_msg,
                'remaining_attempts': 0,
                'locked': True
            }, status=status.HTTP_403_FORBIDDEN)

        remaining = LOGIN_LOCKOUT_MAX_ATTEMPTS - new_attempts
        return Response({
            'detail': f'Sai tên đăng nhập hoặc mật khẩu. Bạn còn {remaining}/5 lần thử trước khi tài khoản bị tạm khóa 15 phút.',
            'failed_attempts': new_attempts,
            'remaining_attempts': remaining,
            'max_attempts': LOGIN_LOCKOUT_MAX_ATTEMPTS
        }, status=status.HTTP_401_UNAUTHORIZED)

    if not user.is_active:
        return Response({'detail': 'This account is disabled.'}, status=status.HTTP_401_UNAUTHORIZED)

    # Success: clear failed attempts
    cache.delete(cache_key)

    # Update online presence in cache and DB
    now = dj_timezone.now()
    cache.set(f'user_online_{user.id}', True, timeout=300)
    user.last_seen = now
    user.save(update_fields=['last_seen'])

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
                'last_seen': user.last_seen.isoformat() if user.last_seen else None,
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


def forgot_password(email, request=None):
    if not email or not str(email).strip():
        return Response({'detail': 'Vui lòng cung cấp địa chỉ email.'}, status=status.HTTP_400_BAD_REQUEST)

    clean_email = str(email).strip().lower()
    user = CustomUser.objects.filter(email__iexact=clean_email).first()
    if user is None:
        return Response(
            {'detail': 'Địa chỉ email này chưa được đăng ký cho bất kỳ cán bộ y tế nào trong hệ thống.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    import random
    import sys
    otp = f"{random.randint(100000, 999999)}"

    cache.set(f'password_reset_otp_{clean_email}', otp, timeout=300)
    cache.set(f'otp_{clean_email}', otp, timeout=300)
    if user.email:
        cache.set(f'otp_{user.email}', otp, timeout=300)

    role_map = {
        CustomUser.Role.DOCTOR: 'Bác sĩ',
        CustomUser.Role.NURSE: 'Điều dưỡng',
        CustomUser.Role.ADMIN: 'Quản trị viên',
    }
    role_title = role_map.get(user.role, 'Cán bộ y tế')
    full_name = f"{user.last_name} {user.first_name}".strip() or user.username
    department_name = user.department.name if user.department else "Toàn viện (Ban Quản trị Trung tâm)"

    now_local = dj_timezone.localtime(dj_timezone.now())
    expiry_local = now_local + timedelta(seconds=300)
    timestamp_str = now_local.strftime('%H:%M:%S - Ngày %d/%m/%Y (GMT+7)')
    issued_time = now_local.strftime('%H:%M:%S')
    expiry_time = expiry_local.strftime('%H:%M:%S')

    client_ip = _get_client_ip(request) if request else '127.0.0.1'
    raw_ua = request.META.get('HTTP_USER_AGENT', 'Trình duyệt Web') if request else 'Trình duyệt Web'
    user_agent_short = raw_ua[:80] + '...' if len(raw_ua) > 80 else raw_ua

    failed_attempts = cache.get(f'login_attempts_{user.username}', 0)
    if failed_attempts >= LOGIN_LOCKOUT_MAX_ATTEMPTS:
        reason_str = "Yêu cầu khôi phục mật khẩu & Mở khóa tài khoản sau khi phát hiện nhập sai mật khẩu quá 5 lần liên tiếp"
    else:
        reason_str = "Yêu cầu cấp mã OTP khôi phục mật khẩu đăng nhập hệ thống từ người dùng"

    app_name = getattr(settings, 'EMAIL_APP_NAME', 'Dialysis Website TTam')
    from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'Trung Tâm Lọc Máu LocMau <admin.ttam.dialysis@gmail.com>')
    subject = f'[{app_name}] Mã xác thực OTP khôi phục mật khẩu - {user.username}'

    plain_text = (
        f"Kính gửi {role_title} {full_name},\n\n"
        f"Hệ thống Quản lý & Hỗ trợ Ra quyết định Trung tâm Lọc máu ({app_name}) đã tiếp nhận yêu cầu cấp mã xác thực OTP khôi phục mật khẩu cho tài khoản của bạn.\n\n"
        f"--- THÔNG TIN CHI TIẾT SỰ KIỆN BẢO MẬT ---\n"
        f"• Tài khoản đăng nhập: {user.username} ({user.email})\n"
        f"• Chức vụ & Đơn vị: {role_title} — {department_name}\n"
        f"• Thời gian ghi nhận: {timestamp_str}\n"
        f"• Vị trí & Thiết bị: IP: {client_ip} ({user_agent_short}) — Khu vực: TP.HCM, Việt Nam\n"
        f"• Lý do cấp mã: {reason_str}\n\n"
        f"--- MÃ XÁC THỰC OTP (06 CHỮ SỐ) ---\n"
        f"               [ {otp} ]\n\n"
        f"Mã OTP của bạn có hiệu lực trong vòng 05 phút (300 giây) kể từ thời điểm cấp ({issued_time} đến {expiry_time}).\n\n"
        f"--- HƯỚNG DẪN XỬ LÝ ---\n"
        f"Vui lòng nhập mã OTP 6 số này vào màn hình Khôi phục mật khẩu trên website để hoàn tất đặt lại mật khẩu mới (và tự động mở khóa tài khoản nếu đang bị tạm khóa).\n\n"
        f"Nếu không phải bạn yêu cầu cấp mã, xin vui lòng bỏ qua thư này hoặc thông báo ngay cho Quản trị viên Trung tâm Lọc máu.\n\n"
        f"Trân trọng,\nBan Quản trị An toàn Thông tin - Trung Tâm Lọc Máu LocMau"
    )

    html_message = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 24px; color: #1e293b; }}
  .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; }}
  .header {{ background: #1e3a8a; padding: 24px; text-align: center; color: #ffffff; }}
  .header h2 {{ margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }}
  .header p {{ margin: 6px 0 0 0; font-size: 13px; color: #93c5fd; }}
  .content {{ padding: 28px 24px; }}
  .greeting {{ font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 16px; }}
  .intro {{ font-size: 14px; line-height: 1.6; color: #334155; margin-bottom: 20px; }}
  .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 13.5px; background: #f8fafc; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }}
  .info-table td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
  .info-table tr:last-child td {{ border-bottom: none; }}
  .info-label {{ font-weight: 600; color: #64748b; width: 38%; }}
  .info-value {{ font-weight: 600; color: #0f172a; }}
  .otp-card {{ background: #eff6ff; border: 2px dashed #3b82f6; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 24px; }}
  .otp-title {{ font-size: 12.5px; font-weight: 700; color: #1d4ed8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .otp-code {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 34px; font-weight: 800; color: #1e3a8a; letter-spacing: 10px; margin: 8px 0; }}
  .otp-timer {{ font-size: 13px; color: #475569; margin-top: 8px; }}
  .otp-timer strong {{ color: #dc2626; }}
  .instructions {{ font-size: 13.5px; line-height: 1.6; color: #475569; border-left: 4px solid #3b82f6; padding-left: 14px; margin-bottom: 24px; }}
  .footer {{ background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748b; line-height: 1.5; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2>HỆ THỐNG QUẢN LÝ & HỖ TRỢ RA QUYẾT ĐỊNH TRUNG TÂM LỌC MÁU</h2>
    <p>Dialysis Website TTam • An Toàn Thông Tin Y Tế</p>
  </div>
  <div class="content">
    <div class="greeting">Kính gửi {role_title} {full_name},</div>
    <div class="intro">
      Hệ thống Trung tâm Lọc máu ghi nhận yêu cầu cấp mã xác thực OTP khôi phục mật khẩu cho tài khoản đăng nhập của bạn.
    </div>
    <table class="info-table">
      <tr><td class="info-label">Tài khoản đăng nhập:</td><td class="info-value">{user.username} ({user.email})</td></tr>
      <tr><td class="info-label">Chức vụ & Đơn vị:</td><td class="info-value">{role_title} — {department_name}</td></tr>
      <tr><td class="info-label">Thời gian ghi nhận:</td><td class="info-value">{timestamp_str}</td></tr>
      <tr><td class="info-label">Vị trí & Thiết bị:</td><td class="info-value">IP: {client_ip} ({user_agent_short})<br><small style="color:#64748b">Khu vực: TP. Hồ Chí Minh, Việt Nam</small></td></tr>
      <tr><td class="info-label">Lý do cấp mã:</td><td class="info-value" style="color:#2563eb;">{reason_str}</td></tr>
    </table>
    <div class="otp-card">
      <div class="otp-title">Mã Xác Thực Khôi Phục Mật Khẩu (OTP)</div>
      <div class="otp-code">{otp}</div>
      <div class="otp-timer">Mã có hiệu lực trong vòng <strong>05 phút (300 giây)</strong> ({issued_time} đến {expiry_time})</div>
    </div>
    <div class="instructions">
      <strong>Hướng dẫn:</strong> Nhập mã OTP 6 số này vào màn hình Khôi phục mật khẩu trên website để hoàn tất đặt lại mật khẩu mới (và tự động mở khóa tài khoản nếu đang bị tạm khóa).
    </div>
  </div>
  <div class="footer">
    Thư này được tạo tự động từ Hệ thống Trung tâm Lọc máu LocMau.<br>
    Nếu không phải bạn yêu cầu, xin vui lòng bỏ qua thư này hoặc thông báo cho Quản trị viên hệ thống.
  </div>
</div>
</body>
</html>"""

    try:
        send_mail(
            subject=subject,
            message=plain_text,
            from_email=from_email,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("SMTP send_mail error: %s", exc)

    return Response({
        'detail': f'Đã gửi mã OTP 6 chữ số tới hộp thư {user.email}. Mã có hiệu lực trong 05 phút.',
        'email': user.email,
        'otp': otp if (getattr(settings, 'DEBUG', False) or 'test' in sys.argv) else None,
    }, status=status.HTTP_200_OK)


def reset_password(email, otp, password, request=None):
    if not email or not str(email).strip():
        return Response({'detail': 'Email là bắt buộc.'}, status=status.HTTP_400_BAD_REQUEST)
    if not otp or not str(otp).strip():
        return Response({'detail': 'Mã OTP là bắt buộc.'}, status=status.HTTP_400_BAD_REQUEST)
    if not password:
        return Response({'detail': 'Mật khẩu mới là bắt buộc.'}, status=status.HTTP_400_BAD_REQUEST)
    if len(password) < 6:
        return Response({'detail': 'Mật khẩu mới phải có ít nhất 6 ký tự.'}, status=status.HTTP_400_BAD_REQUEST)

    clean_email = str(email).strip().lower()
    cached_otp = cache.get(f'password_reset_otp_{clean_email}') or cache.get(f'otp_{clean_email}') or cache.get(f'otp_{email}')
    if cached_otp is None or str(cached_otp).strip() != str(otp).strip():
        return Response({'detail': 'Mã OTP không chính xác hoặc đã hết hạn (hiệu lực 5 phút).'}, status=status.HTTP_400_BAD_REQUEST)

    user = CustomUser.objects.filter(email__iexact=clean_email).first()
    if user is None:
        return Response({'detail': 'Không tìm thấy người dùng khớp với email.'}, status=status.HTTP_404_NOT_FOUND)

    user.set_password(password)
    user.save()

    cache.delete(f'password_reset_otp_{clean_email}')
    cache.delete(f'otp_{clean_email}')
    cache.delete(f'otp_{email}')
    cache.delete(f'login_attempts_{user.username}')
    cache.delete(f'login_attempts_{user.username.lower()}')

    record_audit_log(
        actor=user,
        action=AuditLog.Action.UPDATE,
        instance=user,
        changes={
            'action': 'PASSWORD_RESET_VIA_EMAIL_OTP',
            'client_ip': _get_client_ip(request) if request else '127.0.0.1',
            'timestamp': dj_timezone.now().isoformat(),
        },
    )

    return Response({'detail': 'Đặt lại mật khẩu thành công. Tài khoản đã được mở khóa an toàn.'}, status=status.HTTP_200_OK)


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


MAX_CONCURRENT_SESSIONS_PER_NURSE = 4


def validate_session_data(attrs, instance=None):
    errors = {}

    scheduled_start = _merged(attrs, instance, 'scheduled_start')
    scheduled_end = _merged(attrs, instance, 'scheduled_end')
    if scheduled_start and scheduled_end and scheduled_end <= scheduled_start:
        errors['scheduled_end'] = ['scheduled_end must be after scheduled_start.']

    assigned_nurse = _merged(attrs, instance, 'assigned_nurse')
    if assigned_nurse is not None and assigned_nurse.role != CustomUser.Role.NURSE:
        errors['assigned_nurse'] = ['assigned_nurse must be a user with role NURSE.']

    patient = _merged(attrs, instance, 'patient')
    machine = _merged(attrs, instance, 'machine')

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

    target_status = attrs.get('status', instance.status if instance else DialysisSession.Status.SCHEDULED)

    # Overlap and availability checks apply only when target status is not CANCELLED
    if target_status != DialysisSession.Status.CANCELLED:
        # Machine status check: broken or maintenance machine cannot be scheduled
        if machine and machine.status in (DialysisMachine.Status.BROKEN, DialysisMachine.Status.MAINTENANCE):
            if instance is None or machine != instance.machine or target_status in (DialysisSession.Status.SCHEDULED, DialysisSession.Status.IN_PROGRESS):
                errors['machine'] = ['Cannot schedule session on a machine that is broken or under maintenance.']

        # Patient status check: discharged patient cannot be scheduled
        if patient and patient.status == Patient.Status.DISCHARGED:
            if instance is None or patient != instance.patient or target_status in (DialysisSession.Status.SCHEDULED, DialysisSession.Status.IN_PROGRESS):
                errors['patient'] = ['Cannot schedule session for a discharged patient.']

        if scheduled_start and scheduled_end and scheduled_end > scheduled_start:
            # Machine overlap check
            if machine:
                overlapping = DialysisSession.objects.filter(
                    machine=machine,
                    scheduled_start__lt=scheduled_end,
                    scheduled_end__gt=scheduled_start,
                ).exclude(status=DialysisSession.Status.CANCELLED)
                if instance and instance.pk:
                    overlapping = overlapping.exclude(pk=instance.pk)
                if overlapping.exists():
                    errors['machine'] = ['This machine already has a session overlapping the selected time window.']

            # Patient overlap check
            if patient:
                pat_overlapping = DialysisSession.objects.filter(
                    patient=patient,
                    scheduled_start__lt=scheduled_end,
                    scheduled_end__gt=scheduled_start,
                ).exclude(status=DialysisSession.Status.CANCELLED)
                if instance and instance.pk:
                    pat_overlapping = pat_overlapping.exclude(pk=instance.pk)
                if pat_overlapping.exists():
                    errors['patient'] = ['Patient already has an overlapping dialysis session in this time window.']

            # Nurse concurrency limit (MAX_CONCURRENT_SESSIONS_PER_NURSE = 4)
            if assigned_nurse and assigned_nurse.role == CustomUser.Role.NURSE:
                nurse_overlapping = DialysisSession.objects.filter(
                    assigned_nurse=assigned_nurse,
                    scheduled_start__lt=scheduled_end,
                    scheduled_end__gt=scheduled_start,
                ).exclude(status=DialysisSession.Status.CANCELLED)
                if instance and instance.pk:
                    nurse_overlapping = nurse_overlapping.exclude(pk=instance.pk)
                if nurse_overlapping.count() >= MAX_CONCURRENT_SESSIONS_PER_NURSE:
                    errors['assigned_nurse'] = [
                        f'Nurse cannot be assigned to more than {MAX_CONCURRENT_SESSIONS_PER_NURSE} overlapping sessions at the same time.'
                    ]

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


# ---------------------------------------------------------------------------
# Scheduling Approval & Lifecycle Services (Milestone 9 Phase 2)
# ---------------------------------------------------------------------------

def generate_dialysis_session_id(target_date) -> str:
    """
    Generate unique session_id adhering to SES-YYYYMMDD-XXXX format (17 chars <= 20).
    """
    if isinstance(target_date, datetime):
        d = target_date.date()
    else:
        d = target_date
    date_str = d.strftime('%Y%m%d')
    prefix = f'SES-{date_str}-'

    existing_ids = list(
        DialysisSession.objects.filter(session_id__startswith=prefix).values_list('session_id', flat=True)
    )
    max_seq = 0
    for sid in existing_ids:
        suffix = sid[len(prefix):]
        if suffix.isdigit():
            max_seq = max(max_seq, int(suffix))

    seq = max_seq + 1
    while True:
        candidate = f'{prefix}{seq:04d}'
        if not DialysisSession.objects.filter(session_id=candidate).exists():
            return candidate
        seq += 1


def validate_schedule_plan_for_approval(plan: SchedulePlan) -> dict:
    """
    Run mandatory 10-point validation before approving a SchedulePlan.
    preferred_shift is treated as a SOFT CONSTRAINT (returns warnings, does not block).
    """
    errors = []
    warnings = []

    # 1. plan.status == PROPOSED
    if plan.status != SchedulePlan.Status.PROPOSED:
        errors.append(f'Only PROPOSED plans can be approved. Current status: {plan.status}.')

    # 2. plan has at least 1 ScheduleAssignment
    assignments = list(plan.assignments.select_related('patient', 'machine').all())
    if not assignments:
        errors.append('Plan has no schedule assignments to approve.')
        return {'is_valid': False, 'errors': errors, 'warnings': warnings}

    for asgn in assignments:
        # 3. assignment.scheduled_date in [week_start, week_end]
        if asgn.scheduled_date < plan.week_start or asgn.scheduled_date > plan.week_end:
            errors.append(
                f'Assignment for patient {asgn.patient_id} on {asgn.scheduled_date} is outside plan week [{plan.week_start}, {plan.week_end}].'
            )

        # 4. start_datetime < end_datetime
        if asgn.start_datetime >= asgn.end_datetime:
            errors.append(
                f'Assignment for patient {asgn.patient_id} has invalid time window: start ({asgn.start_datetime}) must be before end ({asgn.end_datetime}).'
            )

        # 5. patient.status != DISCHARGED
        if asgn.patient.status == Patient.Status.DISCHARGED:
            errors.append(f'Patient {asgn.patient_id} is DISCHARGED and cannot be scheduled.')

        # 6. patient.treatment_pattern is not NULL
        if not asgn.patient.treatment_pattern:
            errors.append(f'Patient {asgn.patient_id} does not have a treatment pattern configured.')
        else:
            # 7. scheduled_date conforms to treatment_pattern
            # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
            weekday = asgn.scheduled_date.weekday()
            if asgn.patient.treatment_pattern == TreatmentPattern.T2_T4_T6:
                if weekday not in (0, 2, 4):
                    errors.append(
                        f'Scheduled date {asgn.scheduled_date} does not match patient {asgn.patient_id} treatment pattern (T2_T4_T6).'
                    )
            elif asgn.patient.treatment_pattern == TreatmentPattern.T3_T5_T7:
                if weekday not in (1, 3, 5):
                    errors.append(
                        f'Scheduled date {asgn.scheduled_date} does not match patient {asgn.patient_id} treatment pattern (T3_T5_T7).'
                    )

        # 8. machine.status is not BROKEN or MAINTENANCE
        if asgn.machine.status in ('BROKEN', 'MAINTENANCE'):
            errors.append(f'Machine {asgn.machine_id} is in {asgn.machine.status} status and cannot be used.')

        # 9. Overlap with existing DialysisSession of the same machine
        mach_conflicts = DialysisSession.objects.filter(
            machine=asgn.machine,
            scheduled_start__lt=asgn.end_datetime,
            scheduled_end__gt=asgn.start_datetime,
        ).exclude(status=DialysisSession.Status.CANCELLED)
        if mach_conflicts.exists():
            conflict_session = mach_conflicts.first()
            errors.append(
                f'Machine {asgn.machine_id} has a conflicting DialysisSession ({conflict_session.session_id}) overlapping [{asgn.start_datetime}, {asgn.end_datetime}].'
            )

        # 10. Overlap with existing DialysisSession of the same patient
        pat_conflicts = DialysisSession.objects.filter(
            patient=asgn.patient,
            scheduled_start__lt=asgn.end_datetime,
            scheduled_end__gt=asgn.start_datetime,
        ).exclude(status=DialysisSession.Status.CANCELLED)
        if pat_conflicts.exists():
            conflict_session = pat_conflicts.first()
            errors.append(
                f'Patient {asgn.patient_id} has a conflicting DialysisSession ({conflict_session.session_id}) overlapping [{asgn.start_datetime}, {asgn.end_datetime}].'
            )

        # 11. Cross-department checks: machine and patient must match plan department
        plan_dept_id = getattr(plan, 'department_id', None)
        if plan_dept_id:
            dept = getattr(plan, 'department', None)
            dept_name = getattr(dept, 'name', f'ID {plan_dept_id}')
            dept_code = getattr(dept, 'code', str(plan_dept_id))
            mach_dept_id = getattr(asgn.machine, 'department_id', None)
            pat_dept_id = getattr(asgn.patient, 'department_id', None)
            if mach_dept_id != plan_dept_id:
                errors.append(
                    f'Machine {asgn.machine_id} does not belong to plan department {dept_name} ({dept_code}).'
                )
            if pat_dept_id != plan_dept_id:
                errors.append(
                    f'Patient {asgn.patient_id} does not belong to plan department {dept_name} ({dept_code}).'
                )

        # Soft constraint: preferred_shift mismatch
        if asgn.patient.preferred_shift and asgn.shift != asgn.patient.preferred_shift:
            warnings.append(
                f'Patient {asgn.patient_id} preferred shift is {asgn.patient.preferred_shift}, but scheduled for {asgn.shift} on {asgn.scheduled_date}.'
            )

    # 12. Check max 1 session per patient per day in assignments
    patient_daily_map = defaultdict(list)
    for asgn in assignments:
        patient_daily_map[(asgn.patient_id, asgn.scheduled_date)].append(asgn)
    for (pid, s_date), asgns in patient_daily_map.items():
        if len(asgns) > 1:
            errors.append(
                f'Patient {pid} has {len(asgns)} assignments on {s_date}. Maximum 1 session per patient per day is allowed.'
            )

    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
    }


def approve_schedule_plan(plan_or_id, actor):
    """
    Approve a SchedulePlan within an atomic transaction.
    Validates, generates unique DialysisSessions, links them via schedule_assignment,
    updates plan status to APPROVED, and records comprehensive audit logs.
    """
    plan_id = plan_or_id.pk if hasattr(plan_or_id, 'pk') else plan_or_id

    with transaction.atomic():
        try:
            plan = SchedulePlan.objects.select_for_update().get(pk=plan_id)
        except SchedulePlan.DoesNotExist:
            raise ValidationError({'detail': 'Schedule plan not found.'})

        val_result = validate_schedule_plan_for_approval(plan)
        if not val_result['is_valid']:
            raise ValidationError({'errors': val_result['errors']})

        created_sessions = []
        for asgn in plan.assignments.select_related('patient', 'machine').all():
            session_id = generate_dialysis_session_id(asgn.scheduled_date)
            session = DialysisSession.objects.create(
                session_id=session_id,
                patient=asgn.patient,
                machine=asgn.machine,
                scheduled_start=asgn.start_datetime,
                scheduled_end=asgn.end_datetime,
                assigned_nurse=None,
                schedule_assignment=asgn,
                status=DialysisSession.Status.SCHEDULED,
            )
            created_sessions.append(session)
            record_audit_log(
                actor=actor,
                action=AuditLog.Action.CREATE,
                instance=session,
                changes={
                    'schedule_plan': plan.id,
                    'schedule_assignment': asgn.id,
                },
            )

        old_status = plan.status
        plan.status = SchedulePlan.Status.APPROVED
        plan.approved_by = actor
        plan.approved_at = dj_timezone.now()
        plan.save(update_fields=['status', 'approved_by', 'approved_at', 'updated_at'])

        record_audit_log(
            actor=actor,
            action=AuditLog.Action.UPDATE,
            instance=plan,
            changes={
                'status': {'before': old_status, 'after': SchedulePlan.Status.APPROVED},
                'approved_by': actor.username,
                'approved_at': plan.approved_at.isoformat(),
                'created_sessions_count': len(created_sessions),
            },
        )

        return plan, val_result['warnings'], created_sessions


def reject_schedule_plan(plan_or_id, actor, reason: str):
    """
    Reject a SchedulePlan with an explicit reason.
    Only PROPOSED plans can be rejected.
    """
    if not reason or not isinstance(reason, str) or not reason.strip():
        raise ValidationError({'reason': ['Rejection reason is required and cannot be empty.']})
    trimmed_reason = reason.strip()

    plan_id = plan_or_id.pk if hasattr(plan_or_id, 'pk') else plan_or_id

    with transaction.atomic():
        try:
            plan = SchedulePlan.objects.select_for_update().get(pk=plan_id)
        except SchedulePlan.DoesNotExist:
            raise ValidationError({'detail': 'Schedule plan not found.'})

        if plan.status != SchedulePlan.Status.PROPOSED:
            raise ValidationError({'status': [f'Only PROPOSED plans can be rejected. Current status: {plan.status}.']})

        old_status = plan.status
        plan.status = SchedulePlan.Status.REJECTED
        plan.rejection_reason = trimmed_reason
        plan.save(update_fields=['status', 'rejection_reason', 'updated_at'])

        record_audit_log(
            actor=actor,
            action=AuditLog.Action.UPDATE,
            instance=plan,
            changes={
                'status': {'before': old_status, 'after': SchedulePlan.Status.REJECTED},
                'rejection_reason': trimmed_reason,
            },
        )

        return plan
