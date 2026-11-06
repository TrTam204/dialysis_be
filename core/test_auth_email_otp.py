from django.core import mail
from django.core.cache import cache
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from core.models import AuditLog, CustomUser, Department


class AuthEmailOtpTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

        self.dept = Department.objects.create(
            name="Khoa Lọc Máu A Test",
            code="KLM_A_TEST",
            is_active=True,
            head_doctor_name="BS.CKII Nguyễn Hồ An Thiên",
        )

        self.doctor = CustomUser.objects.create_user(
            username="bs_thien",
            password="Password@123",
            email="bs.thien@dialysis.local",
            role=CustomUser.Role.DOCTOR,
            first_name="Thiên",
            last_name="Nguyễn Hồ An",
            department=self.dept,
        )

        self.admin = CustomUser.objects.create_user(
            username="admin_ttam",
            password="AdminPassword@123",
            email="admin.ttam.dialysis@gmail.com",
            role=CustomUser.Role.ADMIN,
            first_name="Tam",
            last_name="Trường",
        )

    def test_01_forgot_password_missing_email(self):
        resp = self.client.post('/api/auth/forgot-password/', {'email': ''}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_02_forgot_password_unregistered_email(self):
        resp = self.client.post('/api/auth/forgot-password/', {'email': 'unknown@hospital.vn'}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('chưa được đăng ký cho bất kỳ cán bộ y tế nào', resp.data['detail'])

    def test_03_forgot_password_success_doctor_email_content(self):
        resp = self.client.post('/api/auth/forgot-password/', {'email': 'bs.thien@dialysis.local'}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('Đã gửi mã OTP 6 chữ số', resp.data['detail'])

        # Verify email sent via Django mail.outbox
        self.assertEqual(len(mail.outbox), 1)
        sent_mail = mail.outbox[0]
        self.assertIn('bs.thien@dialysis.local', sent_mail.to)
        self.assertIn('Mã xác thực OTP', sent_mail.subject)
        self.assertIn('bs_thien', sent_mail.subject)

        # Content checks: Role title, Name, Department, 5-minute expiry
        body = sent_mail.body
        self.assertIn('Kính gửi Bác sĩ Nguyễn Hồ An Thiên', body)
        self.assertIn('Khoa Lọc Máu A Test', body)
        self.assertIn('05 phút (300 giây)', body)
        self.assertIn('TP.HCM, Việt Nam', body)

        # HTML body check
        self.assertTrue(len(sent_mail.alternatives) > 0)
        html_body, mime_type = sent_mail.alternatives[0]
        self.assertEqual(mime_type, 'text/html')
        self.assertIn('HỆ THỐNG QUẢN LÝ & HỖ TRỢ RA QUYẾT ĐỊNH TRUNG TÂM LỌC MÁU', html_body)
        self.assertIn('Kính gửi Bác sĩ Nguyễn Hồ An Thiên', html_body)

    def test_04_forgot_password_lockout_reason(self):
        # Simulate 5 failed login attempts
        cache.set('login_attempts_admin_ttam', 5, timeout=900)

        resp = self.client.post('/api/auth/forgot-password/', {'email': 'admin.ttam.dialysis@gmail.com'}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        self.assertEqual(len(mail.outbox), 1)
        body = mail.outbox[0].body
        self.assertIn('Kính gửi Quản trị viên Trường Tam', body)
        self.assertIn('sau khi phát hiện nhập sai mật khẩu quá 5 lần liên tiếp', body)

    def test_05_reset_password_invalid_or_expired_otp(self):
        resp = self.client.post(
            '/api/auth/reset-password/',
            {'email': 'bs.thien@dialysis.local', 'otp': '000000', 'password': 'NewPassword@123'},
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('không chính xác hoặc đã hết hạn', resp.data['detail'])

    def test_06_reset_password_short_password(self):
        # Generate valid OTP
        self.client.post('/api/auth/forgot-password/', {'email': 'bs.thien@dialysis.local'}, format='json')
        cached_otp = cache.get('password_reset_otp_bs.thien@dialysis.local')

        resp = self.client.post(
            '/api/auth/reset-password/',
            {'email': 'bs.thien@dialysis.local', 'otp': cached_otp, 'password': '123'},
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('ít nhất 6 ký tự', resp.data['detail'])

    def test_07_reset_password_success_clears_lockout_and_logs(self):
        # Generate OTP
        self.client.post('/api/auth/forgot-password/', {'email': 'bs.thien@dialysis.local'}, format='json')
        cached_otp = cache.get('password_reset_otp_bs.thien@dialysis.local')
        self.assertIsNotNone(cached_otp)

        # Set lockout attempt in cache
        cache.set('login_attempts_bs_thien', 5, timeout=900)

        # Reset password
        resp = self.client.post(
            '/api/auth/reset-password/',
            {'email': 'bs.thien@dialysis.local', 'otp': cached_otp, 'password': 'BrandNewPassword@456'},
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('Đặt lại mật khẩu thành công', resp.data['detail'])

        # Check cache cleared
        self.assertIsNone(cache.get('password_reset_otp_bs.thien@dialysis.local'))
        self.assertIsNone(cache.get('login_attempts_bs_thien'))

        # Check user can login with new password
        login_resp = self.client.post(
            '/api/auth/login/',
            {'username': 'bs_thien', 'password': 'BrandNewPassword@456'},
            format='json',
        )
        self.assertEqual(login_resp.status_code, status.HTTP_200_OK)
        self.assertIn('access', login_resp.data)

        # Check audit log recorded
        audit = AuditLog.objects.filter(entity_type='CustomUser', entity_id=str(self.doctor.id)).order_by('-id').first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.changes.get('action'), 'PASSWORD_RESET_VIA_EMAIL_OTP')

