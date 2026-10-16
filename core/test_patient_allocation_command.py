import json
import tempfile
from datetime import date
from io import StringIO
from django.core.management import call_command, CommandError
from django.test import TestCase

from core.models import Department, Patient, DialysisMachine, DialysisSession, SchedulePlan, ScheduleAssignment


class PatientAllocationCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Tạo khoa thử nghiệm
        cls.dept_a = Department.objects.create(
            name="Khoa Loc Mau A",
            code="KHOA_A",
            is_active=True,
        )
        cls.dept_b = Department.objects.create(
            name="Khoa Loc Mau B",
            code="KHOA_B",
            is_active=True,
        )
        cls.dept_inactive = Department.objects.create(
            name="Khoa Dong Cua",
            code="KHOA_CLOSED",
            is_active=False,
        )

        # Tạo máy và phiên lọc để xác minh phiên không bị ảnh hưởng
        cls.machine_a = DialysisMachine.objects.create(
            machine_id="TEST_MAC_01",
            name="May Loc A1",
            department=cls.dept_a,
            status=DialysisMachine.Status.AVAILABLE,
        )

    def setUp(self):
        # Tạo 10 bệnh nhân theo đúng mã trong DEFAULT_APPROVED_ALLOCATION
        self.patient_ids = [
            "P_T2_1", "P_T2_2", "P_T2_3", "P_T2_4", "P_T2_5",
            "P_T3_1", "P_T3_2", "P_T3_3", "P_T3_4", "P_T3_5",
        ]
        self.patients = []
        for pid in self.patient_ids:
            p = Patient.objects.create(
                patient_id=pid,
                full_name=f"Patient {pid}",
                date_of_birth=date(1990, 1, 1),
                department=None,
            )
            self.patients.append(p)

        # Thêm 1 bệnh nhân ngoài danh sách
        self.unmapped_patient = Patient.objects.create(
            patient_id="P_UNMAPPED_99",
            full_name="Patient Ngoai Danh Sach",
            date_of_birth=date(1985, 5, 5),
            department=None,
        )

        # Tạo phiên lọc cho P_T2_1 để xác minh phiên không bị thay đổi
        self.session = DialysisSession.objects.create(
            session_id="SES_TEST_ALLOC_01",
            patient=self.patients[0],
            machine=self.machine_a,
            scheduled_start="2026-10-19T07:00:00Z",
            scheduled_end="2026-10-19T11:00:00Z",
            status=DialysisSession.Status.SCHEDULED,
        )

    def test_dry_run_default_does_not_modify_database(self):
        """
        Lệnh chạy mặc định (dry-run) không được thay đổi bất kỳ trường department nào.
        """
        out = StringIO()
        call_command('allocate_patient_departments', stdout=out)
        output_str = out.getvalue()

        self.assertIn("DRY-RUN", output_str)
        self.assertIn("DRY-RUN HOAN TAT", output_str)

        # Kiểm tra database: toàn bộ 10 bệnh nhân vẫn department=None
        for pid in self.patient_ids:
            p = Patient.objects.get(patient_id=pid)
            self.assertIsNone(p.department)

        self.assertIsNone(Patient.objects.get(patient_id="P_UNMAPPED_99").department)

    def test_dry_run_explicit_flag_does_not_modify_database(self):
        """
        Chế độ --dry-run chỉ định tường minh không thay đổi database.
        """
        out = StringIO()
        call_command('allocate_patient_departments', dry_run=True, stdout=out)
        self.assertIn("DRY-RUN", out.getvalue())

        for pid in self.patient_ids:
            self.assertIsNone(Patient.objects.get(patient_id=pid).department)

    def test_apply_mode_successfully_allocates_approved_patients(self):
        """
        Chế độ --apply cập nhật chính xác khoa đích cho toàn bộ danh sách được phê duyệt.
        """
        out = StringIO()
        call_command('allocate_patient_departments', apply=True, stdout=out)
        output_str = out.getvalue()

        self.assertIn("HOAN TAT APPLY", output_str)

        # 10 bệnh nhân được phê duyệt phải được gán vào KHOA_A
        for pid in self.patient_ids:
            p = Patient.objects.get(patient_id=pid)
            self.assertEqual(p.department, self.dept_a)

        # Bệnh nhân ngoài danh sách vẫn giữ department=None
        unmapped = Patient.objects.get(patient_id="P_UNMAPPED_99")
        self.assertIsNone(unmapped.department)

        # Phiên lọc DialysisSession không bị thay đổi bất kỳ thuộc tính nào
        session_refreshed = DialysisSession.objects.get(session_id="SES_TEST_ALLOC_01")
        self.assertEqual(session_refreshed.status, DialysisSession.Status.SCHEDULED)
        self.assertEqual(session_refreshed.machine, self.machine_a)

    def test_inactive_target_department_aborts_with_error(self):
        """
        Nếu bảng phân bổ chỉ định khoa ngừng hoạt động, lệnh phải từ chối và báo lỗi.
        """
        mapping = {"P_T2_1": "KHOA_CLOSED"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(mapping, f)
            temp_path = f.name

        out = StringIO()
        with self.assertRaises(CommandError) as ctx:
            call_command('allocate_patient_departments', mapping_file=temp_path, apply=True, stdout=out)

        self.assertIn("Tien trinh phan bo bi huy bo", str(ctx.exception))
        # Không có bệnh nhân nào bị cập nhật
        self.assertIsNone(Patient.objects.get(patient_id="P_T2_1").department)

    def test_patient_already_assigned_aborts_safely_preventing_overwrite(self):
        """
        Nếu bệnh nhân đã có khoa được gán từ trước, lệnh dừng khẩn cấp để ngăn ghi đè trái phép.
        """
        # Gán sẵn khoa cho P_T2_1
        p1 = Patient.objects.get(patient_id="P_T2_1")
        p1.department = self.dept_b
        p1.save()

        out = StringIO()
        with self.assertRaises(CommandError) as ctx:
            call_command('allocate_patient_departments', apply=True, stdout=out)

        self.assertIn("Tien trinh phan bo bi huy bo", str(ctx.exception))
        # P_T2_1 vẫn giữ nguyên khoa KHOA_B cũ
        p1_refreshed = Patient.objects.get(patient_id="P_T2_1")
        self.assertEqual(p1_refreshed.department, self.dept_b)

        # Các bệnh nhân khác không bị cập nhật do transaction/validation abort
        p2_refreshed = Patient.objects.get(patient_id="P_T2_2")
        self.assertIsNone(p2_refreshed.department)

    def test_nonexistent_patient_aborts_safely(self):
        """
        Nếu trong mapping có bệnh nhân không tồn tại trong DB, lệnh dừng khẩn cấp.
        """
        mapping = {"P_NONEXISTENT_999": "KHOA_A"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(mapping, f)
            temp_path = f.name

        out = StringIO()
        with self.assertRaises(CommandError) as ctx:
            call_command('allocate_patient_departments', mapping_file=temp_path, apply=True, stdout=out)

        self.assertIn("Tien trinh phan bo bi huy bo", str(ctx.exception))

    def test_transaction_atomic_rollback_on_apply_failure(self):
        """
        Nếu xảy ra lỗi trong quá trình apply, transaction.atomic() bảo đảm rollback toàn bộ.
        """
        # Thử nghiệm với mapping 2 bệnh nhân hợp lệ
        mapping = {
            "P_T2_1": "KHOA_A",
            "P_T2_2": "KHOA_A",
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(mapping, f)
            temp_path = f.name

        out = StringIO()
        call_command('allocate_patient_departments', mapping_file=temp_path, apply=True, stdout=out)

        self.assertEqual(Patient.objects.get(patient_id="P_T2_1").department, self.dept_a)
        self.assertEqual(Patient.objects.get(patient_id="P_T2_2").department, self.dept_a)
