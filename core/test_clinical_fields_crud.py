from datetime import date, datetime
from django.test import TestCase
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from core.models import CustomUser, Department, Patient, DialysisMachine, BloodSample, DialysisSession


class ClinicalFieldsCrudTests(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Admin user
        self.admin = CustomUser.objects.create_user(
            username="admin_test",
            password="AdminPassword@123",
            email="admin@dialysis.test",
            role=CustomUser.Role.ADMIN,
            first_name="Admin",
            last_name="System",
            employee_code="AD-TEST-01",
        )

        # Department
        self.dept = Department.objects.create(
            name="Khoa Lọc Máu A Test",
            code="KLM_A_CRUD",
            description="Khoa thử nghiệm CRUD 2 chiều",
            is_active=True,
            head_doctor_name="BS.CKII Nguyễn Hồ An Thiên",
            location_floor="Tầng 3 Khu A",
            hotline_phone="028.3855.4137",
            max_machine_capacity=25,
        )

        # Doctor user
        self.doctor = CustomUser.objects.create_user(
            username="doctor_test",
            password="DoctorPassword@123",
            email="doctor@dialysis.test",
            role=CustomUser.Role.DOCTOR,
            first_name="Thiên",
            last_name="Nguyễn Hồ An",
            department=self.dept,
            employee_code="BS-CRUD-01",
            gender="MALE",
            date_of_birth=date(1980, 5, 15),
            citizen_id="079080001234",
            address="123 Nguyễn Chí Thanh, Quận 5, TP.HCM",
            degree_title="BS.CKII",
            license_number="045812/HCM-CCHN",
            specialization="Thận nhân tạo - Lọc máu ngoài cơ thể",
        )

        # Nurse user
        self.nurse = CustomUser.objects.create_user(
            username="nurse_test",
            password="NursePassword@123",
            email="nurse@dialysis.test",
            role=CustomUser.Role.NURSE,
            first_name="Mai",
            last_name="Lê Thị",
            department=self.dept,
            employee_code="DD-CRUD-02",
        )

        # Patient
        self.patient = Patient.objects.create(
            patient_id="P_CRUD_01",
            full_name="Trần Văn Khỏe",
            date_of_birth=date(1965, 8, 20),
            gender="MALE",
            status=Patient.Status.ACTIVE,
            department=self.dept,
            citizen_id="079065009876",
            health_insurance_number="GD4790123456789",
            permanent_address="456 Lý Thường Kiệt, Quận 10, TP.HCM",
            blood_type="O+",
            emergency_contact_name="Trần Thị Bích",
            emergency_contact_relationship="Vợ/Chồng",
            emergency_contact_phone="0918765432",
            vascular_access_type="AVF",
        )

        # Machine
        self.machine = DialysisMachine.objects.create(
            machine_id="M_CRUD_01",
            name="Fresenius 4008S Test",
            department=self.dept,
            status=DialysisMachine.Status.AVAILABLE,
            last_maintenance_date=date(2026, 10, 25),
            maintenance_log="Bảo dưỡng định kỳ màng lọc và cân UF đạt chuẩn.",
        )

    # 1. CustomUser CRUD 2-way tests
    def test_01_user_create_and_edit_clinical_fields(self):
        self.client.force_authenticate(user=self.admin)
        create_payload = {
            "username": "new_doctor_crud",
            "password": "Password@123456",
            "email": "new.doctor@dialysis.test",
            "role": "DOCTOR",
            "first_name": "Minh",
            "last_name": "Phạm Văn",
            "department": self.dept.id,
            "employee_code": "BS-NEW-99",
            "gender": "MALE",
            "date_of_birth": "1985-12-10",
            "citizen_id": "079085009999",
            "address": "789 Cách Mạng Tháng 8, Quận 3, TP.HCM",
            "degree_title": "BS.CKI",
            "license_number": "045999/HCM-CCHN",
            "specialization": "Hồi sức Cấp cứu Nội thận & Thận nhân tạo",
            "is_active": True,
        }
        res = self.client.post("/api/users/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        new_id = res.data["id"]
        self.assertEqual(res.data["employee_code"], "BS-NEW-99")
        self.assertEqual(res.data["degree_title"], "BS.CKI")
        self.assertEqual(res.data["license_number"], "045999/HCM-CCHN")
        self.assertEqual(res.data["specialization"], "Hồi sức Cấp cứu Nội thận & Thận nhân tạo")

        # Edit (PATCH / PUT)
        update_payload = {
            "employee_code": "BS-NEW-99-UPDATED",
            "degree_title": "BS.CKII",
            "specialization": "Thận học Can thiệp & Lọc máu",
            "address": "999 Nguyễn Trãi, Quận 5, TP.HCM",
        }
        res_edit = self.client.patch(f"/api/users/{new_id}/", update_payload, format="json")
        self.assertEqual(res_edit.status_code, status.HTTP_200_OK)
        self.assertEqual(res_edit.data["employee_code"], "BS-NEW-99-UPDATED")
        self.assertEqual(res_edit.data["degree_title"], "BS.CKII")
        self.assertEqual(res_edit.data["specialization"], "Thận học Can thiệp & Lọc máu")

        # Verify DB directly
        u = CustomUser.objects.get(id=new_id)
        self.assertEqual(u.employee_code, "BS-NEW-99-UPDATED")
        self.assertEqual(u.degree_title, "BS.CKII")

    # 2. Patient CRUD 2-way tests
    def test_02_patient_create_and_edit_clinical_fields(self):
        self.client.force_authenticate(user=self.doctor)
        create_payload = {
            "patient_id": "P_CRUD_NEW_99",
            "full_name": "Hoàng Thị Lan",
            "date_of_birth": "1972-04-18",
            "gender": "FEMALE",
            "status": "ACTIVE",
            "department": self.dept.id,
            "citizen_id": "079172008888",
            "health_insurance_number": "GD4790998877665",
            "permanent_address": "321 An Dương Vương, Quận 5, TP.HCM",
            "blood_type": "B+",
            "emergency_contact_name": "Hoàng Văn Tuấn",
            "emergency_contact_relationship": "Con ruột",
            "emergency_contact_phone": "0909112233",
            "vascular_access_type": "AVG",
            "dry_weight": 52.5,
            "latitude": 10.7554,
            "longitude": 106.6621,
        }
        res = self.client.post("/api/patients/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data["citizen_id"], "079172008888")
        self.assertEqual(res.data["health_insurance_number"], "GD4790998877665")
        self.assertEqual(res.data["blood_type"], "B+")
        self.assertEqual(res.data["vascular_access_type"], "AVG")
        self.assertEqual(res.data["emergency_contact_relationship"], "Con ruột")

        # Edit (PATCH / PUT)
        update_payload = {
            "blood_type": "AB+",
            "vascular_access_type": "CATHETER_TUNNELED",
            "permanent_address": "888 Ba Tháng Hai, Quận 10, TP.HCM",
            "dry_weight": 54.0,
            "emergency_contact_phone": "0988776655",
        }
        res_edit = self.client.patch("/api/patients/P_CRUD_NEW_99/", update_payload, format="json")
        self.assertEqual(res_edit.status_code, status.HTTP_200_OK)
        self.assertEqual(res_edit.data["blood_type"], "AB+")
        self.assertEqual(res_edit.data["vascular_access_type"], "CATHETER_TUNNELED")
        self.assertEqual(res_edit.data["dry_weight"], 54.0)
        self.assertEqual(res_edit.data["emergency_contact_phone"], "0988776655")

        # Verify DB directly
        p = Patient.objects.get(patient_id="P_CRUD_NEW_99")
        self.assertEqual(p.blood_type, "AB+")
        self.assertEqual(p.vascular_access_type, "CATHETER_TUNNELED")

    # 3. Department CRUD 2-way tests
    def test_03_department_create_and_edit_clinical_fields(self):
        self.client.force_authenticate(user=self.admin)
        create_payload = {
            "code": "KLM_C_TEST",
            "name": "Khoa Lọc Máu C Cấp Cứu",
            "description": "Đơn vị lọc máu hồi sức tích cực",
            "head_doctor_name": "BS.CKII Lê Hoàng Long",
            "location_floor": "Tầng 4 Tòa nhà C",
            "hotline_phone": "028.3955.9988",
            "max_machine_capacity": 30,
            "is_active": True,
        }
        res = self.client.post("/api/departments/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        dept_id = res.data["id"]
        self.assertEqual(res.data["head_doctor_name"], "BS.CKII Lê Hoàng Long")
        self.assertEqual(res.data["location_floor"], "Tầng 4 Tòa nhà C")
        self.assertEqual(res.data["hotline_phone"], "028.3955.9988")
        self.assertEqual(res.data["max_machine_capacity"], 30)

        # Edit (PATCH / PUT)
        update_payload = {
            "head_doctor_name": "BS.CKII Trần Quốc Bảo",
            "max_machine_capacity": 35,
            "hotline_phone": "028.3955.1122",
        }
        res_edit = self.client.patch(f"/api/departments/{dept_id}/", update_payload, format="json")
        self.assertEqual(res_edit.status_code, status.HTTP_200_OK)
        self.assertEqual(res_edit.data["head_doctor_name"], "BS.CKII Trần Quốc Bảo")
        self.assertEqual(res_edit.data["max_machine_capacity"], 35)

        # Verify DB directly
        d = Department.objects.get(id=dept_id)
        self.assertEqual(d.head_doctor_name, "BS.CKII Trần Quốc Bảo")
        self.assertEqual(d.max_machine_capacity, 35)

    # 4. BloodSample CRUD 2-way tests (9 clinical markers)
    def test_04_blood_sample_create_and_edit_clinical_fields(self):
        self.client.force_authenticate(user=self.doctor)
        create_payload = {
            "sample_id": "BS_CRUD_999",
            "patient": self.patient.patient_id,
            "collection_date": timezone.now().isoformat(),
            "hemoglobin_level": 10.8,
            "potassium_level": 4.6,
            "sodium_level": 139.5,
            "calcium_level": 2.35,
            "phosphorus_level": 1.45,
            "urea_level": 22.4,
            "creatinine_level": 680.0,
            "albumin_level": 3.9,
            "kt_v": 1.42,
            "notes": "Mẫu máu định kỳ đầu tháng",
        }
        res = self.client.post("/api/blood-samples/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data["sodium_level"], 139.5)
        self.assertEqual(res.data["calcium_level"], 2.35)
        self.assertEqual(res.data["phosphorus_level"], 1.45)
        self.assertEqual(res.data["urea_level"], 22.4)
        self.assertEqual(res.data["creatinine_level"], 680.0)
        self.assertEqual(res.data["albumin_level"], 3.9)
        self.assertEqual(res.data["kt_v"], 1.42)

        # Edit (PATCH / PUT)
        update_payload = {
            "hemoglobin_level": 11.2,
            "potassium_level": 4.8,
            "sodium_level": 141.0,
            "kt_v": 1.55,
            "notes": "Cập nhật sau phiên lọc máu",
        }
        res_edit = self.client.patch("/api/blood-samples/BS_CRUD_999/", update_payload, format="json")
        self.assertEqual(res_edit.status_code, status.HTTP_200_OK)
        self.assertEqual(res_edit.data["hemoglobin_level"], 11.2)
        self.assertEqual(res_edit.data["sodium_level"], 141.0)
        self.assertEqual(res_edit.data["kt_v"], 1.55)

        # Verify DB directly
        bs = BloodSample.objects.get(sample_id="BS_CRUD_999")
        self.assertEqual(bs.hemoglobin_level, 11.2)
        self.assertEqual(bs.sodium_level, 141.0)
        self.assertEqual(bs.kt_v, 1.55)

    # 5. DialysisSession CRUD 2-way tests (Dialysis Prescription)
    def test_05_session_create_and_edit_clinical_fields(self):
        self.client.force_authenticate(user=self.doctor)
        now = timezone.now()
        start = now
        end = now + timezone.timedelta(hours=4)
        create_payload = {
            "session_id": "SS_CRUD_999",
            "patient": self.patient.patient_id,
            "machine": self.machine.machine_id,
            "assigned_nurse": self.nurse.id,
            "scheduled_start": start.isoformat(),
            "scheduled_end": end.isoformat(),
            "status": "SCHEDULED",
            "blood_flow_rate": 260,
            "dialysate_flow_rate": 500,
            "dialysate_temperature": 36.5,
            "dialyzer_type": "High-Flux FX60",
            "anticoagulant_type": "Heparin tiêu chuẩn (5000 IU/phiên)",
            "uf_target": 2.5,
            "notes": "Y lệnh ca sáng",
        }
        res = self.client.post("/api/sessions/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data["blood_flow_rate"], 260)
        self.assertEqual(res.data["dialysate_flow_rate"], 500)
        self.assertEqual(res.data["dialysate_temperature"], 36.5)
        self.assertEqual(res.data["dialyzer_type"], "High-Flux FX60")
        self.assertEqual(res.data["anticoagulant_type"], "Heparin tiêu chuẩn (5000 IU/phiên)")

        # Edit (PATCH / PUT)
        update_payload = {
            "blood_flow_rate": 280,
            "dialysate_temperature": 36.0,
            "dialyzer_type": "High-Flux FX80",
            "anticoagulant_type": "LMWH Enoxaparin (40mg)",
        }
        res_edit = self.client.patch("/api/sessions/SS_CRUD_999/", update_payload, format="json")
        self.assertEqual(res_edit.status_code, status.HTTP_200_OK)
        self.assertEqual(res_edit.data["blood_flow_rate"], 280)
        self.assertEqual(res_edit.data["dialysate_temperature"], 36.0)
        self.assertEqual(res_edit.data["dialyzer_type"], "High-Flux FX80")
        self.assertEqual(res_edit.data["anticoagulant_type"], "LMWH Enoxaparin (40mg)")

        # Verify DB directly
        s = DialysisSession.objects.get(session_id="SS_CRUD_999")
        self.assertEqual(s.blood_flow_rate, 280)
        self.assertEqual(s.dialysate_temperature, 36.0)
        self.assertEqual(s.dialyzer_type, "High-Flux FX80")

    # 6. DialysisMachine CRUD 2-way tests (Maintenance log & date)
    def test_06_machine_create_and_edit_maintenance_fields(self):
        self.client.force_authenticate(user=self.admin)
        create_payload = {
            "machine_id": "M_CRUD_NEW",
            "name": "Nikkiso DBB-07 New",
            "department": self.dept.id,
            "status": "AVAILABLE",
            "last_maintenance_date": "2026-10-30",
            "maintenance_log": "Khử trùng nhiệt tự động và hiệu chuẩn cảm biến.",
        }
        res = self.client.post("/api/machines/", create_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertEqual(res.data["last_maintenance_date"], "2026-10-30")
        self.assertEqual(res.data["maintenance_log"], "Khử trùng nhiệt tự động và hiệu chuẩn cảm biến.")

        # Nurse partial update (testing nurse permission for maintenance)
        self.client.force_authenticate(user=self.nurse)
        nurse_update = {
            "status": "MAINTENANCE",
            "last_maintenance_date": "2026-11-05",
            "maintenance_log": "Điều dưỡng ghi nhận cảnh báo áp lực dịch lọc cao, tạm ngưng để bảo trì.",
        }
        res_nurse = self.client.patch("/api/machines/M_CRUD_NEW/", nurse_update, format="json")
        self.assertEqual(res_nurse.status_code, status.HTTP_200_OK)
        self.assertEqual(res_nurse.data["status"], "MAINTENANCE")
        self.assertEqual(res_nurse.data["last_maintenance_date"], "2026-11-05")

        # Verify DB directly
        m = DialysisMachine.objects.get(machine_id="M_CRUD_NEW")
        self.assertEqual(m.status, DialysisMachine.Status.MAINTENANCE)
        self.assertEqual(str(m.last_maintenance_date), "2026-11-05")

