import random
import sys
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.models import (
    CustomUser,
    Department,
    Patient,
    BloodSample,
    DialysisSession,
)


class Command(BaseCommand):
    help = "Seed realistic clinical profiles for all Staff, Patients, Departments, BloodSamples, and Sessions."

    def handle(self, *args, **options):
        self.stdout.write("Bắt đầu điền dữ liệu thực tế lâm sàng (Migration 0012 profiles)...")

        # -------------------------------------------------------------------
        # 1. Departments
        # -------------------------------------------------------------------
        dept_heads = [
            ("BS.CKII Nguyễn Hồ An Thiên", "Khu B - Tầng 3, Trung tâm Thận Nhân Tạo", "(028) 3855 4137 - Ext: 301", 25),
            ("ThS.BS Trần Minh Quang", "Khu B - Tầng 2, Khoa Lọc Máu B", "(028) 3855 4137 - Ext: 302", 20),
            ("BS.CKI Lê Thị Kim Cúc", "Khu C - Tầng 1, Khoa Lọc Máu Cấp Cứu", "(028) 3855 4137 - Ext: 303", 15),
            ("ThS.BS Phạm Hoàng Dũng", "Khu A - Tầng 4, Khoa Thận - Tiết Niệu", "(028) 3855 4137 - Ext: 304", 20),
        ]
        depts = list(Department.objects.all())
        for idx, dept in enumerate(depts):
            head_info = dept_heads[idx % len(dept_heads)]
            dept.head_doctor_name = dept.head_doctor_name or head_info[0]
            dept.location_floor = dept.location_floor or head_info[1]
            dept.hotline_phone = dept.hotline_phone or head_info[2]
            dept.max_machine_capacity = dept.max_machine_capacity or head_info[3]
            dept.save()
        self.stdout.write(f"✓ Đã cập nhật hồ sơ chuyên môn cho {len(depts)} Khoa/Phòng.")

        # -------------------------------------------------------------------
        # 2. Staff (CustomUser)
        # -------------------------------------------------------------------
        staff_list = CustomUser.objects.all().order_by('id')
        doctor_titles = ["BS.CKII", "BS.CKI", "ThS.BS", "Bác sĩ Chuyên khoa"]
        nurse_titles = ["Cử nhân Điều dưỡng", "Điều dưỡng Chuyên khoa Lọc máu", "Điều dưỡng Cao đẳng"]

        for idx, user in enumerate(staff_list):
            if not user.employee_code:
                if user.role == CustomUser.Role.ADMIN:
                    user.employee_code = f"AD-{idx+1:03d}"
                    user.degree_title = user.degree_title or "Kỹ sư CNTT / Quản trị Hệ thống Y tế"
                    user.specialization = user.specialization or "Quản trị Cơ sở dữ liệu & An toàn Thông tin Bệnh viện"
                elif user.role == CustomUser.Role.DOCTOR:
                    user.employee_code = f"BS-KLM-{idx+1:02d}"
                    user.degree_title = user.degree_title or doctor_titles[idx % len(doctor_titles)]
                    user.license_number = user.license_number or f"{45810 + idx:06d}/HCM-CCHN"
                    user.specialization = user.specialization or "Thận nhân tạo - Lọc máu ngoài cơ thể & Hồi sức Nội thận"
                else:
                    user.employee_code = f"DD-KLM-{idx+1:02d}"
                    user.degree_title = user.degree_title or nurse_titles[idx % len(nurse_titles)]
                    user.license_number = user.license_number or f"{78210 + idx:06d}/HCM-CCHN"
                    user.specialization = user.specialization or "Điều dưỡng Lâm sàng Lọc máu Thận nhân tạo"

            user.gender = user.gender or ("FEMALE" if "nurse" in user.username.lower() or idx % 2 == 1 else "MALE")
            user.date_of_birth = user.date_of_birth or date(1980 + (idx % 15), (idx % 12) + 1, 10 + (idx % 18))
            user.citizen_id = user.citizen_id or f"0790{85 + (idx % 10):02d}{100000 + idx:06d}"
            hcm_districts = ["Quận 5, TP.HCM", "Quận 10, TP.HCM", "Quận 1, TP.HCM", "Quận 3, TP.HCM", "TP. Thủ Đức, TP.HCM"]
            user.address = user.address or f"{40 + idx * 7} Đường Nguyễn Chí Thanh, {hcm_districts[idx % len(hcm_districts)]}"
            user.phone_number = user.phone_number or f"090{3000000 + idx * 12345:07d}"
            user.save()
        self.stdout.write(f"✓ Đã cập nhật hồ sơ chuyên môn y tế cho {len(staff_list)} Cán bộ nhân sự.")

        # -------------------------------------------------------------------
        # 3. Patients (Patient)
        # -------------------------------------------------------------------
        patients = Patient.objects.all().order_by('patient_id')
        blood_types = ["O+", "A+", "B+", "AB+", "O-"]
        vascular_types = [
            "AVF",
            "AVG",
            "CATHETER_TUNNELED",
            "CATHETER_TEMP",
        ]
        relationships = ["Vợ/Chồng", "Con ruột", "Cha/Mẹ", "Anh/Chị/Em", "Người giám hộ"]
        street_names = [
            "124 Nguyễn Tri Phương, Phường 9, Quận 5, TP.HCM",
            "45 Lý Thường Kiệt, Phường 7, Quận 10, TP.HCM",
            "78 An Dương Vương, Phường 16, Quận 8, TP.HCM",
            "192 Hùng Vương, Phường 12, Quận 5, TP.HCM",
            "312 Trần Hưng Đạo, Phường 11, Quận 5, TP.HCM",
            "56 Ba Tháng Hai, Phường 14, Quận 10, TP.HCM",
            "89 Lê Hồng Phong, Phường 2, Quận 5, TP.HCM",
            "14 Nguyễn Trãi, Phường Bến Thành, Quận 1, TP.HCM",
            "205 Sư Vạn Hạnh, Phường 12, Quận 10, TP.HCM",
            "67 Tô Hiến Thành, Phường 13, Quận 10, TP.HCM",
        ]

        for idx, pat in enumerate(patients):
            pat.citizen_id = pat.citizen_id or f"0791{50 + (idx % 35):02d}{200000 + idx:06d}"
            bhyt_prefix = ["GD479", "HT279", "DN479", "CH479", "KC479"][idx % 5]
            pat.health_insurance_number = pat.health_insurance_number or f"{bhyt_prefix}{7000000000 + idx * 31415:010d}"
            pat.permanent_address = pat.permanent_address or street_names[idx % len(street_names)]
            pat.blood_type = pat.blood_type or blood_types[idx % len(blood_types)]
            pat.vascular_access_type = pat.vascular_access_type or vascular_types[idx % len(vascular_types)]
            pat.emergency_contact_name = pat.emergency_contact_name or f"Người nhà BN {pat.full_name}"
            pat.emergency_contact_relationship = pat.emergency_contact_relationship or relationships[idx % len(relationships)]
            pat.emergency_contact_phone = pat.emergency_contact_phone or f"091{2000000 + idx * 54321:07d}"
            pat.save()
        self.stdout.write(f"✓ Đã cập nhật hồ sơ hành chính & BHYT cho {len(patients)} Bệnh nhân.")

        # -------------------------------------------------------------------
        # 4. BloodSamples (BloodSample)
        # -------------------------------------------------------------------
        samples = BloodSample.objects.all()
        for idx, sample in enumerate(samples):
            # Realistic Hemodialysis Lab Ranges:
            # Urea: 18.0 - 32.0 mmol/L (Pre-dialysis high)
            # Creatinine: 650 - 1100 µmol/L
            # Sodium: 135 - 142 mmol/L
            # Calcium: 2.10 - 2.45 mmol/L
            # Phosphorus: 1.15 - 1.95 mmol/L
            # Albumin: 3.5 - 4.3 g/dL
            # Kt/V: 1.25 - 1.65
            random.seed(42 + idx)
            sample.urea_level = sample.urea_level or round(random.uniform(18.5, 30.5), 1)
            sample.creatinine_level = sample.creatinine_level or round(random.uniform(680.0, 1080.0), 1)
            sample.sodium_level = sample.sodium_level or round(random.uniform(136.0, 142.5), 1)
            sample.calcium_level = sample.calcium_level or round(random.uniform(2.12, 2.44), 2)
            sample.phosphorus_level = sample.phosphorus_level or round(random.uniform(1.20, 1.88), 2)
            sample.albumin_level = sample.albumin_level or round(random.uniform(3.55, 4.25), 1)
            sample.kt_v = sample.kt_v or round(random.uniform(1.28, 1.62), 2)
            sample.save()
        self.stdout.write(f"✓ Đã cập nhật 7 chỉ số sinh hóa thận cho {len(samples)} Mẫu xét nghiệm.")

        # -------------------------------------------------------------------
        # 5. DialysisSessions (DialysisSession)
        # -------------------------------------------------------------------
        sessions = DialysisSession.objects.all()
        dialyzers = [
            "High-Flux Fresenius FX60",
            "High-Flux Fresenius FX80",
            "Nipro Elisio 15H High-Flux",
            "B.Braun Diacap Pro 16H",
            "Low-Flux Fresenius F6HPS",
        ]
        anticoagulants = [
            "Heparin tiêu chuẩn (5000 IU/phiên)",
            "Heparin trọng lượng phân tử thấp (Lovenox 40mg/0.4ml)",
            "Tráng rửa NaCl 0.9% định kỳ (Không dùng Heparin - Nguy cơ xuất huyết)",
        ]

        for idx, sess in enumerate(sessions):
            random.seed(100 + idx)
            sess.blood_flow_rate = sess.blood_flow_rate or random.choice([230, 250, 280, 300, 320])
            sess.dialysate_flow_rate = sess.dialysate_flow_rate or 500
            sess.dialysate_temperature = sess.dialysate_temperature or random.choice([36.0, 36.5, 36.5, 36.5])
            sess.dialyzer_type = sess.dialyzer_type or dialyzers[idx % len(dialyzers)]
            sess.anticoagulant_type = sess.anticoagulant_type or anticoagulants[idx % len(anticoagulants)]
            sess.save()
        self.stdout.write(f"✓ Đã cập nhật thông số chỉ định kỹ thuật cho {len(sessions)} Phiên lọc máu.")

        self.stdout.write(self.style.SUCCESS("HOÀN TẤT: Toàn bộ 5 phân hệ đã được điền dữ liệu thực tế lâm sàng 100%!"))
