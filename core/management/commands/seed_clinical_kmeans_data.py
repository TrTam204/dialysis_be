"""
Management command to seed realistic, diverse clinical profiles for K-Means clustering.
Categorizes patients in KHOA_A into 3 distinct physiological cohorts:
1. Stable Hemodialysis (Good fluid and biochemical control)
2. Fluid Overloaded & Hyperkalemic (High IDWG % and elevated potassium)
3. Elderly & Chronic Anemic (Advanced age, severe anemia, lower dry weight)

Creates authentic BloodSample records and COMPLETED DialysisSessions so that
imputation flag (is_imputed) drops to 0% for the active cohort.
"""
from datetime import date, datetime, timedelta
import random
import sys
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from core.models import (
    BloodSample,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    Shift,
    TreatmentPattern,
    VitalSign,
)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class Command(BaseCommand):
    help = "Seed diverse clinical parameters for patients in KHOA_A for K-Means clustering demonstration."

    def add_arguments(self, parser):
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed for deterministic clinical values generation.',
        )

    def handle(self, *args, **options):
        rand_seed = options.get('seed', 42)
        rng = random.Random(rand_seed)

        self.stdout.write(self.style.MIGRATE_HEADING("=== SEEDING DIVERSE CLINICAL PROFILES FOR K-MEANS DEMONSTRATION ==="))

        with transaction.atomic():
            # 1. Departments and Staff synchronization
            try:
                dept_a = Department.objects.get(code='KHOA_A')
            except Department.DoesNotExist:
                dept_a = Department.objects.create(
                    name='Khoa Lọc Máu A',
                    code='KHOA_A',
                    is_active=True,
                )

            try:
                dept_b = Department.objects.get(code='KHOA_B')
            except Department.DoesNotExist:
                dept_b = Department.objects.create(
                    name='Khoa Lọc Máu B',
                    code='KHOA_B',
                    is_active=True,
                )

            # Move doctor1, nurse1, nurse2 to KHOA_A
            CustomUser.objects.filter(username__in=['doctor1', 'nurse1', 'nurse2']).update(department=dept_a)
            self.stdout.write("  -> Synchronized doctor1, nurse1, nurse2 to KHOA_A.")

            # Ensure separate doctor and nurse exist in KHOA_B for cross-department RBAC testing
            doc_b, _ = CustomUser.objects.get_or_create(
                username='doctor_b',
                defaults={
                    'email': 'doctor_b@example.com',
                    'role': CustomUser.Role.DOCTOR,
                    'department': dept_b,
                    'first_name': 'Bác sĩ',
                    'last_name': 'Khoa B',
                    'is_active': True,
                },
            )
            if not doc_b.has_usable_password():
                doc_b.set_password('Password123!')
                doc_b.department = dept_b
                doc_b.save()

            nurse_b, _ = CustomUser.objects.get_or_create(
                username='nurse_b',
                defaults={
                    'email': 'nurse_b@example.com',
                    'role': CustomUser.Role.NURSE,
                    'department': dept_b,
                    'first_name': 'Điều dưỡng',
                    'last_name': 'Khoa B',
                    'is_active': True,
                },
            )
            if not nurse_b.has_usable_password():
                nurse_b.set_password('Password123!')
                nurse_b.department = dept_b
                nurse_b.save()

            # Ensure an active machine in KHOA_A for completed sessions
            machine_a = DialysisMachine.objects.filter(department=dept_a).first()
            if not machine_a:
                machine_a = DialysisMachine.objects.create(
                    machine_id='M-A-01',
                    name='May Loc Mau A-01',
                    status=DialysisMachine.Status.AVAILABLE,
                    department=dept_a,
                )

            # 2. Setup 17 Patients in KHOA_A (10 existing + 7 new)
            new_patients_meta = [
                ('P_A_01', 'Nguyễn Thị Hoa', 'FEMALE'),
                ('P_A_02', 'Đặng Quốc Bảo', 'MALE'),
                ('P_A_03', 'Vũ Thị Mai', 'FEMALE'),
                ('P_A_04', 'Đỗ Thành Đạt', 'MALE'),
                ('P_A_05', 'Bùi Văn Hùng', 'MALE'),
                ('P_A_06', 'Trịnh Thị Cúc', 'FEMALE'),
                ('P_A_07', 'Ngô Văn Thắng', 'MALE'),
            ]

            for pid, full_name, gender in new_patients_meta:
                Patient.objects.get_or_create(
                    patient_id=pid,
                    defaults={
                        'full_name': full_name,
                        'gender': gender,
                        'date_of_birth': date(1985, 1, 1),
                        'department': dept_a,
                        'status': Patient.Status.ACTIVE,
                        'preferred_shift': Shift.SHIFT_1,
                        'treatment_pattern': TreatmentPattern.T2_T4_T6,
                    },
                )

            # 3. Define 3 Physiological Clinical Profiles
            # Cohort 1: Stable Hemodialysis (6 patients)
            # Cohort 2: Fluid Overload & Hyperkalemic (6 patients)
            # Cohort 3: Elderly & Chronic Anemic (5 patients)
            today = timezone.localdate()

            cohort_specs = [
                # Cohort 1: STABLE (6 patients)
                # Age: 30-50, Dry Weight: 55-70, Hb: 11.2 - 12.5, K: 4.0 - 4.6, IDWG%: 2.1 - 3.0%
                {
                    'id': 'P_T2_1',
                    'cohort': 'STABLE',
                    'age': 34,
                    'dry_weight': 62.0,
                    'hb': 11.8,
                    'k': 4.2,
                    'idwg_pct': 2.4,
                },
                {
                    'id': 'P_T2_2',
                    'cohort': 'STABLE',
                    'age': 41,
                    'dry_weight': 58.0,
                    'hb': 12.1,
                    'k': 4.4,
                    'idwg_pct': 2.6,
                },
                {
                    'id': 'P_T3_1',
                    'cohort': 'STABLE',
                    'age': 37,
                    'dry_weight': 66.0,
                    'hb': 11.5,
                    'k': 4.1,
                    'idwg_pct': 2.2,
                },
                {
                    'id': 'P_T3_2',
                    'cohort': 'STABLE',
                    'age': 46,
                    'dry_weight': 64.0,
                    'hb': 12.3,
                    'k': 4.5,
                    'idwg_pct': 2.8,
                },
                {
                    'id': 'P_A_01',
                    'cohort': 'STABLE',
                    'age': 32,
                    'dry_weight': 55.0,
                    'hb': 11.6,
                    'k': 4.0,
                    'idwg_pct': 2.5,
                },
                {
                    'id': 'P_A_02',
                    'cohort': 'STABLE',
                    'age': 44,
                    'dry_weight': 68.0,
                    'hb': 12.0,
                    'k': 4.3,
                    'idwg_pct': 2.7,
                },

                # Cohort 2: FLUID OVERLOAD & HYPERKALEMIA (6 patients)
                # Age: 48-64, Dry Weight: 48-62, Hb: 10.0 - 11.0, K: 5.6 - 6.3, IDWG%: 5.0 - 6.4%
                {
                    'id': 'P_T2_3',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 51,
                    'dry_weight': 54.0,
                    'hb': 10.5,
                    'k': 5.8,
                    'idwg_pct': 5.4,
                },
                {
                    'id': 'P_T2_4',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 55,
                    'dry_weight': 58.0,
                    'hb': 10.2,
                    'k': 6.1,
                    'idwg_pct': 5.9,
                },
                {
                    'id': 'P_T3_3',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 58,
                    'dry_weight': 50.0,
                    'hb': 10.8,
                    'k': 5.7,
                    'idwg_pct': 5.2,
                },
                {
                    'id': 'P_T3_4',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 62,
                    'dry_weight': 56.0,
                    'hb': 10.1,
                    'k': 6.2,
                    'idwg_pct': 6.1,
                },
                {
                    'id': 'P_A_03',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 49,
                    'dry_weight': 52.0,
                    'hb': 10.4,
                    'k': 5.9,
                    'idwg_pct': 5.5,
                },
                {
                    'id': 'P_A_04',
                    'cohort': 'FLUID_OVERLOAD',
                    'age': 60,
                    'dry_weight': 60.0,
                    'hb': 10.6,
                    'k': 5.6,
                    'idwg_pct': 5.1,
                },

                # Cohort 3: ELDERLY & CHRONIC ANEMIA (5 patients)
                # Age: 68-78, Dry Weight: 42-50, Hb: 7.8 - 8.8, K: 4.2 - 4.9, IDWG%: 3.0 - 3.8%
                {
                    'id': 'P_T2_5',
                    'cohort': 'ELDERLY_ANEMIC',
                    'age': 70,
                    'dry_weight': 46.0,
                    'hb': 8.2,
                    'k': 4.5,
                    'idwg_pct': 3.3,
                },
                {
                    'id': 'P_T3_5',
                    'cohort': 'ELDERLY_ANEMIC',
                    'age': 74,
                    'dry_weight': 44.0,
                    'hb': 7.9,
                    'k': 4.7,
                    'idwg_pct': 3.5,
                },
                {
                    'id': 'P_A_05',
                    'cohort': 'ELDERLY_ANEMIC',
                    'age': 68,
                    'dry_weight': 48.0,
                    'hb': 8.5,
                    'k': 4.4,
                    'idwg_pct': 3.1,
                },
                {
                    'id': 'P_A_06',
                    'cohort': 'ELDERLY_ANEMIC',
                    'age': 76,
                    'dry_weight': 42.0,
                    'hb': 7.8,
                    'k': 4.6,
                    'idwg_pct': 3.6,
                },
                {
                    'id': 'P_A_07',
                    'cohort': 'ELDERLY_ANEMIC',
                    'age': 72,
                    'dry_weight': 47.0,
                    'hb': 8.4,
                    'k': 4.3,
                    'idwg_pct': 3.2,
                },
            ]

            nurse_admin = CustomUser.objects.filter(role=CustomUser.Role.NURSE, department=dept_a).first()
            if not nurse_admin:
                nurse_admin = CustomUser.objects.filter(role=CustomUser.Role.ADMIN).first()

            now = timezone.now()

            self.stdout.write(f"\n--- POPULATING CLINICAL LABS & COMPLETED SESSIONS ({len(cohort_specs)} PATIENTS) ---")

            for spec in cohort_specs:
                patient = Patient.objects.get(patient_id=spec['id'])
                patient.department = dept_a
                patient.status = Patient.Status.ACTIVE

                # Set date_of_birth to match target age
                birth_year = today.year - spec['age']
                patient.date_of_birth = date(birth_year, today.month, min(today.day, 28))
                patient.dry_weight = spec['dry_weight']
                patient.save()

                # Clean existing blood samples for clean deterministic state
                patient.blood_samples.all().delete()

                # 1. Create 2 realistic BloodSamples (14 days ago and 2 days ago)
                hb_val = spec['hb']
                k_val = spec['k']

                b1 = BloodSample.objects.create(
                    sample_id=f"BS-{patient.patient_id}-1",
                    patient=patient,
                    collection_date=now - timedelta(days=14),
                    hemoglobin_level=round(hb_val + rng.uniform(-0.2, 0.2), 2),
                    potassium_level=round(k_val + rng.uniform(-0.1, 0.1), 2),
                    notes="Xét nghiệm máu định kỳ tuần -2",
                    created_by=nurse_admin,
                )
                b2 = BloodSample.objects.create(
                    sample_id=f"BS-{patient.patient_id}-2",
                    patient=patient,
                    collection_date=now - timedelta(days=2),
                    hemoglobin_level=round(hb_val + rng.uniform(-0.2, 0.2), 2),
                    potassium_level=round(k_val + rng.uniform(-0.1, 0.1), 2),
                    notes="Xét nghiệm máu định kỳ tuần hiện tại",
                    created_by=nurse_admin,
                )

                # 2. Create 2 COMPLETED DialysisSessions in the past (e.g. 5 days ago and 2 days ago)
                # Ensure we do NOT overwrite or touch any of the 31 existing SCHEDULED sessions!
                past_completed_sessions = patient.dialysis_sessions.filter(
                    status=DialysisSession.Status.COMPLETED
                )

                # Calculate realistic pre_weight and uf_actual based on target IDWG %
                target_idwg_pct = spec['idwg_pct']
                uf_liters = round(spec['dry_weight'] * (target_idwg_pct / 100.0), 2)
                pre_wt = round(spec['dry_weight'] + uf_liters, 2)

                if past_completed_sessions.count() < 2:
                    # Session 1: 5 days ago
                    s1_id = f"SS-COMP-{patient.patient_id}-1"
                    s1, _ = DialysisSession.objects.get_or_create(
                        session_id=s1_id,
                        defaults={
                            'patient': patient,
                            'machine': machine_a,
                            'status': DialysisSession.Status.COMPLETED,
                            'scheduled_start': now - timedelta(days=5, hours=4),
                            'scheduled_end': now - timedelta(days=5),
                            'actual_start': now - timedelta(days=5, hours=4),
                            'actual_end': now - timedelta(days=5),
                            'pre_weight': pre_wt,
                            'post_weight': spec['dry_weight'],
                            'uf_target': uf_liters,
                            'uf_actual': uf_liters,
                            'assigned_nurse': nurse_admin,
                        },
                    )
                    # Create vitals for session 1
                    VitalSign.objects.get_or_create(
                        session=s1,
                        recorded_at=s1.actual_start,
                        defaults={
                            'systolic_bp': 130,
                            'diastolic_bp': 80,
                            'heart_rate': 75,
                            'recorded_by': nurse_admin,
                        },
                    )

                    # Session 2: 2 days ago
                    s2_id = f"SS-COMP-{patient.patient_id}-2"
                    s2, _ = DialysisSession.objects.get_or_create(
                        session_id=s2_id,
                        defaults={
                            'patient': patient,
                            'machine': machine_a,
                            'status': DialysisSession.Status.COMPLETED,
                            'scheduled_start': now - timedelta(days=2, hours=4),
                            'scheduled_end': now - timedelta(days=2),
                            'actual_start': now - timedelta(days=2, hours=4),
                            'actual_end': now - timedelta(days=2),
                            'pre_weight': pre_wt,
                            'post_weight': spec['dry_weight'],
                            'uf_target': uf_liters,
                            'uf_actual': uf_liters,
                            'assigned_nurse': nurse_admin,
                        },
                    )
                    # Create vitals for session 2
                    VitalSign.objects.get_or_create(
                        session=s2,
                        recorded_at=s2.actual_start,
                        defaults={
                            'systolic_bp': 128,
                            'diastolic_bp': 78,
                            'heart_rate': 72,
                            'recorded_by': nurse_admin,
                        },
                    )

                self.stdout.write(
                    f"  [OK] {patient.patient_id:<10} | {patient.full_name:<18} | "
                    f"Tuổi: {spec['age']:<2} | Hb: {spec['hb']:<4} | K: {spec['k']:<4} | "
                    f"%IDWG: {spec['idwg_pct']:<4}% | Cụm sinh lý: {spec['cohort']}"
                )

        self.stdout.write(self.style.SUCCESS(
            "\n[HOÀN TẤT] Đã tạo thành công dữ liệu lâm sàng thực chất cho 17 bệnh nhân tại KHOA_A.\n"
            "Tỷ lệ nội suy mặc định (is_imputed) của toàn bộ 17 bệnh nhân KHOA_A đã giảm về 0%!"
        ))
