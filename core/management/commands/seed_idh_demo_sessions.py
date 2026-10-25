"""
Management command to seed 3 active IN_PROGRESS hemodialysis sessions in KHOA_A with
chronological VitalSign trajectories representing 3 clinical scenarios:
1. High Risk IDH (Severe SBP drop, compensatory tachycardia, high UFR)
2. Medium Risk IDH (Moderate SBP drop, borderline UFR)
3. Low Risk IDH (Stable hemodynamics, normal UFR)
"""
import sys
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from core.models import (
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    VitalSign,
)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class Command(BaseCommand):
    help = "Seed 3 IN_PROGRESS hemodialysis sessions in KHOA_A with vital signs for IDH testing."

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("=== SEEDING 3 IN_PROGRESS SESSIONS FOR IDH CLINICAL DEMONSTRATION ==="))

        with transaction.atomic():
            dept_a = Department.objects.filter(code='KHOA_A').first()
            if not dept_a:
                self.stderr.write("Department KHOA_A not found. Please run seed_clinical_kmeans_data first.")
                return

            nurse1 = CustomUser.objects.filter(username='nurse1').first()
            doctor1 = CustomUser.objects.filter(username='doctor1').first()

            # Ensure machines exist in KHOA_A
            m1, _ = DialysisMachine.objects.get_or_create(
                machine_id='M_KHOA_A_1',
                defaults={
                    'name': 'Máy Lọc Máu A-01 (Fresenius 4008S)',
                    'status': DialysisMachine.Status.IN_USE,
                    'department': dept_a,
                },
            )
            m2, _ = DialysisMachine.objects.get_or_create(
                machine_id='M_KHOA_A_2',
                defaults={
                    'name': 'Máy Lọc Máu A-02 (Fresenius 5008S)',
                    'status': DialysisMachine.Status.IN_USE,
                    'department': dept_a,
                },
            )
            m3, _ = DialysisMachine.objects.get_or_create(
                machine_id='M_KHOA_A_3',
                defaults={
                    'name': 'Máy Lọc Máu A-03 (Nikkiso DBB-07)',
                    'status': DialysisMachine.Status.IN_USE,
                    'department': dept_a,
                },
            )

            # Get 3 patients in KHOA_A
            p_high = Patient.objects.filter(department=dept_a, patient_id='P_A_01').first()
            if not p_high:
                p_high = Patient.objects.filter(department=dept_a).first()
            p_high.dry_weight = 48.0
            p_high.save(update_fields=['dry_weight'])

            p_med = Patient.objects.filter(department=dept_a, patient_id='P_A_02').first()
            if not p_med:
                p_med = Patient.objects.filter(department=dept_a).exclude(pk=p_high.pk).first()
            p_med.dry_weight = 58.0
            p_med.save(update_fields=['dry_weight'])

            p_low = Patient.objects.filter(department=dept_a, patient_id='P_A_03').first()
            if not p_low:
                p_low = Patient.objects.filter(department=dept_a).exclude(pk__in=[p_high.pk, p_med.pk]).first()
            p_low.dry_weight = 65.0
            p_low.save(update_fields=['dry_weight'])

            now = timezone.now()
            start_time = now - timedelta(minutes=95)

            # =========================================================================
            # SCENARIO 1: HIGH RISK (SS_INP_HIGH_01)
            # Patient dry weight: 48kg, UF Target: 3.0L -> UFR = 15.6 mL/kg/h (>13.0)
            # SBP drops: 132 -> 122 -> 110 -> 98 mmHg (Delta SBP = +34 mmHg)
            # HR increases: 74 -> 80 -> 90 -> 98 bpm (Compensatory tachycardia)
            # =========================================================================
            s_high, _ = DialysisSession.objects.update_or_create(
                session_id='SS_INP_HIGH_01',
                defaults={
                    'patient': p_high,
                    'machine': m1,
                    'assigned_nurse': nurse1,
                    'status': DialysisSession.Status.IN_PROGRESS,
                    'scheduled_start': start_time,
                    'scheduled_end': start_time + timedelta(hours=4),
                    'actual_start': start_time,
                    'pre_weight': 51.0,
                    'uf_target': 3.0,
                    'uf_actual': 1.15,
                    'clinical_notes': 'Bệnh nhân than mệt mỏi, vã mồ hôi nhẹ ở phút 85.',
                },
            )
            s_high.vital_signs.all().delete()
            high_readings = [
                (0, 132, 82, 74, 36.6, 'Bắt đầu ca lọc ổn định'),
                (30, 122, 76, 80, 36.5, 'Bệnh nhân tỉnh táo'),
                (60, 110, 68, 90, 36.5, 'Huyết áp bắt đầu có xu hướng giảm nhanh'),
                (90, 98, 60, 98, 36.4, 'Huyết áp giảm sâu, mạch nhanh, cảnh báo tiền triệu IDH'),
            ]
            for t_offset, sbp, dbp, hr, temp, note in high_readings:
                VitalSign.objects.create(
                    session=s_high,
                    recorded_by=nurse1 or doctor1,
                    recorded_at=start_time + timedelta(minutes=t_offset),
                    systolic_bp=sbp,
                    diastolic_bp=dbp,
                    heart_rate=hr,
                    temperature=temp,
                    notes=note,
                )
            self.stdout.write(self.style.SUCCESS(
                f"  [1/3] Created HIGH RISK Session: {s_high.session_id} for {p_high.full_name} "
                f"(Dry Weight={p_high.dry_weight}kg, UFR=15.6 mL/kg/h, SBP: 132 -> 98 mmHg)"
            ))

            # =========================================================================
            # SCENARIO 2: MEDIUM RISK (SS_INP_MED_02)
            # Patient dry weight: 58kg, UF Target: 2.4L -> UFR = 10.3 mL/kg/h
            # SBP drops moderately: 130 -> 125 -> 119 -> 114 mmHg (Delta SBP = +16 mmHg)
            # HR: 72 -> 74 -> 77 -> 80 bpm
            # =========================================================================
            s_med, _ = DialysisSession.objects.update_or_create(
                session_id='SS_INP_MED_02',
                defaults={
                    'patient': p_med,
                    'machine': m2,
                    'assigned_nurse': nurse1,
                    'status': DialysisSession.Status.IN_PROGRESS,
                    'scheduled_start': start_time,
                    'scheduled_end': start_time + timedelta(hours=4),
                    'actual_start': start_time,
                    'pre_weight': 60.5,
                    'uf_target': 2.4,
                    'uf_actual': 0.90,
                    'clinical_notes': 'Theo dõi huyết áp định kỳ, đáp ứng lọc vừa.',
                },
            )
            s_med.vital_signs.all().delete()
            med_readings = [
                (0, 130, 80, 72, 36.6, 'Bắt đầu ca lọc'),
                (30, 125, 78, 74, 36.6, 'Sinh hiệu ổn định'),
                (60, 119, 75, 77, 36.5, 'Huyết áp giảm nhẹ 11 mmHg'),
                (90, 114, 72, 80, 36.5, 'Huyết áp giảm vừa phải (16 mmHg), nhịp tim tăng nhẹ'),
            ]
            for t_offset, sbp, dbp, hr, temp, note in med_readings:
                VitalSign.objects.create(
                    session=s_med,
                    recorded_by=nurse1 or doctor1,
                    recorded_at=start_time + timedelta(minutes=t_offset),
                    systolic_bp=sbp,
                    diastolic_bp=dbp,
                    heart_rate=hr,
                    temperature=temp,
                    notes=note,
                )
            self.stdout.write(self.style.SUCCESS(
                f"  [2/3] Created MEDIUM RISK Session: {s_med.session_id} for {p_med.full_name} "
                f"(Dry Weight={p_med.dry_weight}kg, UFR=10.3 mL/kg/h, SBP: 130 -> 114 mmHg)"
            ))

            # =========================================================================
            # SCENARIO 3: LOW RISK (SS_INP_LOW_03)
            # Patient dry weight: 65kg, UF Target: 1.8L -> UFR = 6.9 mL/kg/h (< 10.0)
            # SBP stable with normal physiological drift: 125 -> 123 -> 122 -> 117 mmHg
            # HR: 72 -> 73 -> 74 -> 75 bpm
            # =========================================================================
            s_low, _ = DialysisSession.objects.update_or_create(
                session_id='SS_INP_LOW_03',
                defaults={
                    'patient': p_low,
                    'machine': m3,
                    'assigned_nurse': nurse1,
                    'status': DialysisSession.Status.IN_PROGRESS,
                    'scheduled_start': start_time,
                    'scheduled_end': start_time + timedelta(hours=4),
                    'actual_start': start_time,
                    'pre_weight': 67.0,
                    'uf_target': 1.8,
                    'uf_actual': 0.70,
                    'clinical_notes': 'Người bệnh dung nạp tốt, sinh hiệu ổn định.',
                },
            )
            s_low.vital_signs.all().delete()
            low_readings = [
                (0, 125, 78, 72, 36.7, 'Khởi động ca lọc an toàn'),
                (30, 123, 77, 73, 36.6, 'Huyết áp bình thường'),
                (60, 122, 76, 74, 36.6, 'Không có biến cố huyết động'),
                (90, 117, 74, 75, 36.6, 'Dung nạp tốt, huyết áp dao động sinh lý nhẹ'),
            ]
            for t_offset, sbp, dbp, hr, temp, note in low_readings:
                VitalSign.objects.create(
                    session=s_low,
                    recorded_by=nurse1 or doctor1,
                    recorded_at=start_time + timedelta(minutes=t_offset),
                    systolic_bp=sbp,
                    diastolic_bp=dbp,
                    heart_rate=hr,
                    temperature=temp,
                    notes=note,
                )
            self.stdout.write(self.style.SUCCESS(
                f"  [3/3] Created LOW RISK Session: {s_low.session_id} for {p_low.full_name} "
                f"(Dry Weight={p_low.dry_weight}kg, UFR=6.9 mL/kg/h, SBP: 125 -> 117 mmHg)"
            ))

        self.stdout.write(self.style.MIGRATE_HEADING("=== SEEDING COMPLETED SUCCESSFULLY ==="))
