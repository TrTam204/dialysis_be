import random
from datetime import date, timedelta, datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import Department, Patient, DialysisMachine, DialysisSession, Shift, TreatmentPattern

class Command(BaseCommand):
    help = "Seed real-world GA demo dataset for M9 Phase 3 testing."

    def handle(self, *args, **kwargs):
        print("Bắt đầu khởi tạo dữ liệu DEMO cho Genetic Algorithm...")
        
        # 1. Tuần test
        today = timezone.localdate()
        days_ahead = 0 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_monday = today + timedelta(days=days_ahead)
        week_end = next_monday + timedelta(days=6)
        
        print(f"TEST WEEK: {next_monday.strftime('%Y-%m-%d')} -> {week_end.strftime('%Y-%m-%d')}")

        # 2. Khoa lọc máu
        dept_a, _ = Department.objects.get_or_create(code="KHOA_A", defaults={"name": "Khoa Lọc Máu A"})
        dept_b, _ = Department.objects.get_or_create(code="KHOA_B", defaults={"name": "Khoa Lọc Máu B"})
        
        print(f"TEST DEPARTMENT: {dept_a.name} (chọn Khoa này trên UI)")

        # 3. Máy lọc máu (Khoa A có 8 máy)
        # M1-M6 AVAILABLE
        machines = []
        for i in range(1, 7):
            m, _ = DialysisMachine.objects.get_or_create(
                machine_id=f"TEST_M{i}",
                defaults={
                    "name": f"Máy Lọc {i}",
                    "department": dept_a,
                    "status": DialysisMachine.Status.AVAILABLE
                }
            )
            # Ensure they are available
            m.status = DialysisMachine.Status.AVAILABLE
            m.department = dept_a
            m.save()
            machines.append(m)
            
        m7_maintenance, _ = DialysisMachine.objects.get_or_create(
            machine_id="TEST_M7",
            defaults={"name": "Máy Lọc 7 (Bảo trì)", "department": dept_a, "status": DialysisMachine.Status.MAINTENANCE}
        )
        m7_maintenance.status = DialysisMachine.Status.MAINTENANCE
        m7_maintenance.department = dept_a
        m7_maintenance.save()
        
        m8_broken, _ = DialysisMachine.objects.get_or_create(
            machine_id="TEST_M8",
            defaults={"name": "Máy Lọc 8 (Hỏng)", "department": dept_a, "status": DialysisMachine.Status.BROKEN}
        )
        m8_broken.status = DialysisMachine.Status.BROKEN
        m8_broken.department = dept_a
        m8_broken.save()
        
        # 4. Bệnh nhân
        patients_data = [
            ("P_T2_1", "Nguyen Van T2", TreatmentPattern.T2_T4_T6, Shift.SHIFT_1),
            ("P_T2_2", "Le Thi T2", TreatmentPattern.T2_T4_T6, Shift.SHIFT_2),
            ("P_T2_3", "Tran Van T2", TreatmentPattern.T2_T4_T6, Shift.SHIFT_3),
            ("P_T2_4", "Pham Thi T2", TreatmentPattern.T2_T4_T6, None),
            ("P_T2_5", "Hoang Van T2", TreatmentPattern.T2_T4_T6, Shift.SHIFT_1),
            
            ("P_T3_1", "Nguyen Van T3", TreatmentPattern.T3_T5_T7, Shift.SHIFT_1),
            ("P_T3_2", "Le Thi T3", TreatmentPattern.T3_T5_T7, Shift.SHIFT_2),
            ("P_T3_3", "Tran Van T3", TreatmentPattern.T3_T5_T7, Shift.SHIFT_3),
            ("P_T3_4", "Pham Thi T3", TreatmentPattern.T3_T5_T7, None),
            ("P_T3_5", "Hoang Van T3", TreatmentPattern.T3_T5_T7, Shift.SHIFT_2),
            
            # Một bệnh nhân discharged không nên được pick
            ("P_DIS", "Vo Thi Discharged", TreatmentPattern.T2_T4_T6, None),
            
            # Một bệnh nhân null pattern
            ("P_NULL", "Bui Van Null", None, None),
        ]
        
        for p_id, p_name, pattern, p_shift in patients_data:
            status = Patient.Status.DISCHARGED if "Discharged" in p_name else Patient.Status.ACTIVE
            p, _ = Patient.objects.get_or_create(
                patient_id=p_id,
                defaults={
                    "full_name": p_name,
                    "date_of_birth": "1990-01-01",
                }
            )
            p.status = status
            p.treatment_pattern = pattern
            p.preferred_shift = p_shift
            p.save()
            
        # 5. DialysisSession (Existing conflict)
        # Giả lập M1 đang được xếp 1 session thủ công vào Thứ 2 (Ca 1)
        conflict_date = next_monday
        conflict_start = datetime.combine(conflict_date, datetime.strptime('07:00', '%H:%M').time())
        conflict_end = datetime.combine(conflict_date, datetime.strptime('11:00', '%H:%M').time())
        
        # Make timezone aware if settings.USE_TZ is True
        from django.conf import settings
        if settings.USE_TZ:
            conflict_start = timezone.make_aware(conflict_start)
            conflict_end = timezone.make_aware(conflict_end)
        
        conflict_p = Patient.objects.get(patient_id="P_NULL")
        
        session, created = DialysisSession.objects.get_or_create(
            session_id="S_TEST_CONFLICT",
            defaults={
                "patient": conflict_p,
                "machine": machines[0], # TEST_M1
                "scheduled_start": conflict_start,
                "scheduled_end": conflict_end,
                "status": DialysisSession.Status.SCHEDULED
            }
        )
        if not created:
            session.machine = machines[0]
            session.scheduled_start = conflict_start
            session.scheduled_end = conflict_end
            session.save()
            
        print(f"CONFLICT TẠO SẴN: Máy {machines[0].name} đã có ca lọc ngày {conflict_date.strftime('%Y-%m-%d')} (Ca 1).")

        # Report stats
        print("\n=== KẾT QUẢ SEED ===")
        print(f"Departments: {Department.objects.count()}")
        print(f"Patients: {Patient.objects.count()}")
        
        t2_count = Patient.objects.filter(treatment_pattern=TreatmentPattern.T2_T4_T6).count()
        t3_count = Patient.objects.filter(treatment_pattern=TreatmentPattern.T3_T5_T7).count()
        print(f" - T2_T4_T6: {t2_count}")
        print(f" - T3_T5_T7: {t3_count}")
        
        print(f"Machines: {DialysisMachine.objects.count()}")
        m_avail = DialysisMachine.objects.filter(status=DialysisMachine.Status.AVAILABLE).count()
        m_maint = DialysisMachine.objects.filter(status=DialysisMachine.Status.MAINTENANCE).count()
        m_brok = DialysisMachine.objects.filter(status=DialysisMachine.Status.BROKEN).count()
        print(f" - AVAILABLE: {m_avail}")
        print(f" - MAINTENANCE: {m_maint}")
        print(f" - BROKEN: {m_brok}")
        
        print(f"Dialysis Sessions: {DialysisSession.objects.count()}")
        from core.models import SchedulePlan, ScheduleAssignment
        print(f"Schedule Plans: {SchedulePlan.objects.count()}")
        print(f"Schedule Assignments: {ScheduleAssignment.objects.count()}")
        print("====================")
        print("Hoàn thành seed dữ liệu. Hãy vào giao diện và chạy thuật toán GA.")
