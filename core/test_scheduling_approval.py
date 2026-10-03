from datetime import date, datetime, timedelta
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import (
    AuditLog,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    ScheduleAssignment,
    SchedulePlan,
    Shift,
    TreatmentPattern,
)
from core.services import (
    approve_schedule_plan,
    generate_dialysis_session_id,
    reject_schedule_plan,
    validate_schedule_plan_for_approval,
)


class SchedulingApprovalTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dept = Department.objects.create(
            name='Khoa Than Loc Mau M9 Phase 2',
            code='K-M9P2',
            description='Khoa test M9 Phase 2 Approval',
        )
        cls.admin_user = CustomUser.objects.create_user(
            username='admin_appr',
            email='admin_appr@example.com',
            password='Password123!',
            role=CustomUser.Role.ADMIN,
            department=cls.dept,
        )
        cls.doctor_user = CustomUser.objects.create_user(
            username='doctor_appr',
            email='doctor_appr@example.com',
            password='Password123!',
            role=CustomUser.Role.DOCTOR,
            department=cls.dept,
        )
        cls.nurse_user = CustomUser.objects.create_user(
            username='nurse_appr',
            email='nurse_appr@example.com',
            password='Password123!',
            role=CustomUser.Role.NURSE,
            department=cls.dept,
        )

        cls.machine1 = DialysisMachine.objects.create(
            machine_id='M-APP-01',
            name='Machine Approval 01',
            status='AVAILABLE',
            department=cls.dept,
        )
        cls.machine2 = DialysisMachine.objects.create(
            machine_id='M-APP-02',
            name='Machine Approval 02',
            status='AVAILABLE',
            department=cls.dept,
        )

        # 2026-10-12 is Monday (weekday 0)
        # 2026-10-14 is Wednesday (weekday 2)
        # 2026-10-16 is Friday (weekday 4)
        cls.week_start = date(2026, 10, 12)
        cls.week_end = date(2026, 10, 18)

        cls.patient1 = Patient.objects.create(
            patient_id='P-APP-01',
            full_name='Nguyen Van Valid',
            date_of_birth='1980-05-15',
            gender='MALE',
            status='ACTIVE',
            preferred_shift='SHIFT_1',
            treatment_pattern='T2_T4_T6',
        )
        cls.patient2 = Patient.objects.create(
            patient_id='P-APP-02',
            full_name='Tran Thi Valid 2',
            date_of_birth='1985-08-20',
            gender='FEMALE',
            status='IN_TREATMENT',
            preferred_shift='SHIFT_2',
            treatment_pattern='T2_T4_T6',
        )

    def _create_valid_plan_with_assignments(self, name='Plan Valid'):
        plan = SchedulePlan.objects.create(
            name=name,
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
            created_by=self.doctor_user,
        )
        d_mon = date(2026, 10, 12)
        d_wed = date(2026, 10, 14)
        # Monday assignment
        asgn1 = ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=self.machine1,
            scheduled_date=d_mon,
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0, 0)),
        )
        # Wednesday assignment
        asgn2 = ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient2,
            machine=self.machine2,
            scheduled_date=d_wed,
            shift=Shift.SHIFT_2,
            start_datetime=timezone.make_aware(datetime(2026, 10, 14, 12, 30, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 14, 16, 30, 0)),
        )
        return plan, [asgn1, asgn2]

    # Test 1: approve valid plan -> APPROVED
    def test_01_approve_valid_plan_status_approved(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 01 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        response = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        plan.refresh_from_db()
        self.assertEqual(plan.status, SchedulePlan.Status.APPROVED)
        self.assertEqual(plan.approved_by, self.doctor_user)
        self.assertIsNotNone(plan.approved_at)

    # Test 2: tao dung so DialysisSession
    def test_02_approve_valid_plan_creates_exact_session_count(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 02 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        response = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        sessions = DialysisSession.objects.filter(schedule_assignment__schedule_plan=plan)
        self.assertEqual(sessions.count(), 2)

    # Test 3: session_id auto-generated and <= 20 chars
    def test_03_session_id_auto_generated_length(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 03 Plan')
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        sessions = DialysisSession.objects.filter(schedule_assignment__schedule_plan=plan)
        for s in sessions:
            self.assertTrue(s.session_id.startswith('SES-202610'))
            self.assertLessEqual(len(s.session_id), 20)

    # Test 4: schedule_assignment FK points to correct assignment
    def test_04_schedule_assignment_fk_links_properly(self):
        plan, assignments = self._create_valid_plan_with_assignments('Test 04 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/approve/')

        for asgn in assignments:
            session = DialysisSession.objects.filter(schedule_assignment=asgn).first()
            self.assertIsNotNone(session)
            self.assertEqual(session.schedule_assignment, asgn)

    # Test 5: patient/machine/time mapping dung
    def test_05_field_mapping_patient_machine_time(self):
        plan, assignments = self._create_valid_plan_with_assignments('Test 05 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/approve/')

        for asgn in assignments:
            session = DialysisSession.objects.get(schedule_assignment=asgn)
            self.assertEqual(session.patient, asgn.patient)
            self.assertEqual(session.machine, asgn.machine)
            self.assertEqual(session.scheduled_start, asgn.start_datetime)
            self.assertEqual(session.scheduled_end, asgn.end_datetime)
            self.assertEqual(session.status, DialysisSession.Status.SCHEDULED)

    # Test 6: assigned_nurse is NULL
    def test_06_assigned_nurse_is_null_on_created_sessions(self):
        plan, assignments = self._create_valid_plan_with_assignments('Test 06 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/approve/')

        for asgn in assignments:
            session = DialysisSession.objects.get(schedule_assignment=asgn)
            self.assertIsNone(session.assigned_nurse)

    # Test 7: reject valid plan -> REJECTED
    def test_07_reject_valid_plan(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 07 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        response = self.client.post(
            f'/api/schedule-plans/{plan.id}/reject/',
            data={'reason': 'Bac si yeu cau dieu chinh lai khung gio.'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        plan.refresh_from_db()
        self.assertEqual(plan.status, SchedulePlan.Status.REJECTED)
        self.assertEqual(plan.rejection_reason, 'Bac si yeu cau dieu chinh lai khung gio.')
        # No sessions created
        self.assertEqual(DialysisSession.objects.filter(schedule_assignment__schedule_plan=plan).count(), 0)

    # Test 8: reject without reason -> fail (400)
    def test_08_reject_without_reason_fails(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 08 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        response = self.client.post(f'/api/schedule-plans/{plan.id}/reject/', data={'reason': '   '}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('reason', response.data)
        plan.refresh_from_db()
        self.assertEqual(plan.status, SchedulePlan.Status.PROPOSED)

    # Test 9: cannot approve APPROVED plan
    def test_09_cannot_approve_already_approved(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 09 Plan')
        self.client.force_authenticate(user=self.admin_user)
        res1 = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res1.status_code, status.HTTP_200_OK)

        # Second approve attempt
        res2 = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res2.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 10: cannot reject APPROVED plan
    def test_10_cannot_reject_already_approved(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 10 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/approve/')

        res = self.client.post(f'/api/schedule-plans/{plan.id}/reject/', data={'reason': 'Huy lich'}, format='json')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 11: cannot approve REJECTED plan
    def test_11_cannot_approve_rejected_plan(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 11 Plan')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/reject/', data={'reason': 'Tu choi'}, format='json')

        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 12: empty assignment plan -> fail
    def test_12_empty_assignment_plan_fails_approval(self):
        plan = SchedulePlan.objects.create(
            name='Empty Plan',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 13: DISCHARGED patient -> fail
    def test_13_discharged_patient_fails_approval(self):
        plan = SchedulePlan.objects.create(
            name='Plan with Discharged Patient',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        p_disc = Patient.objects.create(
            patient_id='P-DISC-01',
            full_name='Discharged Patient',
            date_of_birth='1970-01-01',
            status='DISCHARGED',
            treatment_pattern='T2_T4_T6',
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=p_disc,
            machine=self.machine1,
            scheduled_date=date(2026, 10, 12),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 14: missing treatment_pattern -> fail
    def test_14_missing_treatment_pattern_fails_approval(self):
        plan = SchedulePlan.objects.create(
            name='Plan Missing Pattern',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        p_nopat = Patient.objects.create(
            patient_id='P-NOPAT-01',
            full_name='No Pattern Patient',
            date_of_birth='1975-01-01',
            status='ACTIVE',
            treatment_pattern=None,
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=p_nopat,
            machine=self.machine1,
            scheduled_date=date(2026, 10, 12),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 15: invalid treatment pattern weekday -> fail
    def test_15_invalid_treatment_pattern_weekday_fails(self):
        plan = SchedulePlan.objects.create(
            name='Plan Invalid Weekday',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        # patient1 is T2_T4_T6, but scheduled on Tuesday (2026-10-13)
        d_tue = date(2026, 10, 13)
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=self.machine1,
            scheduled_date=d_tue,
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 13, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 13, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 16: BROKEN machine -> fail
    def test_16_broken_machine_fails_approval(self):
        plan = SchedulePlan.objects.create(
            name='Plan Broken Machine',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        m_broken = DialysisMachine.objects.create(
            machine_id='M-BROKEN-01',
            name='Broken Machine',
            status='BROKEN',
            department=self.dept,
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=m_broken,
            scheduled_date=date(2026, 10, 12),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 17: MAINTENANCE machine -> fail
    def test_17_maintenance_machine_fails_approval(self):
        plan = SchedulePlan.objects.create(
            name='Plan Maintenance Machine',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        m_maint = DialysisMachine.objects.create(
            machine_id='M-MAINT-01',
            name='Maint Machine',
            status='MAINTENANCE',
            department=self.dept,
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=m_maint,
            scheduled_date=date(2026, 10, 12),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 18: machine overlap with existing DialysisSession -> fail
    def test_18_machine_overlap_fails_approval(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 18 Machine Overlap')
        # Create an existing active session on machine1 on Monday
        DialysisSession.objects.create(
            session_id='SS-EXISTING-MACH',
            patient=self.patient2,
            machine=self.machine1,
            scheduled_start=timezone.make_aware(datetime(2026, 10, 12, 8, 0)),
            scheduled_end=timezone.make_aware(datetime(2026, 10, 12, 12, 0)),
            status=DialysisSession.Status.SCHEDULED,
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 19: patient overlap with existing DialysisSession -> fail
    def test_19_patient_overlap_fails_approval(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 19 Patient Overlap')
        # Create an existing active session for patient1 on Monday on machine2
        DialysisSession.objects.create(
            session_id='SS-EXISTING-PAT',
            patient=self.patient1,
            machine=self.machine2,
            scheduled_start=timezone.make_aware(datetime(2026, 10, 12, 8, 0)),
            scheduled_end=timezone.make_aware(datetime(2026, 10, 12, 12, 0)),
            status=DialysisSession.Status.SCHEDULED,
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 20: CANCELLED existing session does not block
    def test_20_cancelled_session_does_not_block_approval(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 20 Cancelled Not Block')
        DialysisSession.objects.create(
            session_id='SS-CANCELLED-01',
            patient=self.patient1,
            machine=self.machine1,
            scheduled_start=timezone.make_aware(datetime(2026, 10, 12, 8, 0)),
            scheduled_end=timezone.make_aware(datetime(2026, 10, 12, 12, 0)),
            status=DialysisSession.Status.CANCELLED,
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_200_OK)

    # Test 21: preferred_shift mismatch does not block (soft constraint, generates warning)
    def test_21_preferred_shift_mismatch_generates_warning_only(self):
        plan = SchedulePlan.objects.create(
            name='Plan Shift Mismatch',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        # patient1 prefers SHIFT_1, but we schedule on SHIFT_2
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=self.machine1,
            scheduled_date=date(2026, 10, 12),
            shift=Shift.SHIFT_2,
            start_datetime=timezone.make_aware(datetime(2026, 10, 12, 12, 30)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 12, 16, 30)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertTrue(len(res.data.get('warnings', [])) > 0)
        plan.refresh_from_db()
        self.assertEqual(plan.status, SchedulePlan.Status.APPROVED)

    # Test 22: start >= end -> fail
    def test_22_start_ge_end_fails_validation(self):
        class MockAssignment:
            patient_id = self.patient1.patient_id
            machine_id = self.machine1.machine_id
            scheduled_date = date(2026, 10, 12)
            shift = Shift.SHIFT_1
            start_datetime = timezone.make_aware(datetime(2026, 10, 12, 11, 0))
            end_datetime = timezone.make_aware(datetime(2026, 10, 12, 7, 0))
            patient = self.patient1
            machine = self.machine1

        asgn = MockAssignment()
        mock_plan = type('MockPlan', (), {
            'status': SchedulePlan.Status.PROPOSED,
            'week_start': self.week_start,
            'week_end': self.week_end,
            'assignments': type('MockRel', (), {
                'select_related': lambda *a: type('MockQS', (), {'all': lambda self: [asgn]})()
            })(),
        })()
        val = validate_schedule_plan_for_approval(mock_plan)
        self.assertFalse(val['is_valid'])
        self.assertTrue(any('invalid time window' in e for e in val['errors']))

    # Test 23: assignment outside plan week -> fail
    def test_23_assignment_outside_plan_week_fails(self):
        plan = SchedulePlan.objects.create(
            name='Plan Out of Week',
            department=self.dept,
            week_start=self.week_start,
            week_end=self.week_end,
            status=SchedulePlan.Status.PROPOSED,
        )
        # Date is 2026-10-19 (next Monday, outside week 12-18)
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient1,
            machine=self.machine1,
            scheduled_date=date(2026, 10, 19),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 19, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 19, 11, 0)),
        )
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    # Test 24: transaction rollback if one assignment fails
    def test_24_transaction_rollback_on_failure(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 24 Rollback')
        # Add an invalid assignment (broken machine)
        m_broken = DialysisMachine.objects.create(
            machine_id='M-ROLLBACK-BROK',
            name='Broken Mach Rollback',
            status='BROKEN',
            department=self.dept,
        )
        ScheduleAssignment.objects.create(
            schedule_plan=plan,
            patient=self.patient2,
            machine=m_broken,
            scheduled_date=date(2026, 10, 16),
            shift=Shift.SHIFT_1,
            start_datetime=timezone.make_aware(datetime(2026, 10, 16, 7, 0)),
            end_datetime=timezone.make_aware(datetime(2026, 10, 16, 11, 0)),
        )
        initial_session_count = DialysisSession.objects.count()
        self.client.force_authenticate(user=self.doctor_user)
        res = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

        # Confirm rollback: no sessions created, plan remains PROPOSED
        self.assertEqual(DialysisSession.objects.count(), initial_session_count)
        plan.refresh_from_db()
        self.assertEqual(plan.status, SchedulePlan.Status.PROPOSED)

    # Test 25: audit approval
    def test_25_audit_logged_on_approval(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 25 Audit Appr')
        self.client.force_authenticate(user=self.admin_user)
        self.client.post(f'/api/schedule-plans/{plan.id}/approve/')

        audit = AuditLog.objects.filter(entity_type='SchedulePlan', entity_id=str(plan.id)).first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.actor, self.admin_user)
        self.assertEqual(audit.changes['status']['after'], 'APPROVED')

    # Test 26: audit rejection
    def test_26_audit_logged_on_rejection(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 26 Audit Rej')
        self.client.force_authenticate(user=self.doctor_user)
        self.client.post(
            f'/api/schedule-plans/{plan.id}/reject/',
            data={'reason': 'Ly do kiem toan audit.'},
            format='json',
        )
        audit = AuditLog.objects.filter(entity_type='SchedulePlan', entity_id=str(plan.id)).first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.actor, self.doctor_user)
        self.assertEqual(audit.changes['status']['after'], 'REJECTED')
        self.assertEqual(audit.changes['rejection_reason'], 'Ly do kiem toan audit.')

    # RBAC Test: NURSE cannot approve or reject (403)
    def test_27_nurse_cannot_approve_or_reject(self):
        plan, _ = self._create_valid_plan_with_assignments('Test 27 RBAC Nurse')
        self.client.force_authenticate(user=self.nurse_user)
        res_appr = self.client.post(f'/api/schedule-plans/{plan.id}/approve/')
        self.assertEqual(res_appr.status_code, status.HTTP_403_FORBIDDEN)

        res_rej = self.client.post(f'/api/schedule-plans/{plan.id}/reject/', data={'reason': 'Nurse try'}, format='json')
        self.assertEqual(res_rej.status_code, status.HTTP_403_FORBIDDEN)
