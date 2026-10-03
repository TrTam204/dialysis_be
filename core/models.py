from django.contrib.auth.models import AbstractUser
from django.contrib.gis.db import models


class Department(models.Model):
    name = models.CharField(max_length=120, unique=True)
    code = models.CharField(max_length=30, unique=True)
    description = models.TextField(blank=True, default='')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['is_active'], name='idx_department_is_active'),
        ]

    def save(self, *args, **kwargs):
        if self.code:
            self.code = self.code.upper()
        super().save(*args, **kwargs)

    def __str__(self):
        return f'{self.code} - {self.name}'


class Shift(models.TextChoices):
    SHIFT_1 = 'SHIFT_1', 'Ca 1'
    SHIFT_2 = 'SHIFT_2', 'Ca 2'
    SHIFT_3 = 'SHIFT_3', 'Ca 3'


class TreatmentPattern(models.TextChoices):
    T2_T4_T6 = 'T2_T4_T6', 'Thứ 2 - Thứ 4 - Thứ 6'
    T3_T5_T7 = 'T3_T5_T7', 'Thứ 3 - Thứ 5 - Thứ 7'


class Patient(models.Model):
    class Status(models.TextChoices):
        ACTIVE = 'ACTIVE', 'Active'
        IN_TREATMENT = 'IN_TREATMENT', 'In treatment'
        STABLE = 'STABLE', 'Stable'
        DISCHARGED = 'DISCHARGED', 'Discharged'

    patient_id = models.CharField(max_length=20, primary_key=True)
    full_name = models.CharField(max_length=255)
    date_of_birth = models.DateField()
    gender = models.CharField(max_length=20, choices=[('MALE', 'Male'), ('FEMALE', 'Female'), ('OTHER', 'Other')], blank=True, default='OTHER')
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    medical_history = models.TextField(blank=True, default='')
    dry_weight = models.FloatField(blank=True, null=True)
    location = models.PointField(srid=4326, blank=True, null=True)
    status = models.CharField(max_length=20, choices=Status.choices, default='ACTIVE')
    preferred_shift = models.CharField(
        max_length=20,
        choices=Shift.choices,
        null=True,
        blank=True,
        default=None,
    )
    treatment_pattern = models.CharField(
        max_length=20,
        choices=TreatmentPattern.choices,
        null=True,
        blank=True,
        default=None,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['full_name'], name='idx_patient_name'),
            models.Index(fields=['status'], name='idx_patient_status'),
            models.Index(fields=['date_of_birth'], name='idx_patient_dob'),
        ]

    def __str__(self):
        return f'{self.patient_id} - {self.full_name}'


class DialysisMachine(models.Model):
    class Status(models.TextChoices):
        AVAILABLE = 'AVAILABLE', 'Available'
        IN_USE = 'IN_USE', 'In use'
        MAINTENANCE = 'MAINTENANCE', 'Maintenance'
        BROKEN = 'BROKEN', 'Broken'

    machine_id = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    status = models.CharField(max_length=20, choices=Status.choices, default='AVAILABLE')
    maintenance_log = models.TextField(blank=True, default='')
    last_maintenance_date = models.DateField(blank=True, null=True)
    department = models.ForeignKey(Department, on_delete=models.PROTECT, related_name='machines')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['status'], name='idx_machine_status'),
        ]

    def __str__(self):
        return f'{self.machine_id} - {self.name}'


class CustomUser(AbstractUser):
    class Role(models.TextChoices):
        ADMIN = 'ADMIN', 'Admin'
        DOCTOR = 'DOCTOR', 'Doctor'
        NURSE = 'NURSE', 'Nurse'

    email = models.EmailField(unique=True)
    role = models.CharField(max_length=10, choices=Role.choices, default=Role.NURSE)
    phone_number = models.CharField(max_length=20, null=True, blank=True)
    department = models.ForeignKey(Department, on_delete=models.PROTECT, related_name='staff')
    avatar_url = models.URLField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['role'], name='idx_user_role'),
        ]
        verbose_name = 'user'
        verbose_name_plural = 'users'

    def __str__(self):
        return f'{self.username} ({self.role})'


class BloodSample(models.Model):
    sample_id = models.CharField(max_length=20, primary_key=True)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.PROTECT,
        related_name='blood_samples',
    )
    collection_date = models.DateTimeField()
    hemoglobin_level = models.FloatField()
    potassium_level = models.FloatField()
    notes = models.TextField(blank=True, default='')
    created_by = models.ForeignKey(
        'CustomUser',
        on_delete=models.PROTECT,
        related_name='created_blood_samples',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['collection_date'], name='idx_bloodsample_col_date'),
        ]

    def __str__(self):
        return f'{self.sample_id} - {self.patient.full_name}'


class DialysisSession(models.Model):
    class Status(models.TextChoices):
        SCHEDULED = 'SCHEDULED', 'Scheduled'
        IN_PROGRESS = 'IN_PROGRESS', 'In progress'
        COMPLETED = 'COMPLETED', 'Completed'
        CANCELLED = 'CANCELLED', 'Cancelled'

    session_id = models.CharField(max_length=20, primary_key=True)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.PROTECT,
        related_name='dialysis_sessions',
    )
    machine = models.ForeignKey(
        DialysisMachine,
        on_delete=models.PROTECT,
        related_name='dialysis_sessions',
    )
    assigned_nurse = models.ForeignKey(
        'CustomUser',
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='assigned_dialysis_sessions',
        limit_choices_to={'role': CustomUser.Role.NURSE},
    )
    schedule_assignment = models.ForeignKey(
        'ScheduleAssignment',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='dialysis_sessions',
    )
    scheduled_start = models.DateTimeField()
    scheduled_end = models.DateTimeField()
    # Legacy summary fields kept per database-spec; VitalSign is the primary source now.
    pre_dialysis_bp = models.CharField(max_length=20, blank=True, default='')
    during_dialysis_bp = models.CharField(max_length=20, blank=True, default='')
    post_dialysis_bp = models.CharField(max_length=20, blank=True, default='')
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SCHEDULED)
    actual_start = models.DateTimeField(null=True, blank=True)
    actual_end = models.DateTimeField(null=True, blank=True)
    pre_weight = models.FloatField(null=True, blank=True)
    post_weight = models.FloatField(null=True, blank=True)
    uf_target = models.FloatField(null=True, blank=True)
    uf_actual = models.FloatField(null=True, blank=True)
    clinical_notes = models.TextField(blank=True, default='')
    notes = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['status'], name='idx_session_status'),
            models.Index(fields=['scheduled_start', 'scheduled_end'], name='idx_session_time_range'),
        ]

    def __str__(self):
        return self.session_id


class VitalSign(models.Model):
    session = models.ForeignKey(
        DialysisSession,
        on_delete=models.CASCADE,
        related_name='vital_signs',
    )
    recorded_by = models.ForeignKey(
        'CustomUser',
        on_delete=models.PROTECT,
        related_name='recorded_vital_signs',
    )
    recorded_at = models.DateTimeField()
    systolic_bp = models.IntegerField(null=True, blank=True)
    diastolic_bp = models.IntegerField(null=True, blank=True)
    heart_rate = models.IntegerField(null=True, blank=True)
    spo2 = models.IntegerField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['recorded_at'], name='idx_vital_recorded_at'),
        ]

    def __str__(self):
        return f'VitalSign for {self.session_id}'


class AuditLog(models.Model):
    class Action(models.TextChoices):
        CREATE = 'CREATE', 'Create'
        UPDATE = 'UPDATE', 'Update'
        DELETE = 'DELETE', 'Delete'

    actor = models.ForeignKey(
        'CustomUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_logs',
    )
    action = models.CharField(max_length=20, choices=Action.choices)
    entity_type = models.CharField(max_length=50)
    entity_id = models.CharField(max_length=64)
    changes = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['entity_type'], name='idx_auditlog_entity_type'),
            models.Index(fields=['action'], name='idx_auditlog_action'),
            models.Index(fields=['timestamp'], name='idx_auditlog_timestamp'),
            models.Index(fields=['entity_type', 'entity_id'], name='idx_auditlog_entity_lookup'),
        ]

    def __str__(self):
        actor_name = self.actor.username if self.actor else 'System'
        return f'[{self.timestamp:%Y-%m-%d %H:%M:%S}] {actor_name} {self.action} {self.entity_type}#{self.entity_id}'


class SchedulePlan(models.Model):
    class Status(models.TextChoices):
        PROPOSED = 'PROPOSED', 'Proposed'
        APPROVED = 'APPROVED', 'Approved'
        REJECTED = 'REJECTED', 'Rejected'

    name = models.CharField(max_length=120, blank=True, default='')
    department = models.ForeignKey(
        Department,
        on_delete=models.PROTECT,
        related_name='schedule_plans',
    )
    week_start = models.DateField()
    week_end = models.DateField()
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PROPOSED,
    )
    fitness_score = models.FloatField(null=True, blank=True)
    algorithm_metadata = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey(
        'CustomUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_schedule_plans',
    )
    approved_by = models.ForeignKey(
        'CustomUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='approved_schedule_plans',
    )
    approved_at = models.DateTimeField(null=True, blank=True)
    rejection_reason = models.TextField(
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-week_start', '-created_at']
        indexes = [
            models.Index(fields=['department', 'week_start', 'status'], name='idx_plan_dept_week_status'),
            models.Index(fields=['status'], name='idx_plan_status'),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['department', 'week_start'],
                condition=models.Q(status='APPROVED'),
                name='uq_approved_plan_per_dept_week',
            ),
        ]

    def __str__(self):
        return f'{self.name or "SchedulePlan"} [{self.week_start} to {self.week_end}] - {self.status}'


class ScheduleAssignment(models.Model):
    class Source(models.TextChoices):
        GA = 'GA', 'GA'
        MANUAL = 'MANUAL', 'Manual'

    schedule_plan = models.ForeignKey(
        SchedulePlan,
        on_delete=models.CASCADE,
        related_name='assignments',
    )
    patient = models.ForeignKey(
        Patient,
        on_delete=models.PROTECT,
        related_name='schedule_assignments',
    )
    machine = models.ForeignKey(
        DialysisMachine,
        on_delete=models.PROTECT,
        related_name='schedule_assignments',
    )
    scheduled_date = models.DateField()
    shift = models.CharField(max_length=20, choices=Shift.choices)
    start_datetime = models.DateTimeField()
    end_datetime = models.DateTimeField()
    source = models.CharField(
        max_length=20,
        choices=Source.choices,
        default=Source.GA,
    )
    original_machine = models.ForeignKey(
        DialysisMachine,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
    )
    original_shift = models.CharField(
        max_length=20,
        choices=Shift.choices,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['scheduled_date', 'shift', 'machine']
        indexes = [
            models.Index(fields=['schedule_plan', 'scheduled_date'], name='idx_asgn_plan_date'),
            models.Index(fields=['patient', 'scheduled_date'], name='idx_asgn_patient_date'),
            models.Index(fields=['machine', 'scheduled_date', 'shift'], name='idx_asgn_machine_date_shift'),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['schedule_plan', 'machine', 'scheduled_date', 'shift'],
                name='uq_asgn_plan_machine_date_shift',
            ),
            models.UniqueConstraint(
                fields=['schedule_plan', 'patient', 'scheduled_date'],
                name='uq_asgn_plan_patient_date',
            ),
            models.CheckConstraint(
                check=models.Q(start_datetime__lt=models.F('end_datetime')),
                name='chk_asgn_start_before_end',
            ),
        ]

    def __str__(self):
        return f'{self.patient_id} @ {self.machine_id} [{self.scheduled_date} {self.shift}]'

