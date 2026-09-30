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
        related_name='assigned_dialysis_sessions',
        limit_choices_to={'role': CustomUser.Role.NURSE},
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
