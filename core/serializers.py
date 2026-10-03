import json

from rest_framework import serializers

from .models import (
    AuditLog,
    BloodSample,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    Patient,
    ScheduleAssignment,
    SchedulePlan,
    VitalSign,
)
from .services import (
    validate_blood_sample_data,
    validate_machine_data,
    validate_patient_data,
    validate_session_data,
    validate_vital_sign_data,
)


def _user_display(user):
    if user is None:
        return None
    full_name = f'{user.last_name} {user.first_name}'.strip()
    return full_name or user.username


class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = ['id', 'name', 'code', 'description', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class CustomUserSerializer(serializers.ModelSerializer):
    department = serializers.PrimaryKeyRelatedField(queryset=Department.objects.all())
    department_name = serializers.CharField(source='department.name', read_only=True)
    password = serializers.CharField(write_only=True, required=False, trim_whitespace=False)

    class Meta:
        model = CustomUser
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'role',
            'phone_number',
            'department',
            'department_name',
            'is_active',
            'avatar_url',
            'password',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate(self, attrs):
        if self.instance is None and not attrs.get('password'):
            raise serializers.ValidationError({'password': ['Password is required when creating a staff account.']})
        return attrs

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        user = super().update(instance, validated_data)
        if password:
            user.set_password(password)
            user.save()
        return user


class PatientSerializer(serializers.ModelSerializer):
    location = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
            'patient_id',
            'full_name',
            'date_of_birth',
            'gender',
            'phone_number',
            'medical_history',
            'dry_weight',
            'location',
            'status',
            'preferred_shift',
            'treatment_pattern',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_location(self, obj):
        # Phase 1: expose the PostGIS point as GeoJSON, read-only.
        if obj.location is None:
            return None
        return json.loads(obj.location.json)

    def validate(self, attrs):
        validate_patient_data(attrs, self.instance)
        return attrs


class DialysisMachineSerializer(serializers.ModelSerializer):
    department = serializers.PrimaryKeyRelatedField(queryset=Department.objects.all())
    department_name = serializers.CharField(source='department.name', read_only=True)

    class Meta:
        model = DialysisMachine
        fields = [
            'machine_id',
            'name',
            'status',
            'maintenance_log',
            'last_maintenance_date',
            'department',
            'department_name',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def validate(self, attrs):
        validate_machine_data(attrs, self.instance)
        return attrs


class DialysisSessionSerializer(serializers.ModelSerializer):
    patient = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all())
    machine = serializers.PrimaryKeyRelatedField(queryset=DialysisMachine.objects.all())
    assigned_nurse = serializers.PrimaryKeyRelatedField(
        queryset=CustomUser.objects.filter(role=CustomUser.Role.NURSE),
        required=False,
        allow_null=True,
    )
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    patient_code = serializers.CharField(source='patient.patient_id', read_only=True)
    machine_name = serializers.CharField(source='machine.name', read_only=True)
    patient_dry_weight = serializers.FloatField(source='patient.dry_weight', read_only=True)
    nurse_name = serializers.SerializerMethodField()

    class Meta:
        model = DialysisSession
        fields = [
            'session_id',
            'patient',
            'patient_name',
            'patient_code',
            'patient_dry_weight',
            'machine',
            'machine_name',
            'assigned_nurse',
            'nurse_name',
            'scheduled_start',
            'scheduled_end',
            'actual_start',
            'actual_end',
            'pre_weight',
            'post_weight',
            'uf_target',
            'uf_actual',
            'clinical_notes',
            'pre_dialysis_bp',
            'during_dialysis_bp',
            'post_dialysis_bp',
            'status',
            'notes',
            'created_at',
            'updated_at',
        ]
        # Legacy BP fields are kept read-only: VitalSign is the primary data source now.
        read_only_fields = [
            'created_at',
            'updated_at',
            'pre_dialysis_bp',
            'during_dialysis_bp',
            'post_dialysis_bp',
        ]

    def get_nurse_name(self, obj):
        return _user_display(obj.assigned_nurse)

    def validate(self, attrs):
        validate_session_data(attrs, self.instance)
        return attrs


class BloodSampleSerializer(serializers.ModelSerializer):
    patient = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all())
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    patient_code = serializers.CharField(source='patient.patient_id', read_only=True)
    created_by = serializers.PrimaryKeyRelatedField(read_only=True)
    created_by_name = serializers.SerializerMethodField()

    class Meta:
        model = BloodSample
        fields = [
            'sample_id',
            'patient',
            'patient_name',
            'patient_code',
            'collection_date',
            'hemoglobin_level',
            'potassium_level',
            'notes',
            'created_by',
            'created_by_name',
            'created_at',
        ]
        read_only_fields = ['created_by', 'created_at']

    def get_created_by_name(self, obj):
        return _user_display(obj.created_by)

    def validate(self, attrs):
        validate_blood_sample_data(attrs, self.instance)
        return attrs


class VitalSignSerializer(serializers.ModelSerializer):
    session = serializers.PrimaryKeyRelatedField(queryset=DialysisSession.objects.all(), required=False)
    recorded_by = serializers.PrimaryKeyRelatedField(read_only=True)
    recorded_by_name = serializers.SerializerMethodField()

    class Meta:
        model = VitalSign
        fields = [
            'id',
            'session',
            'recorded_by',
            'recorded_by_name',
            'recorded_at',
            'systolic_bp',
            'diastolic_bp',
            'heart_rate',
            'spo2',
            'temperature',
            'notes',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_recorded_by_name(self, obj):
        return _user_display(obj.recorded_by)

    def validate(self, attrs):
        if self.instance is None:
            session = attrs.get('session') if 'session' in attrs else self.context.get('session')
            if session is None:
                raise serializers.ValidationError({'session': ['session is required.']})
            attrs['session'] = session
        validate_vital_sign_data(attrs)
        return attrs


class AuditLogSerializer(serializers.ModelSerializer):
    actor_name = serializers.SerializerMethodField()
    actor_role = serializers.CharField(source='actor.role', read_only=True, default=None)

    class Meta:
        model = AuditLog
        fields = [
            'id',
            'actor',
            'actor_name',
            'actor_role',
            'action',
            'entity_type',
            'entity_id',
            'changes',
            'timestamp',
        ]
        read_only_fields = fields

    def get_actor_name(self, obj):
        return _user_display(obj.actor)


class SchedulePlanSerializer(serializers.ModelSerializer):
    created_by_name = serializers.SerializerMethodField()
    approved_by_name = serializers.SerializerMethodField()
    department_name = serializers.CharField(source='department.name', read_only=True)
    assignments_count = serializers.IntegerField(source='assignments.count', read_only=True)

    class Meta:
        model = SchedulePlan
        fields = [
            'id',
            'name',
            'department',
            'department_name',
            'week_start',
            'week_end',
            'status',
            'fitness_score',
            'algorithm_metadata',
            'created_by',
            'created_by_name',
            'approved_by',
            'approved_by_name',
            'approved_at',
            'assignments_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'approved_by',
            'approved_at',
            'created_at',
            'updated_at',
            'assignments_count',
        ]

    def get_created_by_name(self, obj):
        return _user_display(obj.created_by)

    def get_approved_by_name(self, obj):
        return _user_display(obj.approved_by)

    def validate(self, attrs):
        week_start = attrs.get('week_start') or getattr(self.instance, 'week_start', None)
        week_end = attrs.get('week_end') or getattr(self.instance, 'week_end', None)
        if week_start and week_end and week_end < week_start:
            raise serializers.ValidationError({'week_end': ['week_end must be on or after week_start.']})
        return attrs


class ScheduleAssignmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.full_name', read_only=True)
    patient_code = serializers.CharField(source='patient.patient_id', read_only=True)
    machine_name = serializers.CharField(source='machine.name', read_only=True)

    class Meta:
        model = ScheduleAssignment
        fields = [
            'id',
            'schedule_plan',
            'patient',
            'patient_name',
            'patient_code',
            'machine',
            'machine_name',
            'scheduled_date',
            'shift',
            'start_datetime',
            'end_datetime',
            'source',
            'original_machine',
            'original_shift',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate(self, attrs):
        start_datetime = attrs.get('start_datetime') or getattr(self.instance, 'start_datetime', None)
        end_datetime = attrs.get('end_datetime') or getattr(self.instance, 'end_datetime', None)
        if start_datetime and end_datetime and end_datetime <= start_datetime:
            raise serializers.ValidationError({'end_datetime': ['end_datetime must be after start_datetime.']})

        # When updating: if machine or shift changed, mark source as MANUAL
        if self.instance is not None:
            new_machine = attrs.get('machine')
            new_shift = attrs.get('shift')
            if (new_machine and new_machine != self.instance.machine) or (new_shift and new_shift != self.instance.shift):
                attrs['source'] = ScheduleAssignment.Source.MANUAL

            # original_machine and original_shift are IMMUTABLE once created
            if 'original_machine' in attrs and attrs['original_machine'] != self.instance.original_machine:
                attrs.pop('original_machine')
            if 'original_shift' in attrs and attrs['original_shift'] != self.instance.original_shift:
                attrs.pop('original_shift')
        else:
            # On creation: default original_machine/shift from initial machine/shift
            if 'original_machine' not in attrs:
                attrs['original_machine'] = attrs.get('machine')
            if 'original_shift' not in attrs:
                attrs['original_shift'] = attrs.get('shift')

        return attrs
