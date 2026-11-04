import json
import math

CENTER_LAT = 10.7556
CENTER_LON = 106.6602
CENTER_NAME = "Bệnh viện Chợ Rẫy TP.HCM"
CENTER_ADDRESS = "201B Nguyễn Chí Thanh, Phường 12, Quận 5, TP.HCM"


def calculate_haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 2)


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
    MedicalMessage,
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
    department = serializers.PrimaryKeyRelatedField(queryset=Department.objects.all(), required=False, allow_null=True)
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
    department = serializers.PrimaryKeyRelatedField(
        queryset=Department.objects.all(),
        required=False,
        allow_null=True,
    )
    department_name = serializers.CharField(source='department.name', read_only=True, default=None)
    location = serializers.SerializerMethodField()
    latitude = serializers.FloatField(required=False, allow_null=True)
    longitude = serializers.FloatField(required=False, allow_null=True)
    distance_km = serializers.FloatField(read_only=True, required=False, allow_null=True)
    center_name = serializers.CharField(read_only=True, required=False)
    center_address = serializers.CharField(read_only=True, required=False)
    center_lat = serializers.FloatField(read_only=True, required=False)
    center_lon = serializers.FloatField(read_only=True, required=False)

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
            'latitude',
            'longitude',
            'distance_km',
            'center_name',
            'center_address',
            'center_lat',
            'center_lon',
            'status',
            'department',
            'department_name',
            'preferred_shift',
            'treatment_pattern',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_location(self, obj):
        if obj.location is None:
            return None
        return json.loads(obj.location.json)

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['center_name'] = CENTER_NAME
        ret['center_address'] = CENTER_ADDRESS
        ret['center_lat'] = CENTER_LAT
        ret['center_lon'] = CENTER_LON
        if instance.location:
            ret['latitude'] = round(instance.location.y, 6)
            ret['longitude'] = round(instance.location.x, 6)
            ret['distance_km'] = calculate_haversine_km(instance.location.y, instance.location.x, CENTER_LAT, CENTER_LON)
        else:
            ret['latitude'] = None
            ret['longitude'] = None
            ret['distance_km'] = None
        return ret

    def validate(self, attrs):
        lat = attrs.get('latitude')
        lng = attrs.get('longitude')
        if lat is not None and not (-90.0 <= lat <= 90.0):
            raise serializers.ValidationError({'latitude': ['Vĩ độ phải nằm trong khoảng [-90.0, 90.0].']})
        if lng is not None and not (-180.0 <= lng <= 180.0):
            raise serializers.ValidationError({'longitude': ['Kinh độ phải nằm trong khoảng [-180.0, 180.0].']})
        if (lat is not None and lng is None) or (lat is None and lng is not None):
            raise serializers.ValidationError({'location': ['Cần cung cấp đồng thời cả vĩ độ (latitude) và kinh độ (longitude).']})
        validate_patient_data(attrs, self.instance)

        request = self.context.get('request')
        user = getattr(request, 'user', None) if request else None
        is_admin = bool(user and user.is_authenticated and user.role == CustomUser.Role.ADMIN)

        # 1. CREATE rules (Milestone 1B Part 2)
        if self.instance is None:
            if not is_admin:
                user_dept = getattr(user, 'department', None) if user else None
                if not user_dept or not getattr(user_dept, 'is_active', False):
                    raise serializers.ValidationError({
                        'department': ['Tài khoản người dùng không thuộc khoa nào đang hoạt động.']
                    })

                req_dept = attrs.get('department')
                if req_dept is not None and req_dept != user_dept:
                    raise serializers.ValidationError({
                        'department': ['Bác sĩ chỉ được tạo bệnh nhân thuộc khoa của mình.']
                    })

                # Tự động gán khoa của người tạo
                attrs['department'] = user_dept
            else:
                req_dept = attrs.get('department')
                if req_dept is not None and not req_dept.is_active:
                    raise serializers.ValidationError({
                        'department': ['Khoa đã ngừng hoạt động không thể tiếp nhận bệnh nhân mới.']
                    })
                # Nếu ADMIN không truyền department, giữ nguyên None cho tương thích chuyển tiếp

        # 2. UPDATE / PATCH rules (Milestone 1B)
        else:
            if 'department' in attrs:
                new_dept = attrs['department']
                old_dept = self.instance.department
                if new_dept != old_dept:
                    if not is_admin:
                        raise serializers.ValidationError({
                            'department': ['Chỉ ADMIN mới có quyền gán khoa cho bệnh nhân.']
                        })

                    if old_dept is not None:
                        raise serializers.ValidationError({
                            'department': [
                                'Không thể thay đổi khoa đã gán qua cập nhật thông thường. '
                                'Việc chuyển khoa phải được thực hiện qua quy trình chuyển khoa (transfer).'
                            ]
                        })

                    if new_dept is not None and not new_dept.is_active:
                        raise serializers.ValidationError({
                            'department': ['Khoa đã ngừng hoạt động không thể tiếp nhận bệnh nhân mới.']
                        })

        return attrs

    def create(self, validated_data):
        lat = validated_data.pop('latitude', None)
        lng = validated_data.pop('longitude', None)
        if lat is not None and lng is not None:
            from django.contrib.gis.geos import Point
            validated_data['location'] = Point(float(lng), float(lat), srid=4326)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        has_lat = 'latitude' in validated_data
        has_lng = 'longitude' in validated_data
        lat = validated_data.pop('latitude', None)
        lng = validated_data.pop('longitude', None)
        if has_lat and has_lng:
            if lat is not None and lng is not None:
                from django.contrib.gis.geos import Point
                instance.location = Point(float(lng), float(lat), srid=4326)
            else:
                instance.location = None
        return super().update(instance, validated_data)


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
            'schedule_assignment',
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
            'is_emergency',
            'emergency_details',
            'notes',
            'created_at',
            'updated_at',
        ]
        # Legacy BP fields are kept read-only: VitalSign is the primary data source now.
        # schedule_assignment is managed only through the approval workflow, not manual session API.
        read_only_fields = [
            'created_at',
            'updated_at',
            'schedule_assignment',
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
    actor_username = serializers.CharField(source='actor.username', read_only=True, default=None)
    actor_role = serializers.CharField(source='actor.role', read_only=True, default=None)
    user_full_name = serializers.SerializerMethodField()
    username = serializers.CharField(source='actor.username', read_only=True, default=None)
    user_role = serializers.CharField(source='actor.role', read_only=True, default=None)

    class Meta:
        model = AuditLog
        fields = [
            'id',
            'actor',
            'actor_name',
            'actor_username',
            'actor_role',
            'user_full_name',
            'username',
            'user_role',
            'action',
            'entity_type',
            'entity_id',
            'changes',
            'timestamp',
        ]
        read_only_fields = fields

    def get_actor_name(self, obj):
        return _user_display(obj.actor)

    def get_user_full_name(self, obj):
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
            'rejection_reason',
            'assignments_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'status',
            'approved_by',
            'approved_at',
            'rejection_reason',
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


class MedicalMessageSerializer(serializers.ModelSerializer):
    sender = serializers.PrimaryKeyRelatedField(read_only=True)
    sender_name = serializers.SerializerMethodField()
    sender_role = serializers.CharField(source='sender.role', read_only=True)
    receiver_name = serializers.SerializerMethodField()
    recipient = serializers.PrimaryKeyRelatedField(source='receiver', queryset=CustomUser.objects.all(), required=False, allow_null=True)
    recipient_name = serializers.SerializerMethodField()
    department_name = serializers.CharField(source='department.name', read_only=True)
    session_code = serializers.CharField(source='session_reference.session_id', read_only=True)
    content = serializers.CharField(required=False, allow_blank=True, default='')
    attachment_file = serializers.FileField(required=False, allow_null=True)
    attachment_type = serializers.CharField(max_length=20, required=False, allow_blank=True, allow_null=True)
    attachment_name = serializers.CharField(max_length=255, required=False, allow_blank=True, allow_null=True)
    attachment_url = serializers.SerializerMethodField()
    metadata = serializers.JSONField(required=False, default=dict)
    read_by_users = serializers.JSONField(read_only=True)

    class Meta:
        model = MedicalMessage
        fields = [
            'id',
            'sender',
            'sender_name',
            'sender_role',
            'receiver',
            'receiver_name',
            'recipient',
            'recipient_name',
            'department',
            'department_name',
            'content',
            'session_reference',
            'session_code',
            'is_read',
            'read_by_users',
            'attachment_file',
            'attachment_type',
            'attachment_name',
            'attachment_url',
            'metadata',
            'created_at',
        ]
        read_only_fields = ['id', 'sender', 'department', 'created_at', 'read_by_users', 'attachment_url']

    def to_internal_value(self, data):
        if hasattr(data, 'dict'):
            data = data.dict()
        else:
            data = dict(data.items()) if hasattr(data, 'items') else dict(data)

        if 'recipient' in data and 'receiver' not in data:
            data['receiver'] = data['recipient']
        elif 'receiver' in data and 'recipient' not in data:
            data['recipient'] = data['receiver']

        # Parse metadata if it was passed as a JSON string (e.g. from FormData)
        if 'metadata' in data and isinstance(data['metadata'], str):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except Exception:
                data['metadata'] = {}

        return super().to_internal_value(data)

    def get_sender_name(self, obj):
        return f"{obj.sender.last_name} {obj.sender.first_name}".strip() or obj.sender.username

    def get_receiver_name(self, obj):
        if not obj.receiver:
            return 'Toàn khoa'
        return f"{obj.receiver.last_name} {obj.receiver.first_name}".strip() or obj.receiver.username

    def get_recipient_name(self, obj):
        return self.get_receiver_name(obj)

    def get_attachment_url(self, obj):
        if obj.attachment_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.attachment_file.url)
            return obj.attachment_file.url
        return None

