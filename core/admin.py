from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import (
	BloodSample,
	CustomUser,
	Department,
	DialysisMachine,
	DialysisSession,
	Patient,
	VitalSign,
)


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
	list_display = ('code', 'name', 'is_active')
	list_filter = ('is_active',)
	search_fields = ('name', 'code')


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
	list_display = ('patient_id', 'full_name', 'date_of_birth', 'dry_weight', 'status')
	list_filter = ('status', 'gender')
	search_fields = ('patient_id', 'full_name', 'phone_number')


@admin.register(DialysisMachine)
class DialysisMachineAdmin(admin.ModelAdmin):
	list_display = ('machine_id', 'name', 'status', 'department', 'last_maintenance_date')
	list_filter = ('status', 'department')
	search_fields = ('machine_id', 'name')


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
	list_display = ('username', 'first_name', 'last_name', 'role', 'department', 'is_active')
	list_filter = ('role', 'department')
	search_fields = ('username', 'first_name', 'last_name', 'phone_number')
	fieldsets = UserAdmin.fieldsets + (
		('Staff details', {'fields': ('role', 'phone_number', 'department', 'avatar_url')}),
	)
	add_fieldsets = UserAdmin.add_fieldsets + (
		('Staff details', {'fields': ('role', 'phone_number', 'department', 'avatar_url')}),
	)


@admin.register(BloodSample)
class BloodSampleAdmin(admin.ModelAdmin):
	list_display = (
		'sample_id',
		'patient',
		'collection_date',
		'hemoglobin_level',
		'potassium_level',
	)
	list_filter = ('collection_date',)
	search_fields = ('sample_id', 'patient__patient_id', 'patient__full_name')


@admin.register(DialysisSession)
class DialysisSessionAdmin(admin.ModelAdmin):
	list_display = (
		'session_id',
		'patient',
		'machine',
		'assigned_nurse',
		'scheduled_start',
		'status',
	)
	list_filter = ('status', 'scheduled_start')
	search_fields = (
		'session_id',
		'patient__patient_id',
		'patient__full_name',
		'machine__machine_id',
		'assigned_nurse__username',
	)


@admin.register(VitalSign)
class VitalSignAdmin(admin.ModelAdmin):
	list_display = ('id', 'session', 'recorded_at', 'recorded_by')
	list_filter = ('recorded_at',)
	search_fields = ('session__session_id', 'recorded_by__username')
