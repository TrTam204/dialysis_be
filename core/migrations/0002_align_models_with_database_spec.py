# Hand-written migration: aligns the database schema with the current models
# (database-spec.md Phase 1). It creates Department and VitalSign, adds the
# missing columns, and repairs existing rows (role / department backfill)
# before altering CustomUser.department from a plain CharField to a real
# ForeignKey, so no existing row is lost.

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


def backfill_customuser_role_and_department(apps, schema_editor):
    Department = apps.get_model('core', 'Department')
    CustomUser = apps.get_model('core', 'CustomUser')

    default_dept = Department.objects.filter(code='KLM').first()
    if default_dept is None:
        default_dept = Department.objects.create(
            name='Khoa Lọc Máu',
            code='KLM',
            description='Khoa mặc định được tạo tự động bởi migration để đáp ứng ràng buộc department bắt buộc.',
            is_active=True,
        )

    valid_roles = {'ADMIN', 'DOCTOR', 'NURSE'}
    for user in CustomUser.objects.all().iterator():
        changed = False
        if user.role not in valid_roles:
            user.role = 'ADMIN'
            changed = True
        if not user.department:
            user.department = str(default_dept.pk)
            changed = True
        if changed:
            user.save()


class Migration(migrations.Migration):

    initial = False

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Department',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=120, unique=True)),
                ('code', models.CharField(max_length=30, unique=True)),
                ('description', models.TextField(blank=True, default='')),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'indexes': [models.Index(fields=['is_active'], name='idx_department_is_active')],
            },
        ),
        # Repair existing rows BEFORE altering CustomUser.department CharField -> FK.
        migrations.RunPython(backfill_customuser_role_and_department, lambda apps, schema_editor: None),
        migrations.AlterField(
            model_name='customuser',
            name='department',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='staff', to='core.department'),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='email',
            field=models.EmailField(max_length=254, unique=True),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='role',
            field=models.CharField(choices=[('ADMIN', 'Admin'), ('DOCTOR', 'Doctor'), ('NURSE', 'Nurse')], default='NURSE', max_length=10),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='phone_number',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='customuser',
            name='avatar_url',
            field=models.URLField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='customuser',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='customuser',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
            preserve_default=False,
        ),
        migrations.AddIndex(
            model_name='customuser',
            index=models.Index(fields=['role'], name='idx_user_role'),
        ),
        # --- DialysisMachine ---
        migrations.AlterField(
            model_name='dialysismachine',
            name='machine_id',
            field=models.CharField(max_length=20, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='dialysismachine',
            name='name',
            field=models.CharField(max_length=100, unique=True),
        ),
        migrations.AlterField(
            model_name='dialysismachine',
            name='status',
            field=models.CharField(choices=[('AVAILABLE', 'Available'), ('IN_USE', 'In use'), ('MAINTENANCE', 'Maintenance'), ('BROKEN', 'Broken')], default='AVAILABLE', max_length=20),
        ),
        migrations.AlterField(
            model_name='dialysismachine',
            name='maintenance_log',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='dialysismachine',
            name='last_maintenance_date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dialysismachine',
            name='department',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, related_name='machines', to='core.department'),
        ),
        migrations.AddField(
            model_name='dialysismachine',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='dialysismachine',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='dialysismachine',
            name='department',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='machines', to='core.department'),
        ),
        migrations.AddIndex(
            model_name='dialysismachine',
            index=models.Index(fields=['status'], name='idx_machine_status'),
        ),
        # --- Patient ---
        migrations.AlterField(
            model_name='patient',
            name='patient_id',
            field=models.CharField(max_length=20, primary_key=True, serialize=False),
        ),
        migrations.AddField(
            model_name='patient',
            name='gender',
            field=models.CharField(blank=True, choices=[('MALE', 'Male'), ('FEMALE', 'Female'), ('OTHER', 'Other')], default='OTHER', max_length=20),
        ),
        migrations.AddField(
            model_name='patient',
            name='phone_number',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='patient',
            name='status',
            field=models.CharField(choices=[('ACTIVE', 'Active'), ('IN_TREATMENT', 'In treatment'), ('STABLE', 'Stable'), ('DISCHARGED', 'Discharged')], default='ACTIVE', max_length=20),
        ),
        migrations.AlterField(
            model_name='patient',
            name='medical_history',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='patient',
            name='dry_weight',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='patient',
            name='location',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326),
        ),
        migrations.AddIndex(
            model_name='patient',
            index=models.Index(fields=['full_name'], name='idx_patient_name'),
        ),
        migrations.AddIndex(
            model_name='patient',
            index=models.Index(fields=['status'], name='idx_patient_status'),
        ),
        migrations.AddIndex(
            model_name='patient',
            index=models.Index(fields=['date_of_birth'], name='idx_patient_dob'),
        ),
        # --- BloodSample ---
        migrations.AlterField(
            model_name='bloodsample',
            name='notes',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='bloodsample',
            name='patient',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='blood_samples', to='core.patient'),
        ),
        migrations.AddField(
            model_name='bloodsample',
            name='created_by',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, related_name='created_blood_samples', to='core.customuser'),
        ),
        migrations.AddField(
            model_name='bloodsample',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='bloodsample',
            name='created_by',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='created_blood_samples', to='core.customuser'),
        ),
        migrations.AddIndex(
            model_name='bloodsample',
            index=models.Index(fields=['collection_date'], name='idx_bloodsample_col_date'),
        ),
        # --- DialysisSession ---
        migrations.AddField(
            model_name='dialysissession',
            name='notes',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='dialysissession',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='dialysissession',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='status',
            field=models.CharField(choices=[('SCHEDULED', 'Scheduled'), ('IN_PROGRESS', 'In progress'), ('COMPLETED', 'Completed'), ('CANCELLED', 'Cancelled')], default='SCHEDULED', max_length=20),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='pre_dialysis_bp',
            field=models.CharField(blank=True, default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='during_dialysis_bp',
            field=models.CharField(blank=True, default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='post_dialysis_bp',
            field=models.CharField(blank=True, default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='patient',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='dialysis_sessions', to='core.patient'),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='machine',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='dialysis_sessions', to='core.dialysismachine'),
        ),
        migrations.AlterField(
            model_name='dialysissession',
            name='assigned_nurse',
            field=models.ForeignKey(limit_choices_to={'role': 'NURSE'}, on_delete=django.db.models.deletion.PROTECT, related_name='assigned_dialysis_sessions', to='core.customuser'),
        ),
        migrations.AddIndex(
            model_name='dialysissession',
            index=models.Index(fields=['status'], name='idx_session_status'),
        ),
        migrations.AddIndex(
            model_name='dialysissession',
            index=models.Index(fields=['scheduled_start', 'scheduled_end'], name='idx_session_time_range'),
        ),
        # --- VitalSign ---
        migrations.CreateModel(
            name='VitalSign',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('recorded_at', models.DateTimeField()),
                ('systolic_bp', models.IntegerField(blank=True, null=True)),
                ('diastolic_bp', models.IntegerField(blank=True, null=True)),
                ('heart_rate', models.IntegerField(blank=True, null=True)),
                ('spo2', models.IntegerField(blank=True, null=True)),
                ('temperature', models.FloatField(blank=True, null=True)),
                ('notes', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('recorded_by', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='recorded_vital_signs', to='core.customuser')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='vital_signs', to='core.dialysissession')),
            ],
            options={
                'indexes': [models.Index(fields=['recorded_at'], name='idx_vital_recorded_at')],
            },
        ),
    ]
