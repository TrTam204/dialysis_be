from datetime import date, datetime, time
from decimal import Decimal
from django.db import transaction
from django.db.models import ProtectedError
from rest_framework import status
from rest_framework.response import Response

from .models import AuditLog


SENSITIVE_FIELDS = {
    'password',
    'token',
    'access_token',
    'refresh_token',
    'secret',
    'otp',
    'credential',
    'credentials',
}

EXCLUDED_UPDATE_FIELDS = {
    'created_at',
    'updated_at',
}


def is_sensitive_field(field_name: str) -> bool:
    name_lower = field_name.lower()
    return name_lower in SENSITIVE_FIELDS or any(s in name_lower for s in ('password', 'token', 'secret', 'otp'))


def serialize_value(val):
    if val is None:
        return None
    if isinstance(val, (datetime, date, time)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if hasattr(val, 'x') and hasattr(val, 'y'):
        # GeoDjango PointField
        return [val.x, val.y]
    if isinstance(val, (int, float, str, bool, dict, list)):
        return val
    return str(val)


def get_model_snapshot(instance) -> dict:
    snapshot = {}
    for field in instance._meta.concrete_fields:
        field_name = field.name
        if is_sensitive_field(field_name):
            continue

        # For foreign keys, store attname (e.g. department_id, patient_id)
        if field.is_relation and field.many_to_one:
            raw_val = getattr(instance, field.attname)
        else:
            raw_val = getattr(instance, field_name)

        snapshot[field_name] = serialize_value(raw_val)
    return snapshot


def calculate_changes(before_snapshot: dict, after_snapshot: dict) -> dict:
    changes = {}
    all_keys = set(before_snapshot.keys()) | set(after_snapshot.keys())
    for key in all_keys:
        if key in EXCLUDED_UPDATE_FIELDS or is_sensitive_field(key):
            continue
        before_val = before_snapshot.get(key)
        after_val = after_snapshot.get(key)
        if before_val != after_val:
            changes[key] = {
                'before': before_val,
                'after': after_val,
            }
    return changes


def record_audit_log(actor, action: str, instance, changes: dict = None) -> AuditLog:
    if not getattr(actor, 'is_authenticated', False):
        log_actor = None
    else:
        log_actor = actor

    entity_type = instance.__class__.__name__
    entity_id = str(instance.pk)

    if changes is None:
        if action in (AuditLog.Action.CREATE, AuditLog.Action.DELETE):
            changes = get_model_snapshot(instance)
        else:
            changes = {}

    return AuditLog.objects.create(
        actor=log_actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        changes=changes,
    )


class AuditLogMixin:
    """
    Mixin for DRF ViewSets to automatically capture CREATE, UPDATE, DELETE actions
    with before/after diffs, sanitization, and database transaction consistency.
    """

    def create(self, request, *args, **kwargs):
        with transaction.atomic():
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            instance = serializer.instance
            record_audit_log(actor=request.user, action=AuditLog.Action.CREATE, instance=instance)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        before_snapshot = get_model_snapshot(instance)
        with transaction.atomic():
            serializer = self.get_serializer(instance, data=request.data, partial=partial)
            serializer.is_valid(raise_exception=True)
            self.perform_update(serializer)
            instance.refresh_from_db()
            after_snapshot = get_model_snapshot(instance)
            changes = calculate_changes(before_snapshot, after_snapshot)
            if changes:
                record_audit_log(
                    actor=request.user,
                    action=AuditLog.Action.UPDATE,
                    instance=instance,
                    changes=changes,
                )
            return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        before_snapshot = get_model_snapshot(instance)
        try:
            with transaction.atomic():
                record_audit_log(
                    actor=request.user,
                    action=AuditLog.Action.DELETE,
                    instance=instance,
                    changes=before_snapshot,
                )
                self.perform_destroy(instance)
        except ProtectedError:
            return Response(
                {'detail': 'Cannot delete this record because it is still referenced by other data.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)
