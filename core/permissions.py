from rest_framework import permissions


class IsAdmin(permissions.BasePermission):
    message = 'Forbidden: ADMIN role required.'

    def has_permission(self, request, view):
        return bool(
            request.user and request.user.is_authenticated and request.user.role == 'ADMIN'
        )


class IsDoctor(permissions.BasePermission):
    message = 'Forbidden: DOCTOR role required.'

    def has_permission(self, request, view):
        return bool(
            request.user and request.user.is_authenticated and request.user.role == 'DOCTOR'
        )


class IsNurse(permissions.BasePermission):
    message = 'Forbidden: NURSE role required.'

    def has_permission(self, request, view):
        return bool(
            request.user and request.user.is_authenticated and request.user.role == 'NURSE'
        )


class IsAdminOrDoctor(permissions.BasePermission):
    message = 'Forbidden: ADMIN or DOCTOR role required.'

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role in {'ADMIN', 'DOCTOR'}
        )


class IsAdminOrNurse(permissions.BasePermission):
    message = 'Forbidden: ADMIN or NURSE role required.'

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role in {'ADMIN', 'NURSE'}
        )


class IsStaffMember(permissions.BasePermission):
    message = 'Forbidden: valid staff account required.'

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role in {'ADMIN', 'DOCTOR', 'NURSE'}
        )


class IsNursePatientStatusOnly(permissions.BasePermission):
    """NURSE workflow-limited update: may PATCH Patient.status only."""

    message = 'NURSE may only update the patient status field.'

    def has_permission(self, request, view):
        user = request.user
        if not (user and user.is_authenticated and user.role == 'NURSE'):
            return False
        if request.method != 'PATCH':
            return False
        data = request.data if isinstance(request.data, dict) else {}
        return set(data.keys()) <= {'status'}


class IsNurseAssignedSessionStatusOnly(permissions.BasePermission):
    """NURSE may PATCH status and clinical records only, on sessions assigned to them."""

    message = 'NURSE may only update status and clinical records of sessions assigned to them.'

    ALLOWED_FIELDS = {
        'status',
        'actual_start',
        'actual_end',
        'pre_weight',
        'post_weight',
        'uf_target',
        'uf_actual',
        'clinical_notes',
        'notes',
    }

    def has_permission(self, request, view):
        user = request.user
        if not (user and user.is_authenticated and user.role == 'NURSE'):
            return False
        if request.method != 'PATCH':
            return False
        data = request.data if isinstance(request.data, dict) else {}
        return set(data.keys()) <= self.ALLOWED_FIELDS

    def has_object_permission(self, request, view, obj):
        return obj.assigned_nurse_id == request.user.id


class IsNurseAssignedSessionVitals(permissions.BasePermission):
    """NURSE may create/update vital signs only for sessions assigned to them."""

    message = 'NURSE may only manage vital signs of sessions assigned to them.'

    def has_permission(self, request, view):
        user = request.user
        if not (user and user.is_authenticated and user.role == 'NURSE'):
            return False
        if request.method not in ('POST', 'PUT', 'PATCH'):
            return False
        if request.method == 'POST':
            data = request.data if isinstance(request.data, dict) else {}
            session_ref = data.get('session')
            if session_ref is None:
                return False
            from .models import DialysisSession

            try:
                session = DialysisSession.objects.get(pk=session_ref)
            except (DialysisSession.DoesNotExist, ValueError, TypeError):
                # Let the serializer surface a clean validation error for bad FKs.
                return True
            return session.assigned_nurse_id == user.id
        return True

    def has_object_permission(self, request, view, obj):
        return obj.session.assigned_nurse_id == request.user.id


def enforce_role_error_message(required_roles):
    return f'Forbidden: required role(s): {", ".join(required_roles)}.'
