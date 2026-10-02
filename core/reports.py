from datetime import datetime, time, timedelta

from django.db.models import Avg, Count, Q
from django.utils import timezone as dj_timezone
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import BloodSample, CustomUser, DialysisMachine, DialysisSession, Patient
from .permissions import IsStaffMember


def _parse_date(value, field_name):
    if not value:
        return None, None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date(), None
    except ValueError:
        return None, {field_name: ['Use YYYY-MM-DD format.']}


def _date_range(request):
    date_from, error = _parse_date(request.query_params.get('date_from'), 'date_from')
    if error:
        return None, None, error
    date_to, error = _parse_date(request.query_params.get('date_to'), 'date_to')
    if error:
        return None, None, error
    if date_from and date_to and date_from > date_to:
        return None, None, {'date_range': ['date_from must be before or equal to date_to.']}
    return date_from, date_to, None


def _apply_session_range(queryset, date_from, date_to):
    if date_from:
        queryset = queryset.filter(scheduled_start__date__gte=date_from)
    if date_to:
        queryset = queryset.filter(scheduled_start__date__lte=date_to)
    return queryset


def _apply_sample_range(queryset, date_from, date_to):
    if date_from:
        queryset = queryset.filter(collection_date__date__gte=date_from)
    if date_to:
        queryset = queryset.filter(collection_date__date__lte=date_to)
    return queryset


class SessionReportView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        date_from, date_to, error = _date_range(request)
        if error:
            return Response(error, status=400)

        queryset = _apply_session_range(DialysisSession.objects.all(), date_from, date_to)
        status_rows = queryset.values('status').annotate(count=Count('session_id')).order_by('status')
        machine_rows = (
            queryset.values('machine_id', 'machine__name')
            .annotate(count=Count('session_id'))
            .order_by('machine_id')
        )
        daily_rows = (
            queryset.values('scheduled_start__date')
            .annotate(count=Count('session_id'))
            .order_by('scheduled_start__date')
        )

        return Response({
            'total_sessions': queryset.count(),
            'unique_patients': queryset.values('patient_id').distinct().count(),
            'status_breakdown': [
                {'status': row['status'], 'count': row['count']} for row in status_rows
            ],
            'machine_breakdown': [
                {
                    'machine_id': row['machine_id'],
                    'machine_name': row['machine__name'],
                    'count': row['count'],
                }
                for row in machine_rows
            ],
            'daily_trend': [
                {'date': row['scheduled_start__date'].isoformat(), 'count': row['count']}
                for row in daily_rows
            ],
        })


class PatientReportView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        date_from, date_to, error = _date_range(request)
        if error:
            return Response(error, status=400)

        session_queryset = _apply_session_range(DialysisSession.objects.all(), date_from, date_to)
        treated_patient_ids = session_queryset.values('patient_id').distinct()
        status_rows = Patient.objects.values('status').annotate(count=Count('patient_id')).order_by('status')

        return Response({
            'total_patients': Patient.objects.count(),
            'treated_patients': len(treated_patient_ids),
            'status_breakdown': [
                {'status': row['status'], 'count': row['count']} for row in status_rows
            ],
        })


class MachineReportView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        date_from, date_to, error = _date_range(request)
        if error:
            return Response(error, status=400)

        session_queryset = _apply_session_range(DialysisSession.objects.all(), date_from, date_to)
        session_counts = {
            row['machine_id']: row['count']
            for row in session_queryset.values('machine_id').annotate(count=Count('session_id'))
        }
        machines = DialysisMachine.objects.order_by('machine_id')

        return Response({
            'total_machines': machines.count(),
            'status_breakdown': [
                {'status': row['status'], 'count': row['count']}
                for row in machines.values('status').annotate(count=Count('machine_id')).order_by('status')
            ],
            'machines': [
                {
                    'machine_id': machine.machine_id,
                    'machine_name': machine.name,
                    'status': machine.status,
                    'session_count': session_counts.get(machine.machine_id, 0),
                }
                for machine in machines
            ],
        })


class NurseWorkloadReportView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        date_from, date_to, error = _date_range(request)
        if error:
            return Response(error, status=400)

        queryset = _apply_session_range(DialysisSession.objects.all(), date_from, date_to)
        rows = (
            queryset.values('assigned_nurse_id', 'assigned_nurse__username', 'assigned_nurse__first_name', 'assigned_nurse__last_name')
            .annotate(
                assigned_sessions=Count('session_id'),
                completed_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.COMPLETED)),
                cancelled_sessions=Count('session_id', filter=Q(status=DialysisSession.Status.CANCELLED)),
            )
            .order_by('assigned_nurse_id')
        )

        return Response([
            {
                'nurse_id': row['assigned_nurse_id'],
                'username': row['assigned_nurse__username'],
                'first_name': row['assigned_nurse__first_name'],
                'last_name': row['assigned_nurse__last_name'],
                'assigned_sessions': row['assigned_sessions'],
                'completed_sessions': row['completed_sessions'],
                'cancelled_sessions': row['cancelled_sessions'],
            }
            for row in rows
        ])


class BloodSampleReportView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        date_from, date_to, error = _date_range(request)
        if error:
            return Response(error, status=400)

        queryset = _apply_sample_range(BloodSample.objects.all(), date_from, date_to)
        stats = queryset.aggregate(
            average_hemoglobin=Avg('hemoglobin_level'),
            average_potassium=Avg('potassium_level'),
        )
        daily_rows = (
            queryset.values('collection_date__date')
            .annotate(count=Count('sample_id'))
            .order_by('collection_date__date')
        )

        return Response({
            'total_samples': queryset.count(),
            'average_hemoglobin': stats['average_hemoglobin'],
            'average_potassium': stats['average_potassium'],
            'daily_trend': [
                {'date': row['collection_date__date'].isoformat(), 'count': row['count']}
                for row in daily_rows
            ],
        })
