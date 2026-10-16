"""
TOPSIS Decision Support & Triage API Views
Task: T9-01 — TOPSIS V1: ALGORITHM, BACKEND, FRONTEND & ACCEPTANCE
Author: Nguyen Ho Truong Tam
"""

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.permissions import IsStaffMember
from core.topsis_service import TopsisTriageService


class TopsisCriteriaView(APIView):
    """
    GET /api/topsis/criteria/
    Returns the TOPSIS criteria catalogue, default weights, and clinical disclaimers.
    """
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def get(self, request):
        catalogue = TopsisTriageService.get_criteria_catalogue()
        return Response(
            {
                "criteria": catalogue,
                "is_clinical_approved": False,
                "version": "TOPSIS_V1_DEMO",
                "disclaimer": (
                    "CHÚ Ý CHUYÊN MÔN: Bộ tiêu chí và trọng số này được xây dựng cho mục đích thử nghiệm kỹ thuật (DEMO) "
                    "và hỗ trợ nghiên cứu đề tài tốt nghiệp. Chưa được thẩm định lâm sàng và không thay thế quyết định y khoa."
                ),
            },
            status=status.HTTP_200_OK,
        )


class TopsisRankView(APIView):
    """
    POST /api/topsis/rank/
    Executes TOPSIS multi-criteria priority ranking for patients in a department.
    Ensures zero cross-department data leakage and preserves read-only safety.
    """
    permission_classes = [permissions.IsAuthenticated, IsStaffMember]

    def post(self, request):
        data = request.data or {}
        department_id = data.get("department")
        custom_weights = data.get("custom_weights")
        patient_ids = data.get("patient_ids")
        center_coords = data.get("center_coordinates")

        # Validate department parameter
        if department_id is not None and department_id != "":
            try:
                department_id = int(department_id)
            except (ValueError, TypeError):
                return Response(
                    {"detail": "Mã khoa phòng (department) phải là số nguyên hợp lệ."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            department_id = None

        # Validate center coordinates if provided
        if center_coords is not None:
            if not isinstance(center_coords, (list, tuple)) or len(center_coords) != 2:
                return Response(
                    {"detail": "Tọa độ trung tâm (center_coordinates) phải là mảng gồm [latitude, longitude]."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                center_coords = (float(center_coords[0]), float(center_coords[1]))
            except (ValueError, TypeError):
                return Response(
                    {"detail": "Tọa độ trung tâm phải chứa các giá trị số thực hợp lệ."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Validate custom weights if provided
        if custom_weights is not None:
            if not isinstance(custom_weights, dict):
                return Response(
                    {"detail": "Trọng số tùy chỉnh (custom_weights) phải là đối tượng key-value."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            for k, w in custom_weights.items():
                try:
                    w_float = float(w)
                    if w_float < 0:
                        return Response(
                            {"detail": f"Trọng số của tiêu chí '{k}' không được mang giá trị âm."},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                except (ValueError, TypeError):
                    return Response(
                        {"detail": f"Trọng số của tiêu chí '{k}' phải là giá trị số."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            if sum(float(w) for w in custom_weights.values()) <= 0:
                return Response(
                    {"detail": "Tổng các trọng số tùy chỉnh phải lớn hơn 0."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        try:
            result = TopsisTriageService.rank_department_patients(
                department_id=department_id,
                user=request.user,
                custom_weights=custom_weights,
                patient_ids=patient_ids,
                center_coords=center_coords,
            )
            return Response(result, status=status.HTTP_200_OK)
        except PermissionError as e:
            return Response({"detail": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"detail": f"Lỗi trong quá trình tính toán TOPSIS: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
