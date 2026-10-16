import json
import logging
import sys
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from core.models import Department, Patient

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger('core.allocation')

# Bảng phân bổ được phê duyệt theo định danh ổn định (patient_id)
# Chỉ bao gồm 10 bệnh nhân có cơ sở lịch sử đồng nhất tại KHOA_A đã được rà soát trong Milestone 1D
DEFAULT_APPROVED_ALLOCATION = {
    "P_T2_1": "KHOA_A",
    "P_T2_2": "KHOA_A",
    "P_T2_3": "KHOA_A",
    "P_T2_4": "KHOA_A",
    "P_T2_5": "KHOA_A",
    "P_T3_1": "KHOA_A",
    "P_T3_2": "KHOA_A",
    "P_T3_3": "KHOA_A",
    "P_T3_4": "KHOA_A",
    "P_T3_5": "KHOA_A",
}


class Command(BaseCommand):
    help = (
        "Phan bo khoa ban dau an toan cho benh nhan (Milestone 1E). "
        "Mac dinh chay o che do --dry-run (chi mo phong, khong ghi DB). "
        "Yeu cau gan co --apply khi duoc chi dinh thuc thi thuc te."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply',
            action='store_true',
            default=False,
            help='Ap dung thuc te cac thay doi vao co so du lieu. Mac dinh la False (dry-run).',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            default=False,
            help='Chi hien thi ke hoach phan bo du kien ma khong thay doi du lieu.',
        )
        parser.add_argument(
            '--mapping-file',
            type=str,
            default=None,
            help='Duong dan file JSON chua bang anh xa tuy chinh {patient_id: dept_code}.',
        )

    def handle(self, *args, **options):
        is_apply = options.get('apply', False)
        # Mac dinh an toan: neu khong co --apply thi luon la dry-run
        is_dry_run = not is_apply or options.get('dry_run', False)

        mode_str = "APPLY (GHI DU LIEU THUC TE)" if not is_dry_run else "DRY-RUN (CHI MO PHONG - KHONG THAY DOI DB)"
        self.stdout.write(self.style.MIGRATE_HEADING(f"=== BAT DAU TIEN TRINH PHAN BO KHOA BENH NHAN [{mode_str}] ==="))

        # 1. Xac dinh bang anh xa
        mapping_file = options.get('mapping_file')
        if mapping_file:
            self.stdout.write(f"Doc bang anh xa tu file: {mapping_file}")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                allocation_map = json.load(f)
        else:
            self.stdout.write("Su dung danh sach phan bo duoc phe duyet mac dinh (10 benh nhan KHOA_A).")
            allocation_map = DEFAULT_APPROVED_ALLOCATION

        self.stdout.write(f"Tong so benh nhan trong danh sach phe duyet: {len(allocation_map)}")

        # 2. Tien kiem tra (Pre-validation) truoc khi thuc hien
        validation_errors = []
        plan_actions = []

        # Cache cac department de tranh query lap lai
        departments_cache = {}
        for dept in Department.objects.all():
            departments_cache[dept.code] = dept

        for patient_id, target_dept_code in allocation_map.items():
            # Kiem tra khoa dich
            if target_dept_code not in departments_cache:
                validation_errors.append(
                    f"[LOI] Khoa dich '{target_dept_code}' cho benh nhan '{patient_id}' khong ton tai trong he thong."
                )
                continue

            target_dept = departments_cache[target_dept_code]
            if not target_dept.is_active:
                validation_errors.append(
                    f"[LOI] Khoa dich '{target_dept.name}' ({target_dept_code}) dang o trang thai NGUNG HOAT DONG. "
                    f"Tu choi phan bo benh nhan '{patient_id}'."
                )
                continue

            # Kiem tra benh nhan
            try:
                patient = Patient.objects.get(patient_id=patient_id)
            except Patient.DoesNotExist:
                validation_errors.append(
                    f"[LOI] Benh nhan '{patient_id}' khong ton tai trong co so du lieu."
                )
                continue

            # Kiem tra bao ve ghi de
            if patient.department is not None:
                validation_errors.append(
                    f"[CANH BAO/LOI] Benh nhan '{patient_id}' hien da thuoc khoa "
                    f"'{patient.department.code}' (ID={patient.department.id}). "
                    f"Co che an toan khong cho phep ghi de tu dong. Phai dung quy trinh chuyen khoa."
                )
                continue

            plan_actions.append({
                'patient': patient,
                'patient_id': patient.patient_id,
                'current_dept': None,
                'target_dept': target_dept,
            })

        # Bao cao cac truong hop ngoai danh sach phan bo (unmapped patients)
        all_existing_patients = list(Patient.objects.all().values('patient_id', 'department__code', 'status'))
        unmapped_patients = [
            p for p in all_existing_patients if p['patient_id'] not in allocation_map
        ]

        self.stdout.write("\n--- CAC TRUONG HOP NGOAI DANH SACH PHE DUYET (GIU NGUYEN HIEN TRANG) ---")
        for p in unmapped_patients:
            curr = p['department__code'] or 'NULL'
            self.stdout.write(
                f"  - Ma tham chieu: {p['patient_id']} | Khoa hien tai: {curr} | "
                f"Trang thai: {p['status']} | Hanh dong: BO QUA (Cho xac nhan thu cong)"
            )

        # 3. Xu ly loi xac thuc neu co
        if validation_errors:
            self.stdout.write(self.style.ERROR("\n[DUNG KHAN CAP] Phat hien loi xac thuc trong du lieu phan bo:"))
            for err in validation_errors:
                self.stdout.write(self.style.ERROR(f"  {err}"))
            raise CommandError("Tien trinh phan bo bi huy bo an toan do du lieu khong hop le.")

        # 4. Hien thi ke hoach phan bo
        self.stdout.write(self.style.SUCCESS(f"\n--- KE HOACH PHAN BO HOP LE ({len(plan_actions)} BENH NHAN) ---"))
        for item in plan_actions:
            self.stdout.write(
                f"  [DU KIEN] Patient ID: {item['patient_id']} | "
                f"Khoa hien tai: NULL -> Khoa moi: {item['target_dept'].name} ({item['target_dept'].code})"
            )

        # 5. Thuc thi (neu la apply) hoac ket thuc mo phong (neu la dry-run)
        if is_dry_run:
            self.stdout.write(self.style.WARNING(
                "\n[DRY-RUN HOAN TAT] Khong co thay doi nao duoc ghi vao co so du lieu. "
                "Can gan co --apply sau khi nguoi dung phe duyet de thuc hien cap nhat."
            ))
            return

        # APPLY THUC TE voi transaction atomic
        self.stdout.write(self.style.NOTICE("\nDang ap dung thay doi trong Database Transaction Atomic..."))
        with transaction.atomic():
            updated_count = 0
            for item in plan_actions:
                # Reload de dam bao khong bi race condition
                patient = Patient.objects.select_for_update().get(patient_id=item['patient_id'])
                if patient.department is not None:
                    raise CommandError(
                        f"Benh nhan {patient.patient_id} da bi thay doi dong thoi trong qua trinh xu ly! Transaction rollback."
                    )
                patient.department = item['target_dept']
                # Chi cap nhat truong department va updated_at
                patient.save(update_fields=['department', 'updated_at'])
                updated_count += 1
                self.stdout.write(
                    f"  [DA CAP NHAT] Patient ID: {patient.patient_id} -> Khoa: {item['target_dept'].code}"
                )

        self.stdout.write(self.style.SUCCESS(
            f"\n[HOAN TAT APPLY] Da cap nhat thanh cong khoa cho {updated_count} benh nhan trong transaction atomic."
        ))
