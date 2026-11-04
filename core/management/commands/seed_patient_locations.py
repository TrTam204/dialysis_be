"""
Management command to seed realistic geographic coordinates (WGS84 / PostGIS SRID 4326)
for patients across Ho Chi Minh City districts.
Enables TOPSIS Criterion C5 (Distance in km) to compute true geodesic distances
from the Hemodialysis Center (District 5 - Cho Ray / DH Y Duoc: 10.7556° N, 106.6602° E).
"""
import math
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from core.models import Patient, Department

# Dialysis Center Reference Coordinates (Benh vien Cho Ray TP.HCM - 201B Nguyen Chi Thanh, Q.5)
CENTER_LAT = 10.7556
CENTER_LON = 106.6602

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 2)

# Predefined realistic residential locations in HCMC
HCMC_LOCATIONS = [
    {"district": "Quận Tân Bình (Lê Văn Sỹ)", "lat": 10.7925, "lon": 106.6620},
    {"district": "Quận 3 (Võ Thị Sáu)", "lat": 10.7845, "lon": 106.6870},
    {"district": "Quận Phú Nhuận (Phan Xích Long)", "lat": 10.7980, "lon": 106.6850},
    {"district": "Quận 10 (Đường 3 Tháng 2)", "lat": 10.7720, "lon": 106.6690},
    {"district": "Quận Tân Bình (Hoàng Hoa Thám)", "lat": 10.8010, "lon": 106.6450},
    {"district": "Quận 11 (Lạc Long Quân)", "lat": 10.7680, "lon": 106.6480},
    {"district": "Quận 1 (Nguyễn Thị Minh Khai)", "lat": 10.7769, "lon": 106.7009},
    {"district": "Quận 5 (Trần Hưng Đạo)", "lat": 10.7550, "lon": 106.6660},
    {"district": "Quận Bình Thạnh (Bạch Đằng)", "lat": 10.8030, "lon": 106.7010},
    {"district": "Quận Tân Phú (Lũy Bán Bích)", "lat": 10.7830, "lon": 106.6270},
    {"district": "Quận 6 (Hậu Giang)", "lat": 10.7500, "lon": 106.6340},
    {"district": "Quận Gò Vấp (Quang Trung)", "lat": 10.8380, "lon": 106.6600},
    {"district": "Quận 8 (Phạm Hùng)", "lat": 10.7380, "lon": 106.6780},
    {"district": "Quận Bình Tân (Tên Lửa)", "lat": 10.7530, "lon": 106.6080},
    {"district": "Quận 7 (Nguyễn Văn Linh)", "lat": 10.7300, "lon": 106.7150},
    {"district": "Quận 12 (Tô Ký)", "lat": 10.8670, "lon": 106.6410},
    {"district": "Huyện Hóc Môn (Lê Thị Hà)", "lat": 10.8850, "lon": 106.5920},
    {"district": "TP. Thủ Đức (Võ Văn Ngân)", "lat": 10.8510, "lon": 106.7720},
    {"district": "Huyện Bình Chánh (Quốc Lộ 1A)", "lat": 10.6950, "lon": 106.5820},
    {"district": "Huyện Nhà Bè (Huỳnh Tấn Phát)", "lat": 10.6920, "lon": 106.7450},
    {"district": "TP. Thủ Đức (Nguyễn Duy Trinh - Q9)", "lat": 10.7900, "lon": 106.7950},
]

class Command(BaseCommand):
    help = "Seed real HCMC PostGIS coordinates for existing patients to empower TOPSIS Criterion C5."

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Overwrite coordinates even if already set.',
        )

    def handle(self, *args, **options):
        force = options.get('force', False)
        patients = list(Patient.objects.all().order_by('patient_id'))
        count = len(patients)

        self.stdout.write(self.style.NOTICE(f"=== SEEDING POSTGIS COORDINATES FOR {count} PATIENTS ==="))
        self.stdout.write(f"Dialysis Center: {CENTER_LAT}° N, {CENTER_LON}° E (Tan Binh / Q3)")

        updated_count = 0
        for idx, patient in enumerate(patients):
            if patient.location and not force:
                self.stdout.write(f"  [SKIPPED] {patient.patient_id} ({patient.full_name}) already has location.")
                continue

            loc = HCMC_LOCATIONS[idx % len(HCMC_LOCATIONS)]
            lat = loc["lat"]
            lon = loc["lon"]
            district = loc["district"]

            # PostGIS Point takes (x=lon, y=lat, srid=4326)
            patient.location = Point(lon, lat, srid=4326)
            
            # If patient has no department, assign to KHOA_B (14) or KHOA_A (13) for demo
            if not patient.department_id:
                khoa_b = Department.objects.filter(code='KHOA_B').first()
                if khoa_b:
                    patient.department = khoa_b
            
            patient.save(update_fields=['location', 'department'])
            dist_km = haversine_km(lat, lon, CENTER_LAT, CENTER_LON)
            self.stdout.write(
                self.style.SUCCESS(
                    f"  [UPDATED] {patient.patient_id} - {patient.full_name}: "
                    f"({lat}, {lon}) - {district} -> Khoảng cách: {dist_km} km (Khoa: {patient.department.code if patient.department else 'N/A'})"
                )
            )
            updated_count += 1

        self.stdout.write(self.style.SUCCESS(f"\nSuccessfully seeded coordinates for {updated_count}/{count} patients!"))

