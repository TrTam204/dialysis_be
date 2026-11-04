from django.core.management.base import BaseCommand
from django.contrib.auth import authenticate
from django.core.cache import cache
from core.models import CustomUser

class Command(BaseCommand):
    help = "Sync all user passwords to match DEMO_TEST_GUIDE.md and verify authentication"

    PASSWORD_MAP = {
        'admin': 'Admin@123',
        'doctor1': 'Doctor@123',
        'doctor2': 'Doctor@123',
        'bs_khoa_b': 'Doctor@123',
        'nurse1': 'Nurse@123',
        'nurse2': 'Nurse@123',
    }

    def handle(self, *args, **options):
        # 1. Clear lockout caches
        cache.clear()
        self.stdout.write(self.style.SUCCESS("Cleared all Django cache (including login attempts)."))

        users = CustomUser.objects.all().order_by('id')
        self.stdout.write(f"Found {users.count()} users in database.")

        results = []
        for user in users:
            # Determine standard password
            if user.username in self.PASSWORD_MAP:
                pwd = self.PASSWORD_MAP[user.username]
            elif user.role == CustomUser.Role.ADMIN:
                pwd = 'Admin@123'
            elif user.role == CustomUser.Role.DOCTOR:
                pwd = 'Doctor@123'
            else:
                pwd = 'Nurse@123'

            user.set_password(pwd)
            user.is_active = True
            user.save()

            # Verify with authenticate()
            auth_user = authenticate(username=user.username, password=pwd)
            is_ok = (auth_user is not None and auth_user.id == user.id)
            results.append((user.username, user.role, pwd, is_ok))

        self.stdout.write("\n=== AUTHENTICATION VERIFICATION RESULTS ===")
        all_passed = True
        for username, role, pwd, is_ok in results:
            status_str = "PASS" if is_ok else "FAIL"
            if not is_ok:
                all_passed = False
            self.stdout.write(f"[{status_str}] User: {username:<12} | Role: {role:<8} | Pwd: {pwd:<12}")

        if all_passed:
            self.stdout.write(self.style.SUCCESS("\n100% of user accounts authenticated successfully!"))
        else:
            self.stdout.write(self.style.ERROR("\nSome user accounts failed authentication!"))

