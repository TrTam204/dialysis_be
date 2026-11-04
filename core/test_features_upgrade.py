from django.core.cache import cache
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.gis.geos import Point

from core.models import (
    AuditLog,
    CustomUser,
    Department,
    DialysisMachine,
    DialysisSession,
    MedicalMessage,
    Patient,
)

class FeatureUpgradeTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

        self.dept = Department.objects.create(name="Khoa Lọc Máu Test", code="KLM_TEST", is_active=True)
        self.dept2 = Department.objects.create(name="Khoa Lọc Máu B Test", code="KLM_B_TEST", is_active=True)

        self.user = CustomUser.objects.create_user(
            username="doctor_test",
            password="Password@123",
            email="doctor_test@dialysis.local",
            role=CustomUser.Role.DOCTOR,
            department=self.dept,
            first_name="An",
            last_name="Nguyễn",
        )
        self.nurse_user = CustomUser.objects.create_user(
            username="nurse_test",
            password="Password@123",
            email="nurse_test@dialysis.local",
            role=CustomUser.Role.NURSE,
            department=self.dept,
            first_name="Bình",
            last_name="Trần",
        )
        self.other_user = CustomUser.objects.create_user(
            username="doctor_other",
            password="Password@123",
            email="doctor_other@dialysis.local",
            role=CustomUser.Role.DOCTOR,
            department=self.dept,
            first_name="Cường",
            last_name="Lê",
        )

        self.patient = Patient.objects.create(
            patient_id="PAT_TEST_01",
            full_name="Bệnh Nhân Test",
            date_of_birth="1980-01-01",
            gender="MALE",
            dry_weight=60.0,
            department=self.dept,
            status=Patient.Status.ACTIVE,
        )
        self.machine = DialysisMachine.objects.create(
            machine_id="MAC_TEST_01",
            name="Máy Lọc Test 01",
            department=self.dept,
            status=DialysisMachine.Status.AVAILABLE,
        )
        self.session = DialysisSession.objects.create(
            session_id="SS_TEST_01",
            patient=self.patient,
            machine=self.machine,
            scheduled_start=timezone.now(),
            scheduled_end=timezone.now() + timezone.timedelta(hours=4),
            status=DialysisSession.Status.IN_PROGRESS,
            actual_start=timezone.now(),
        )

    def test_01_login_lockout_and_remaining_attempts(self):
        # 4 failed attempts -> 401 with remaining_attempts counting down
        for i in range(4):
            resp = self.client.post("/api/auth/login/", {"username": "doctor_test", "password": "WrongPassword"})
            self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
            self.assertEqual(resp.data["remaining_attempts"], 5 - (i + 1))
            self.assertEqual(resp.data["failed_attempts"], i + 1)
            self.assertIn("lần thử", resp.data["detail"])

        # 5th failed attempt -> 403 Forbidden with lockout notice
        resp = self.client.post("/api/auth/login/", {"username": "doctor_test", "password": "WrongPassword"})
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(resp.data["remaining_attempts"], 0)
        self.assertTrue(resp.data.get("locked"))
        self.assertIn("tạm khóa 15 phút", resp.data["detail"])

        # Subsequent attempt is immediately rejected as 403
        resp = self.client.post("/api/auth/login/", {"username": "doctor_test", "password": "WrongPassword"})
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(resp.data["remaining_attempts"], 0)

        # Check AuditLog was recorded
        audit = AuditLog.objects.filter(entity_id=str(self.user.id), changes__has_key="security_alert").first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.changes["security_alert"], "LOGIN_LOCKOUT_5_FAILED_ATTEMPTS")

        # Clear cache -> correct login works
        cache.delete("login_attempts_doctor_test")
        resp = self.client.post("/api/auth/login/", {"username": "doctor_test", "password": "Password@123"})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)

    def test_02_online_status_api(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.get("/api/users/online-status/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("count_online", resp.data)
        self.assertIn("staff", resp.data)
        self.assertTrue(any(s["username"] == "doctor_test" and s["is_online"] for s in resp.data["staff"]))

    def test_03_emergency_sos_flow_and_auto_broadcast(self):
        self.client.force_authenticate(user=self.user)
        sos_url = f"/api/sessions/{self.session.session_id}/sos/"
        resp = self.client.post(sos_url, {"reason": "Tụt huyết áp nặng SBP < 80 mmHg", "notes": "Bệnh nhân vã mồ hôi"})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(resp.data["is_emergency"])
        self.assertIn("protocol_guidelines", resp.data)

        # Verify DB updated
        self.session.refresh_from_db()
        self.assertTrue(self.session.is_emergency)
        self.assertIn("BÁO ĐỘNG SOS", self.session.clinical_notes)

        # Verify auto-broadcast message was created in department chat
        sos_msg = MedicalMessage.objects.filter(session_reference=self.session, receiver__isnull=True).first()
        self.assertIsNotNone(sos_msg)
        self.assertIn("BÁO ĐỘNG SOS KHẨN CẤP", sos_msg.content)

        # Clear SOS
        clear_url = f"/api/sessions/{self.session.session_id}/clear-sos/"
        resp = self.client.post(clear_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.session.refresh_from_db()
        self.assertFalse(self.session.is_emergency)

    def test_04_patient_latitude_longitude_serializer(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.patch(
            f"/api/patients/{self.patient.patient_id}/",
            {"latitude": 10.7915, "longitude": 106.6650},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertAlmostEqual(resp.data["latitude"], 10.7915, places=4)
        self.assertAlmostEqual(resp.data["longitude"], 106.6650, places=4)

        self.patient.refresh_from_db()
        self.assertIsNotNone(self.patient.location)
        self.assertAlmostEqual(self.patient.location.y, 10.7915, places=4)
        self.assertAlmostEqual(self.patient.location.x, 106.6650, places=4)

    def test_05_medical_chat_privacy_protection(self):
        # 1. Doctor A sends a private message to Nurse B
        self.client.force_authenticate(user=self.user)
        resp = self.client.post("/api/chat/messages/", {
            "content": "Tin nhắn riêng bảo mật từ Doctor A gửi Nurse B",
            "receiver": self.nurse_user.id,
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        msg_id = resp.data["id"]

        # 2. Doctor A sends a broadcast message to Department
        resp = self.client.post("/api/chat/messages/", {
            "content": "Tin nhắn chung Toàn khoa từ Doctor A",
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        broadcast_id = resp.data["id"]

        # 3. Nurse B logs in: Nurse B MUST see both broadcast and the private message from Doctor A
        self.client.force_authenticate(user=self.nurse_user)
        resp = self.client.get("/api/chat/messages/")
        nurse_msg_ids = [m["id"] for m in (resp.data.get("results") or resp.data)]
        self.assertIn(msg_id, nurse_msg_ids)
        self.assertIn(broadcast_id, nurse_msg_ids)

        # 4. Doctor C (third party) logs in: Doctor C MUST see broadcast, but MUST NEVER see the private message!
        self.client.force_authenticate(user=self.other_user)
        resp = self.client.get("/api/chat/messages/")
        other_msg_ids = [m["id"] for m in (resp.data.get("results") or resp.data)]
        self.assertIn(broadcast_id, other_msg_ids)
        self.assertNotIn(msg_id, other_msg_ids, "Strict Privacy Violated: Third party doctor_other must NOT see private message between doctor_test and nurse_test!")

        # 5. Doctor C queries department channel explicitly (?channel=department): only broadcast messages
        resp = self.client.get("/api/chat/messages/?channel=department")
        dept_msg_ids = [m["id"] for m in (resp.data.get("results") or resp.data)]
        self.assertIn(broadcast_id, dept_msg_ids)
        self.assertNotIn(msg_id, dept_msg_ids)

    def test_06_patient_transfer_department_action(self):
        self.client.force_authenticate(user=self.user)
        transfer_url = f"/api/patients/{self.patient.patient_id}/transfer/"

        # Transfer to same department -> 400
        resp = self.client.post(transfer_url, {"target_department_id": self.dept.id})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

        # Transfer to non-existent department -> 404
        resp = self.client.post(transfer_url, {"target_department_id": 999999})
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

        # Transfer to dept2 -> 200 OK
        resp = self.client.post(transfer_url, {
            "target_department_id": self.dept2.id,
            "reason": "Điều chuyển sang Khoa B để lọc máu chu kỳ",
        })
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(resp.data["success"])

        # Verify DB updated
        self.patient.refresh_from_db()
        self.assertEqual(self.patient.department_id, self.dept2.id)

        # Verify AuditLog recorded
        audit = AuditLog.objects.filter(entity_id=str(self.patient.patient_id), changes__action="TRANSFER_DEPARTMENT").first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.changes["to_department_id"], self.dept2.id)


    def test_07_admin_to_nurse_direct_message_and_conversations_order(self):
        admin_user = CustomUser.objects.create_user(
            username="admin_test_chat",
            password="Password@123",
            email="admin_chat@dialysis.local",
            role=CustomUser.Role.ADMIN,
            department=None,
            first_name="Admin",
            last_name="System",
        )

        # 1. Admin (department=None) sends direct message to nurse_test (department=KLM_TEST)
        self.client.force_authenticate(user=admin_user)
        msg_content = "Admin nhắn kiểm tra ca trực cho Nurse1"
        resp = self.client.post("/api/chat/messages/", {
            "content": msg_content,
            "receiver": self.nurse_user.id,
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)

        # 2. Nurse logs in: gets 1-1 messages with admin
        self.client.force_authenticate(user=self.nurse_user)
        resp = self.client.get(f"/api/chat/messages/?receiver={admin_user.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = resp.data.get("results") if isinstance(resp.data, dict) and "results" in resp.data else resp.data
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[-1]["content"], msg_content)

        # 3. Nurse gets conversations list:
        # Check pinned_channel exists, and admin is ranked #1 (index 0) due to latest message
        resp = self.client.get("/api/chat/messages/conversations/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("pinned_channel", resp.data)
        conversations = resp.data.get("conversations", [])
        self.assertGreater(len(conversations), 0)
        self.assertEqual(conversations[0]["id"], admin_user.id, "Admin must be ranked at index 0 because admin just sent the latest message!")
        self.assertEqual(conversations[0]["last_message"], msg_content)

        # 4. Nurse replies to Admin
        reply_content = "Nurse1 đã nhận lệnh từ Admin"
        resp = self.client.post("/api/chat/messages/", {
            "content": reply_content,
            "receiver": admin_user.id,
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)

        # 5. Admin queries 1-1 messages and sees Nurse's reply
        self.client.force_authenticate(user=admin_user)
        resp = self.client.get(f"/api/chat/messages/?receiver={self.nurse_user.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        admin_results = resp.data.get("results") if isinstance(resp.data, dict) and "results" in resp.data else resp.data
        self.assertEqual(len(admin_results), 2)
        self.assertEqual(admin_results[-1]["content"], reply_content)

    def test_08_unread_badge_mentions_and_multimedia_chat(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        import json

        admin_user = CustomUser.objects.create_user(
            username="admin_test_media",
            password="Password@123",
            email="admin_media@dialysis.local",
            role=CustomUser.Role.ADMIN,
            department=None,
            first_name="Admin",
            last_name="System",
        )

        # 1. Admin sends 2 messages to nurse_user
        self.client.force_authenticate(user=admin_user)

        # Message 1: plain text
        resp1 = self.client.post("/api/chat/messages/", {
            "content": "Tin nhắn 1 từ Admin chưa đọc",
            "receiver": self.nurse_user.id,
        })
        self.assertEqual(resp1.status_code, status.HTTP_201_CREATED)

        # Message 2: with @ mention of PT-001 and audio attachment
        fake_audio = SimpleUploadedFile("voice_note_01.wav", b"RIFFfake_wave_data", content_type="audio/wav")
        mention_metadata = {
            "mentions": [
                {
                    "ref_type": "PATIENT",
                    "id": str(self.patient.pk),
                    "code": str(self.patient.patient_id),
                    "title": f"Bệnh nhân: {self.patient.full_name} ({self.patient.patient_id})",
                    "subtitle": "Khoa A • Nhóm máu O+",
                    "doc_summary": "Hồ sơ bệnh án và kết quả xét nghiệm",
                    "link": f"/patients/{self.patient.patient_id}",
                }
            ]
        }

        resp2 = self.client.post("/api/chat/messages/", {
            "content": "Bác sĩ kiểm tra ghi âm ca lọc và hồ sơ @PT-001",
            "receiver": self.nurse_user.id,
            "attachment_file": fake_audio,
            "attachment_type": "AUDIO",
            "attachment_name": "voice_note_01.wav",
            "metadata": json.dumps(mention_metadata),
        }, format='multipart')
        self.assertEqual(resp2.status_code, status.HTTP_201_CREATED)

        # 2. Nurse calls GET /api/chat/messages/conversations/
        self.client.force_authenticate(user=self.nurse_user)
        resp_conv = self.client.get("/api/chat/messages/conversations/")
        self.assertEqual(resp_conv.status_code, status.HTTP_200_OK)
        self.assertIn("total_unread", resp_conv.data)
        self.assertGreaterEqual(resp_conv.data["total_unread"], 2)

        convs = resp_conv.data.get("conversations", [])
        admin_conv = next((c for c in convs if c["id"] == admin_user.id), None)
        self.assertIsNotNone(admin_conv, "Admin conversation must exist in nurse's list")
        self.assertEqual(admin_conv["unread_count"], 2, "Unread count for admin must be exactly 2")

        # 3. Nurse calls GET /api/chat/messages/mention-suggestions/?q=PAT
        resp_sug = self.client.get("/api/chat/messages/mention-suggestions/?q=PAT")
        self.assertEqual(resp_sug.status_code, status.HTTP_200_OK)
        self.assertIn("patients", resp_sug.data)
        self.assertIn("machines", resp_sug.data)
        self.assertIn("staff", resp_sug.data)
        self.assertGreaterEqual(len(resp_sug.data["patients"]), 1)
        found_pt = any(self.patient.patient_id in p["code"] for p in resp_sug.data["patients"])
        self.assertTrue(found_pt, "Mention suggestions must contain test patient")

        # 4. Nurse opens the conversation: GET /api/chat/messages/?receiver=<admin.id>
        resp_msgs = self.client.get(f"/api/chat/messages/?receiver={admin_user.id}")
        self.assertEqual(resp_msgs.status_code, status.HTTP_200_OK)
        results = resp_msgs.data.get("results") if isinstance(resp_msgs.data, dict) and "results" in resp_msgs.data else resp_msgs.data
        self.assertEqual(len(results), 2)

        # Check message 2 metadata and attachment
        msg2 = results[-1]
        self.assertIsNotNone(msg2.get("metadata"))
        self.assertIn("mentions", msg2["metadata"])
        self.assertEqual(msg2["attachment_type"], "AUDIO")
        self.assertEqual(msg2["attachment_name"], "voice_note_01.wav")
        self.assertIsNotNone(msg2.get("attachment_url"))

        # Check conversations again -> unread_count for admin is now 0
        resp_conv_after = self.client.get("/api/chat/messages/conversations/")
        admin_conv_after = next((c for c in resp_conv_after.data["conversations"] if c["id"] == admin_user.id), None)
        self.assertEqual(admin_conv_after["unread_count"], 0, "Unread count must be 0 after reading messages")
