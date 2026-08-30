"""Phase 1 API smoke test against the running dev server.

Covers: auth (login/refresh/logout), RBAC matrix for 3 roles, CRUD for every
resource, business-rule validations, nested vital signs, and dashboard stats.
Run: python api_smoke_test.py
"""
import json
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8000/api'
PASSED, FAILED = [], []


def call(method, path, token=None, body=None):
    url = f'{BASE}{path}'
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or '{}')
        except Exception:
            return e.code, {}


def check(name, cond, extra=''):
    if cond:
        PASSED.append(name)
    else:
        FAILED.append(f'{name} {extra}')
        print(f'  FAIL: {name} {extra}')


def login(username, password):
    status, data = call('POST', '/auth/login/', body={'username': username, 'password': password})
    if status != 200:
        raise SystemExit(f'login failed for {username}: {status} {data}')
    return data['access'], data


def cleanup(admin_tok):
    """Remove data created by previous smoke-test runs (keeps the DB tidy)."""
    call('DELETE', '/blood-samples/BL-001/', token=admin_tok)
    for sid in ('SS-001', 'SS-OK2', 'SS-N2', 'SS-BAD', 'SS-OV', 'SS-DOC'):
        call('DELETE', f'/sessions/{sid}/', token=admin_tok)
    call('DELETE', '/machines/M-001/', token=admin_tok)
    for pid in ('PT-001', 'PT-FUT', 'PT-NEG', 'PT-ST', 'PT-N'):
        call('DELETE', f'/patients/{pid}/', token=admin_tok)
    call('DELETE', '/users/999999/', token=admin_tok)  # teststaff removed by username below
    s, d = call('GET', '/users/?search=teststaff', token=admin_tok)
    for row in d.get('results', []):
        if row['username'] == 'teststaff':
            call('DELETE', f"/users/{row['id']}/", token=admin_tok)
    for code in ('KTM', 'OTHER', 'XX'):
        s, d = call('GET', f'/departments/?search={code}', token=admin_tok)
        for row in d.get('results', []):
            if row['code'] in ('KTM', 'OTHER', 'XX'):
                call('DELETE', f"/departments/{row['id']}/", token=admin_tok)


print('== AUTH ==')
admin_tok, admin_login = login('admin', 'admin123')
doctor_tok, doctor_login = login('doctor1', 'Doctor@123')
nurse_tok, nurse_login = login('nurse1', 'Nurse@123')
cleanup(admin_tok)
check('login returns user with department id int', isinstance(admin_login['user']['department'], int))
check('login returns role', admin_login['user']['role'] == 'ADMIN')

s, d = call('POST', '/auth/login/', body={'username': 'admin', 'password': 'wrong'})
check('wrong password -> 401', s == 401)

s, d = call('POST', '/auth/refresh/', body={'refresh': admin_login['refresh']})
check('refresh -> 200 with access', s == 200 and 'access' in d)

s, d = call('POST', '/auth/logout/', token=admin_tok)
check('logout -> 200 message', s == 200 and 'detail' in d)

print('== RBAC: read access ==')
s, d = call('GET', '/departments/')
check('unauthenticated departments list -> 401', s == 401)
for role, tok in [('admin', admin_tok), ('doctor', doctor_tok), ('nurse', nurse_tok)]:
    s, d = call('GET', '/departments/', token=tok)
    check(f'{role} can list departments', s == 200 and 'results' in d)
    s, d = call('GET', '/users/', token=tok)
    check(f'{role} can list users', s == 200)
    s, d = call('GET', '/patients/', token=tok)
    check(f'{role} can list patients', s == 200)
    s, d = call('GET', '/machines/', token=tok)
    check(f'{role} can list machines', s == 200)
    s, d = call('GET', '/sessions/', token=tok)
    check(f'{role} can list sessions', s == 200)
    s, d = call('GET', '/blood-samples/', token=tok)
    check(f'{role} can list blood samples', s == 200)
    s, d = call('GET', '/dashboard/summary/', token=tok)
    check(f'{role} dashboard summary', s == 200)

s, d = call('POST', '/departments/', token=nurse_tok, body={'name': 'X', 'code': 'XX'})
check('nurse create department -> 403', s == 403)
s, d = call('POST', '/departments/', token=doctor_tok, body={'name': 'X', 'code': 'XX'})
check('doctor create department -> 403', s == 403)

print('== DEPARTMENT CRUD (admin) ==')
s, dept = call('POST', '/departments/', token=admin_tok, body={
    'name': 'Khoa Tim Mach', 'code': 'ktm', 'description': 'Cardiology', 'is_active': True})
check('create department', s == 201 and dept['code'] == 'KTM', str(dept))
dept_id = dept['id']
s, d = call('POST', '/departments/', token=admin_tok, body={'name': 'Khoa Tim Mach', 'code': 'OTHER'})
check('duplicate name -> 400', s == 400 and 'name' in d)
s, d = call('GET', f'/departments/?search=KTM', token=admin_tok)
check('department search works', s == 200 and d['count'] >= 1)
s, d = call('GET', '/departments/?is_active=true&ordering=-code', token=admin_tok)
check('department filter+ordering works', s == 200)
s, d = call('PATCH', f'/departments/{dept_id}/', token=admin_tok, body={'description': 'Updated'})
check('patch department', s == 200 and d['description'] == 'Updated')
s, d = call('PUT', f'/departments/{dept_id}/', token=admin_tok, body={
    'name': 'Khoa Tim Mach', 'code': 'KTM', 'description': 'Full update', 'is_active': False})
check('put department', s == 200 and d['is_active'] is False)

print('== STAFF CRUD (admin) ==')
s, user = call('POST', '/users/', token=admin_tok, body={
    'username': 'teststaff', 'email': 'teststaff@dialysis.local', 'first_name': 'D',
    'last_name': 'Vo', 'role': 'DOCTOR', 'department': dept_id, 'password': 'Staff@123'})
check('create user', s == 201, str(user))
check('user response has no password', 'password' not in user)
check('user response department is id', user['department'] == dept_id)
check('user response has department_name', user.get('department_name') == 'Khoa Tim Mach')
staff_id = user['id']
s, d = call('POST', '/users/', token=admin_tok, body={
    'username': 'nopass', 'email': 'nopass@dialysis.local', 'role': 'NURSE', 'department': dept_id})
check('create user without password -> 400', s == 400 and 'password' in d)
s, d = call('POST', '/users/', token=admin_tok, body={
    'username': 'badrole', 'email': 'badrole@dialysis.local', 'role': 'GOD', 'department': dept_id, 'password': 'x'})
check('invalid role -> 400', s == 400)
s, d = call('POST', '/users/', token=admin_tok, body={
    'username': 'baddept', 'email': 'baddept@dialysis.local', 'role': 'NURSE', 'department': 99999, 'password': 'x'})
check('invalid department FK -> 400', s == 400)
s, d = call('PATCH', f'/users/{staff_id}/', token=admin_tok, body={'password': 'NewPass@123'})
check('patch user password ok', s == 200)
s2, _ = login('teststaff', 'NewPass@123')
check('user can login with patched password', s2 is not None)
s, d = call('POST', '/users/', token=doctor_tok, body={
    'username': 'hax', 'email': 'hax@dialysis.local', 'role': 'ADMIN', 'department': dept_id, 'password': 'x'})
check('doctor create user -> 403', s == 403)
s, d = call('GET', '/users/?role=DOCTOR&department=%d&is_active=true&search=teststaff' % dept_id, token=admin_tok)
check('user filter+search works', s == 200 and d['count'] >= 1)

print('== PATIENT CRUD ==')
s, pat = call('POST', '/patients/', token=doctor_tok, body={
    'patient_id': 'PT-001', 'full_name': 'Nguyen Van A', 'date_of_birth': '1980-05-10',
    'gender': 'MALE', 'phone_number': '0901000001', 'medical_history': 'HBV',
    'dry_weight': 62.5, 'status': 'IN_TREATMENT'})
check('doctor create patient', s == 201, str(pat))
s, d = call('POST', '/patients/', token=doctor_tok, body={
    'patient_id': 'PT-FUT', 'full_name': 'X', 'date_of_birth': '2999-01-01'})
check('future date_of_birth -> 400', s == 400 and 'date_of_birth' in d)
s, d = call('POST', '/patients/', token=doctor_tok, body={
    'patient_id': 'PT-NEG', 'full_name': 'X', 'date_of_birth': '1980-01-01', 'dry_weight': -5})
check('negative dry_weight -> 400', s == 400 and 'dry_weight' in d)
s, d = call('POST', '/patients/', token=doctor_tok, body={
    'patient_id': 'PT-ST', 'full_name': 'X', 'date_of_birth': '1980-01-01', 'status': 'WEIRD'})
check('invalid status -> 400', s == 400)
s, d = call('GET', '/patients/?search=Nguyen&status=IN_TREATMENT&ordering=full_name', token=nurse_tok)
check('patient search+filter+ordering', s == 200 and d['count'] >= 1)
s, d = call('PATCH', '/patients/PT-001/', token=nurse_tok, body={'status': 'STABLE'})
check('nurse patch patient status -> 200', s == 200 and d['status'] == 'STABLE', str(d))
s, d = call('PATCH', '/patients/PT-001/', token=nurse_tok, body={'full_name': 'Hacked'})
check('nurse patch patient full_name -> 403', s == 403)
s, d = call('PUT', '/patients/PT-001/', token=nurse_tok, body={
    'patient_id': 'PT-001', 'full_name': 'Nguyen Van A', 'date_of_birth': '1980-05-10'})
check('nurse put patient -> 403', s == 403)
s, d = call('POST', '/patients/', token=nurse_tok, body={
    'patient_id': 'PT-N', 'full_name': 'N', 'date_of_birth': '1980-01-01'})
check('nurse create patient -> 403', s == 403)

print('== MACHINE CRUD ==')
s, mach = call('POST', '/machines/', token=admin_tok, body={
    'machine_id': 'M-001', 'name': 'Machine 01', 'status': 'AVAILABLE', 'department': dept_id})
check('admin create machine', s == 201 and mach['department'] == dept_id, str(mach))
s, d = call('POST', '/machines/', token=admin_tok, body={
    'machine_id': 'M-FUT', 'name': 'Bad', 'department': dept_id, 'last_maintenance_date': '2999-01-01'})
check('future maintenance date -> 400', s == 400 and 'last_maintenance_date' in d)
s, d = call('POST', '/machines/', token=doctor_tok, body={
    'machine_id': 'M-D', 'name': 'Doc', 'department': dept_id})
check('doctor create machine -> 403', s == 403)
s, d = call('GET', '/machines/?status=AVAILABLE&department=%d&search=M-001' % dept_id, token=nurse_tok)
check('machine filter+search (nurse read)', s == 200 and d['count'] == 1)

print('== SESSION CRUD + validation ==')
s, sess = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-001', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': nurse_login['user']['id'],
    'scheduled_start': '2026-09-01T08:00:00Z', 'scheduled_end': '2026-09-01T12:00:00Z'})
check('doctor create session', s == 201, str(sess))
check('session has display names', sess.get('patient_name') == 'Nguyen Van A' and sess.get('machine_name') == 'Machine 01')
s, d = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-BAD', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': nurse_login['user']['id'],
    'scheduled_start': '2026-09-01T08:00:00Z', 'scheduled_end': '2026-09-01T07:00:00Z'})
check('end before start -> 400', s == 400 and 'scheduled_end' in d)
s, d = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-OV', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': nurse_login['user']['id'],
    'scheduled_start': '2026-09-01T10:00:00Z', 'scheduled_end': '2026-09-01T14:00:00Z'})
check('machine overlap -> 400', s == 400 and 'machine' in d)
s, d = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-DOC', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': staff_id,
    'scheduled_start': '2026-09-02T08:00:00Z', 'scheduled_end': '2026-09-02T12:00:00Z'})
check('non-nurse assigned_nurse -> 400', s == 400 and 'assigned_nurse' in d, str(d))
s, d = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-OK2', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': nurse_login['user']['id'],
    'scheduled_start': '2026-09-01T13:00:00Z', 'scheduled_end': '2026-09-01T16:00:00Z'})
check('non-overlapping same machine ok', s == 201)
s, d = call('GET', '/sessions/?patient=PT-001&machine=M-001&status=SCHEDULED&date_from=2026-09-01T00:00:00Z', token=nurse_tok)
check('session filters work', s == 200 and d['count'] >= 1)
s, d = call('PATCH', '/sessions/SS-001/', token=nurse_tok, body={'status': 'IN_PROGRESS'})
check('assigned nurse patch session status -> 200', s == 200 and d['status'] == 'IN_PROGRESS', str(d))
s, d = call('POST', '/sessions/', token=doctor_tok, body={
    'session_id': 'SS-N2', 'patient': 'PT-001', 'machine': 'M-001',
    'assigned_nurse': 4,  # nurse2 (seeded order: admin=1, doctor1=2, nurse1=3, nurse2=4)
    'scheduled_start': '2026-09-03T08:00:00Z', 'scheduled_end': '2026-09-03T12:00:00Z'})
check('create session assigned to nurse2', s == 201, str(d))
s, d = call('PATCH', '/sessions/SS-N2/', token=nurse_tok, body={'status': 'CANCELLED'})
check('nurse (not assigned) patch session -> 403', s == 403, f'status={s}')
s, d = call('PATCH', '/sessions/SS-001/', token=nurse_tok, body={'scheduled_end': '2026-09-01T18:00:00Z'})
check('nurse patch session time field -> 403', s == 403)
s, d = call('PATCH', '/sessions/SS-001/', token=doctor_tok, body={'status': 'IN_PROGRESS'})
check('doctor patch session status -> 200', s == 200)

print('== VITAL SIGNS (nested + flat) ==')
s, d = call('GET', '/sessions/SS-001/vitals/', token=nurse_tok)
check('nested list vitals (empty)', s == 200 and d == [])
s, vital = call('POST', '/sessions/SS-001/vitals/', token=nurse_tok, body={
    'recorded_at': '2026-09-01T08:30:00Z', 'systolic_bp': 120, 'diastolic_bp': 80,
    'heart_rate': 75, 'spo2': 98, 'temperature': 36.5, 'notes': 'Binh thuong'})
check('assigned nurse nested POST vital', s == 201 and vital['session'] == 'SS-001', str(vital))
check('vital recorded_by auto-set', vital['recorded_by'] == nurse_login['user']['id'])
check('vital recorded_by_name present', bool(vital.get('recorded_by_name')))
s, d = call('POST', '/sessions/SS-001/vitals/', token=doctor_tok, body={
    'recorded_at': '2026-09-01T09:00:00Z', 'systolic_bp': 118, 'diastolic_bp': 78,
    'heart_rate': 78, 'spo2': 97, 'temperature': 36.6})
check('doctor nested POST vital', s == 201)
s, d = call('POST', '/sessions/SS-001/vitals/', token=admin_tok, body={
    'recorded_at': '2026-09-01T09:30:00Z', 'spo2': 150})
check('spo2 out of range -> 400', s == 400 and 'spo2' in d)
s, d = call('POST', '/sessions/SS-001/vitals/', token=admin_tok, body={
    'recorded_at': '2026-09-01T09:30:00Z', 'heart_rate': 500})
check('heart_rate out of range -> 400', s == 400 and 'heart_rate' in d)
# nurse2 posts to nurse1's session -> 403
nurse2_tok, _ = login('nurse2', 'Nurse@123')
s, d = call('POST', '/sessions/SS-001/vitals/', token=nurse2_tok, body={
    'recorded_at': '2026-09-01T10:00:00Z', 'spo2': 98})
check('non-assigned nurse nested POST vital -> 403', s == 403, str(d))
vital_id = vital['id']
s, d = call('GET', f'/vital-signs/{vital_id}/', token=doctor_tok)
check('flat GET vital by id', s == 200 and d['id'] == vital_id)
s, d = call('PATCH', f'/vital-signs/{vital_id}/', token=nurse_tok, body={'notes': 'Da cap nhat'})
check('assigned nurse flat PATCH vital', s == 200 and d['notes'] == 'Da cap nhat')
s, d = call('PATCH', f'/vital-signs/{vital_id}/', token=nurse2_tok, body={'notes': 'hack'})
check('non-assigned nurse flat PATCH vital -> 403', s == 403)
s, d = call('PUT', f'/vital-signs/{vital_id}/', token=doctor_tok, body={
    'recorded_at': '2026-09-01T08:35:00Z', 'systolic_bp': 122, 'diastolic_bp': 82,
    'heart_rate': 77, 'spo2': 98, 'temperature': 36.5, 'notes': 'PUT update'})
check('doctor flat PUT vital', s == 200 and d['systolic_bp'] == 122)
s, d = call('GET', '/sessions/SS-001/vitals/', token=admin_tok)
check('nested list vitals has 2 records', s == 200 and len(d) == 2)
s, d = call('DELETE', f'/vital-signs/{vital_id}/', token=admin_tok)
check('admin flat DELETE vital', s == 204)
s, d = call('DELETE', f'/vital-signs/', token=doctor_tok)
check('DELETE on list not routable (405/404)', s in (405, 404))

print('== BLOOD SAMPLE CRUD ==')
s, bs = call('POST', '/blood-samples/', token=doctor_tok, body={
    'sample_id': 'BL-001', 'patient': 'PT-001', 'collection_date': '2026-08-28T09:00:00Z',
    'hemoglobin_level': 11.5, 'potassium_level': 4.8, 'notes': 'Lay mau tai nha'})
check('doctor create blood sample', s == 201 and bs['created_by'] == doctor_login['user']['id'], str(bs))
s, d = call('POST', '/blood-samples/', token=doctor_tok, body={
    'sample_id': 'BL-FUT', 'patient': 'PT-001', 'collection_date': '2999-01-01T09:00:00Z',
    'hemoglobin_level': 11.5, 'potassium_level': 4.8})
check('future collection_date -> 400', s == 400 and 'collection_date' in d)
s, d = call('POST', '/blood-samples/', token=doctor_tok, body={
    'sample_id': 'BL-NEG', 'patient': 'PT-001', 'collection_date': '2026-08-28T09:00:00Z',
    'hemoglobin_level': -1, 'potassium_level': 4.8})
check('negative hemoglobin -> 400', s == 400 and 'hemoglobin_level' in d)
s, d = call('POST', '/blood-samples/', token=nurse_tok, body={
    'sample_id': 'BL-N', 'patient': 'PT-001', 'collection_date': '2026-08-28T09:00:00Z',
    'hemoglobin_level': 11.5, 'potassium_level': 4.8})
check('nurse create blood sample -> 403', s == 403)
s, d = call('GET', '/blood-samples/?patient=PT-001&search=BL-001', token=nurse_tok)
check('blood sample filter+search (nurse read)', s == 200 and d['count'] == 1)

print('== DASHBOARD ==')
s, d = call('GET', '/dashboard/summary/', token=admin_tok)
check('summary fields', s == 200 and all(k in d for k in
      ('total_patients', 'total_staff', 'total_sessions', 'active_machines')), str(d))
check('summary counts sane', d['total_patients'] >= 1 and d['total_staff'] >= 3)
s, d = call('GET', '/dashboard/dialysis-stats/', token=doctor_tok)
check('dialysis-stats shape', s == 200 and isinstance(d, list) and len(d) == 7 and 'date' in d[0])
s, d = call('GET', '/dashboard/machine-stats/', token=nurse_tok)
check('machine-stats shape', s == 200 and isinstance(d, list) and any(x['status'] == 'AVAILABLE' for x in d))

print('== PAGINATION ==')
s, d = call('GET', '/patients/?page=1&page_size=5', token=admin_tok)
check('pagination envelope', all(k in d for k in ('count', 'next', 'previous', 'results')))

print('== PROTECT constraints ==')
s, d = call('DELETE', f'/departments/{dept_id}/', token=admin_tok)
check('delete department in use -> 400/403/409', s in (400, 403, 409), f'status={s}')
s, d = call('DELETE', '/patients/PT-001/', token=admin_tok)
check('delete patient with sessions -> protected', s in (400, 403, 409), f'status={s}')

print()
print(f'PASSED: {len(PASSED)}  FAILED: {len(FAILED)}')
if FAILED:
    print('FAILED LIST:')
    for f in FAILED:
        print(' -', f)
