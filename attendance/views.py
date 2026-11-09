import json
import csv
import random
import threading
import uuid
from datetime import timedelta

import cv2
import numpy as np
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Count
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import AdminAuditLog, AttendanceRecord, AttendanceSession, Course, Student

_recognizer = None
_recognizer_lock = threading.Lock()
CHALLENGE_STATES = {}
CHALLENGE_LOCK = threading.Lock()


def _get_recognizer():
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            from src.recognition import FaceRecognizer
            _recognizer = FaceRecognizer()
    return _recognizer


def _decode_frame(upload):
    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read camera frame")
    return frame


@ensure_csrf_cookie
def home(request):
    active_sessions = [
        session for session in AttendanceSession.objects.select_related("course").filter(is_active=True)
        if session.currently_open
    ]
    return render(request, "attendance/home.html", {"active_sessions": active_sessions})


def student_login(request):
    login_type = request.POST.get("login_type", request.GET.get("type", "student"))
    if login_type == "admin":
        return unified_login(request)
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        user = authenticate(request, username=username, password=request.POST.get("password", ""))
        if user is None:
            messages.error(request, "Username or password is incorrect. If you just created this account, verify its password in Django Admin.")
        elif user.is_staff:
            messages.error(request, "This is a staff/admin account. Sign in with a regular student account.")
        elif not Student.objects.filter(user=user).exists():
            # Convenience for accounts created directly in Django Admin: if a
            # student profile has the same username as student_number and no
            # login linked yet, attach this user on first successful login.
            linked = Student.objects.filter(user__isnull=True, student_number=username).first()
            if linked:
                linked.user = user
                linked.full_name = user.get_full_name().strip() or user.get_username()
                linked.save(update_fields=["user", "full_name"])
            else:
                messages.error(request, "No student profile is linked to this account. Ask an administrator to link your user under Attendance → Students.")
                return render(request, "attendance/login.html")
            login(request, user)
            return redirect("student-dashboard")
        else:
            login(request, user)
            return redirect("student-dashboard")
    return render(request, "attendance/login.html", {"login_type": "student"})


def unified_login(request):
    login_type = request.POST.get("login_type") or request.GET.get("type", "student")
    if login_type == "admin":
        next_url = request.GET.get("next") or "/portal/admin/"
        if request.method == "POST":
            user = authenticate(request, username=request.POST.get("username", "").strip(), password=request.POST.get("password", ""))
            if user and user.is_active and user.is_staff:
                login(request, user)
                return redirect(next_url)
            messages.error(request, "Invalid admin credentials. Use an active staff account.")
        return render(request, "attendance/login.html", {"login_type": "admin", "next_url": next_url})
    if request.method == "POST":
        # Pass the request through the student flow without recursing into the
        # unified endpoint for the default student selection.
        user = authenticate(request, username=request.POST.get("username", "").strip(), password=request.POST.get("password", ""))
        if user is None:
            messages.error(request, "Username or password is incorrect.")
        elif user.is_staff:
            messages.error(request, "Choose Admin / Staff for this account.")
        elif not Student.objects.filter(user=user).exists():
            messages.error(request, "This account is not linked to a student profile. Ask an administrator to link it under Attendance → Students.")
        else:
            login(request, user)
            return redirect("student-dashboard")
    return render(request, "attendance/login.html", {"login_type": "student"})


def student_logout(request):
    logout(request)
    return redirect("home")


@login_required
def student_dashboard(request):
    if request.user.is_staff or not hasattr(request.user, "student"):
        return redirect("home")
    records = AttendanceRecord.objects.filter(student=request.user.student).select_related("session__course")
    summary = []
    for course in Course.objects.order_by("code"):
        total = AttendanceSession.objects.filter(course=course).count()
        attended = records.filter(session__course=course, status__in=[AttendanceRecord.Status.PRESENT, AttendanceRecord.Status.LATE]).count()
        if total:
            summary.append({"course": course, "attended": attended, "total": total, "percentage": round(attended * 100 / total)})
    return render(request, "attendance/student_dashboard.html", {"student": request.user.student, "records": records, "summary": summary})


@user_passes_test(lambda user: user.is_authenticated and user.is_staff, login_url="/admin/login/")
def admin_dashboard(request):
    if request.method == "POST" and request.POST.get("action") == "start-today-session":
        course = Course.objects.filter(pk=request.POST.get("session_course"), is_active=True).first()
        if not course:
            messages.error(request, "Choose an active course first.")
            return redirect("admin-dashboard")
        now = timezone.localtime()
        title = request.POST.get("session_title", "").strip()
        end_time_text = request.POST.get("ends_at", "").strip()
        if not title or not end_time_text:
            messages.error(request, "Enter a session name and end time.")
            return redirect("admin-dashboard")
        try:
            end_time = timezone.datetime.strptime(end_time_text, "%H:%M").time()
        except ValueError:
            messages.error(request, "Enter a valid session end time.")
            return redirect("admin-dashboard")
        ends_at = timezone.make_aware(timezone.datetime.combine(timezone.localdate(), end_time))
        if ends_at <= now:
            messages.error(request, "The end time must be later than now.")
            return redirect("admin-dashboard")
        late_text = request.POST.get("late_after", "").strip()
        late_after = None
        if late_text:
            try:
                late_clock = timezone.datetime.strptime(late_text, "%H:%M").time()
            except ValueError:
                messages.error(request, "Enter a valid late cutoff time.")
                return redirect("admin-dashboard")
            late_after = timezone.make_aware(timezone.datetime.combine(timezone.localdate(), late_clock))
            if late_after <= now or late_after >= ends_at:
                messages.error(request, "Late cutoff must be after now and before the session end time.")
                return redirect("admin-dashboard")
        session, created = AttendanceSession.objects.get_or_create(
            course=course,
            title=title,
            defaults={"starts_at": now, "ends_at": ends_at, "late_after": late_after, "is_active": True},
        )
        if not created:
            session.starts_at = now
            session.ends_at = ends_at
            session.late_after = late_after
            session.is_active = True
            session.save(update_fields=["starts_at", "ends_at", "late_after", "is_active"])
        AdminAuditLog.objects.create(actor=request.user, action="session_started", target_type="AttendanceSession", target_id=str(session.pk), details={"course": course.code, "title": title, "late_after": late_after.isoformat() if late_after else None})
        messages.success(request, f"Today's attendance session for {course.code} is active.")
        return redirect("admin-dashboard")

    records = AttendanceRecord.objects.select_related("student", "session__course")
    course_id = request.GET.get("course", "").strip()
    session_id = request.GET.get("session", "").strip()
    date_from = request.GET.get("from", "").strip()
    date_to = request.GET.get("to", "").strip()
    search = request.GET.get("q", "").strip()
    if course_id:
        records = records.filter(session__course_id=course_id)
    if session_id:
        records = records.filter(session_id=session_id)
    if date_from:
        records = records.filter(marked_at__date__gte=date_from)
    if date_to:
        records = records.filter(marked_at__date__lte=date_to)
    if search:
        records = records.filter(Q(student__full_name__icontains=search) | Q(student__student_number__icontains=search))

    if request.GET.get("export") == "csv":
        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="attendance-report.csv"'
        response.write("\ufeff")  # Excel-friendly UTF-8 BOM
        writer = csv.writer(response)
        writer.writerow(["Roll no.", "Name", "Username", "Course code", "Course name", "Session", "Marked at", "Status"])
        for record in records.order_by("-marked_at"):
            writer.writerow([
                record.student.student_number,
                record.student.full_name,
                record.student.user.get_username() if record.student.user_id else "",
                record.session.course.code,
                record.session.course.name,
                record.session.title,
                timezone.localtime(record.marked_at).strftime("%Y-%m-%d %H:%M:%S"),
                record.get_status_display(),
            ])
        return response

    context = {
        "students": Student.objects.count(),
        "courses": Course.objects.count(),
        "sessions": AttendanceSession.objects.count(),
        "recent_records": records.order_by("-marked_at")[:100],
        "course_options": Course.objects.order_by("code"),
        "session_options": AttendanceSession.objects.select_related("course").order_by("-starts_at"),
        "selected_course": course_id,
        "selected_session": session_id,
        "date_from": date_from,
        "date_to": date_to,
        "search": search,
        "filtered_count": records.count(),
        "course_stats": [],
        "student_stats": [],
    }
    active_student_count = Student.objects.filter(is_active=True).count()
    for course in Course.objects.order_by("code"):
        session_count = AttendanceSession.objects.filter(course=course).count()
        possible = session_count * active_student_count
        attended = AttendanceRecord.objects.filter(
            session__course=course,
            status__in=[AttendanceRecord.Status.PRESENT, AttendanceRecord.Status.LATE],
        ).count()
        context["course_stats"].append({
            "course": course,
            "attended": attended,
            "possible": possible,
            "percentage": round(attended * 100 / possible) if possible else 0,
        })
    total_sessions = AttendanceSession.objects.count()
    for student in Student.objects.filter(is_active=True).order_by("full_name"):
        attended = AttendanceRecord.objects.filter(
            student=student,
            status__in=[AttendanceRecord.Status.PRESENT, AttendanceRecord.Status.LATE],
        ).count()
        context["student_stats"].append({
            "student": student,
            "attended": attended,
            "total": total_sessions,
            "percentage": round(attended * 100 / total_sessions) if total_sessions else 0,
        })
    return render(request, "attendance/admin_dashboard.html", context)


@user_passes_test(lambda user: user.is_authenticated and user.is_staff, login_url="/admin/login/")
def admin_enroll(request):
    if request.method == "POST":
        try:
            student_number = request.POST.get("student_number", "").strip()
            full_name = request.POST.get("full_name", "").strip()
            if not student_number or not full_name:
                raise ValueError("Student roll number and name are required.")
            frame = _decode_frame(request.FILES["frame"])
            embedding, _ = _get_recognizer().embed_face(frame)
            if embedding is None:
                raise ValueError("No face detected. Center your face and capture again.")
            student, created = Student.objects.update_or_create(
                student_number=student_number,
                defaults={"full_name": full_name, "face_embedding": embedding.astype(float).tolist(), "is_active": True},
            )
            AdminAuditLog.objects.create(actor=request.user, action="enrolled" if created else "reenrolled", target_type="Student", target_id=str(student.pk), details={"roll_number": student_number, "name": full_name})
            messages.success(request, f"Face enrollment saved for {full_name}.")
        except (KeyError, ValueError) as exc:
            messages.error(request, str(exc))
    return render(request, "attendance/admin_enroll.html")


@require_POST
def attendance_start(request):
    session_id = request.POST.get("session_id")
    session = AttendanceSession.objects.filter(pk=session_id, is_active=True).select_related("course").first()
    if not session or not session.currently_open:
        return JsonResponse({"ok": False, "message": "There is no active attendance session."}, status=409)
    try:
        frame = _decode_frame(request.FILES["frame"])
        recognizer = _get_recognizer()
        embedding, _ = recognizer.embed_face(frame)
        if embedding is None:
            return JsonResponse({"ok": False, "message": "No face detected. Center your face and try again."}, status=400)
        gallery = [(s.full_name, s.face_embedding, s.student_number) for s in Student.objects.filter(is_active=True, face_embedding__isnull=False)]
        name, score, student_number = recognizer.identify(embedding, gallery)
        if score < 0.70:
            return JsonResponse({"ok": False, "message": "Face not recognized."}, status=403)
        student = Student.objects.get(student_number=student_number, is_active=True)
        if AttendanceRecord.objects.filter(session=session, student=student).exists():
            return JsonResponse({"ok": False, "message": "Attendance is already marked for this session."}, status=409)
        # Smile currently needs a neutral-expression calibration phase; avoid
        # offering it until that phase is implemented in the kiosk UI.
        challenge = random.choice(["blink2", "yaw_right", "mouth_open"])
        from src.config import load_config
        from src.liveness import LivenessChecker
        config = load_config(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
        checker = LivenessChecker(config)
        checker.begin(challenge)
        token = uuid.uuid4().hex
        with CHALLENGE_LOCK:
            CHALLENGE_STATES[token] = {"student_id": student.pk, "session_id": session.pk, "score": float(score), "challenge": challenge, "checker": checker, "created": timezone.now()}
        return JsonResponse({"ok": True, "token": token, "challenge": challenge, "name": name})
    except (KeyError, ValueError) as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, status=400)


@require_POST
def attendance_challenge(request):
    token = request.POST.get("token", "")
    with CHALLENGE_LOCK:
        state = CHALLENGE_STATES.get(token)
    if not state:
        return JsonResponse({"ok": False, "message": "Challenge expired. Start again."}, status=404)
    if (timezone.now() - state["created"]).total_seconds() > 25:
        with CHALLENGE_LOCK:
            CHALLENGE_STATES.pop(token, None)
        return JsonResponse({"ok": False, "message": "Challenge timed out. Start again."}, status=408)
    try:
        frame = _decode_frame(request.FILES["frame"])
        passed, message = state["checker"].process(frame)
        if not passed:
            if message == "Timeout":
                with CHALLENGE_LOCK:
                    CHALLENGE_STATES.pop(token, None)
                return JsonResponse({"ok": False, "message": "Challenge timed out. Start again."}, status=408)
            return JsonResponse({"ok": True, "complete": False, "message": message})
        try:
            attendance_session = AttendanceSession.objects.get(pk=state["session_id"])
            AttendanceRecord.objects.create(
                session_id=state["session_id"], student_id=state["student_id"],
                similarity=state["score"], liveness_challenge=state["challenge"],
                status=AttendanceRecord.Status.LATE if attendance_session.late_after and timezone.now() >= attendance_session.late_after else AttendanceRecord.Status.PRESENT,
            )
        except Exception:
            return JsonResponse({"ok": False, "message": "Attendance is already marked for this session."}, status=409)
        with CHALLENGE_LOCK:
            CHALLENGE_STATES.pop(token, None)
        return JsonResponse({"ok": True, "complete": True, "message": "Attendance marked successfully."})
    except (KeyError, ValueError) as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, status=400)
