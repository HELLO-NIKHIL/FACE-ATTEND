from django.conf import settings
from django.db import models
from django.utils import timezone


class Student(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    student_number = models.CharField("Student roll number", max_length=40, unique=True, blank=True)
    full_name = models.CharField(max_length=160)
    face_embedding = models.JSONField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["full_name"]

    def __str__(self):
        return f"{self.full_name} (Roll no. {self.student_number})"


class Course(models.Model):
    code = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=160)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.code} — {self.name}"


class AttendanceSession(models.Model):
    course = models.ForeignKey(Course, on_delete=models.PROTECT, related_name="sessions")
    title = models.CharField(max_length=160)
    starts_at = models.DateTimeField()
    ends_at = models.DateTimeField()
    late_after = models.DateTimeField(null=True, blank=True, help_text="Attendance at or after this time is late.")
    is_active = models.BooleanField(default=False)

    class Meta:
        ordering = ["-starts_at"]
        constraints = [models.CheckConstraint(condition=models.Q(ends_at__gt=models.F("starts_at")), name="session_end_after_start")]

    def __str__(self):
        return f"{self.course.code}: {self.title}"

    @property
    def currently_open(self):
        now = timezone.now()
        return self.is_active and self.starts_at <= now < self.ends_at


class AttendanceRecord(models.Model):
    class Status(models.TextChoices):
        PRESENT = "present", "Present"
        LATE = "late", "Late"
        ABSENT = "absent", "Absent"

    session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE, related_name="records")
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="attendance_records")
    marked_at = models.DateTimeField(default=timezone.now)
    similarity = models.FloatField()
    liveness_challenge = models.CharField(max_length=40, default="face-match")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PRESENT)

    class Meta:
        ordering = ["-marked_at"]
        constraints = [
            models.UniqueConstraint(fields=["session", "student"], name="one_attendance_per_session"),
            models.CheckConstraint(condition=models.Q(similarity__gte=0, similarity__lte=1), name="similarity_between_zero_one"),
        ]


class AdminAuditLog(models.Model):
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=80)
    target_type = models.CharField(max_length=80)
    target_id = models.CharField(max_length=80, blank=True)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
