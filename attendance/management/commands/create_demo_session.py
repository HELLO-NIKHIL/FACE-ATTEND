from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from attendance.models import AttendanceSession, Course


class Command(BaseCommand):
    help = "Create today's general active session for a local demo."

    def handle(self, *args, **options):
        now = timezone.now()
        course, _ = Course.objects.get_or_create(
            code="GENERAL", defaults={"name": "General Attendance", "is_active": True}
        )
        title = f"Demo attendance {timezone.localdate().isoformat()}"
        session, created = AttendanceSession.objects.get_or_create(
            course=course,
            title=title,
            defaults={
                "starts_at": now - timedelta(minutes=1),
                "ends_at": now + timedelta(hours=12),
                "is_active": True,
            },
        )
        if not created and not session.currently_open:
            session.starts_at = now - timedelta(minutes=1)
            session.ends_at = now + timedelta(hours=12)
            session.is_active = True
            session.save(update_fields=["starts_at", "ends_at", "is_active"])
        self.stdout.write(self.style.SUCCESS(f"Active demo session ready: {session}"))
