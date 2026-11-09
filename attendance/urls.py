from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("student/login/", views.student_login, name="student-login"),
    path("student/logout/", views.student_logout, name="student-logout"),
    path("student/", views.student_dashboard, name="student-dashboard"),
    path("portal/admin/", views.admin_dashboard, name="admin-dashboard"),
    path("portal/admin/reports/", views.admin_dashboard, name="attendance-report"),
    path("portal/admin/enroll/", views.admin_enroll, name="admin-enroll"),
    path("api/attendance/start/", views.attendance_start, name="attendance-start"),
    path("api/attendance/challenge/", views.attendance_challenge, name="attendance-challenge"),
]
