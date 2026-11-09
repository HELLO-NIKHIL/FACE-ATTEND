from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import Group
from django import forms
from .models import AdminAuditLog, AttendanceRecord, AttendanceSession, Course, Student

# Groups are not used by this project; keep Django authentication intact but
# remove the unused Groups screen from the admin navigation.
admin.site.unregister(Group)


class ProjectUserCreationForm(UserCreationForm):
    ACCOUNT_TYPES = (("student", "Student"), ("staff", "Admin / staff"))
    account_type = forms.ChoiceField(choices=ACCOUNT_TYPES, label="Account type", initial="student")

    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = ("username", "first_name", "last_name", "email")


class ProjectUserAdmin(UserAdmin):
    add_form = ProjectUserCreationForm
    add_fieldsets = ((None, {"classes": ("wide",), "fields": ("username", "first_name", "last_name", "email", "account_type", "password1", "password2")}),)

    def save_model(self, request, obj, form, change):
        if not change:
            obj.is_staff = form.cleaned_data.get("account_type") == "staff"
            obj.is_superuser = False
        super().save_model(request, obj, form, change)


User = get_user_model()
admin.site.unregister(User)
admin.site.register(User, ProjectUserAdmin)


def write_audit(request, action, obj):
    AdminAuditLog.objects.create(actor=request.user, action=action, target_type=obj.__class__.__name__, target_id=str(obj.pk), details={"object": str(obj)[:200]})


class StudentAdminForm(forms.ModelForm):
    full_name = forms.CharField(required=False, widget=forms.HiddenInput())

    class Meta:
        model = Student
        fields = "__all__"
        widgets = {"student_number": forms.HiddenInput()}

    def clean(self):
        cleaned_data = super().clean()
        user = cleaned_data.get("user")
        if user is None and self.instance and self.instance.pk:
            user = self.instance.user
        if user:
            cleaned_data["full_name"] = user.get_full_name().strip() or user.get_username()
            if not cleaned_data.get("student_number"):
                # Usernames need not be unique as student identifiers; generate
                # a unique internal ID instead of failing on a duplicate.
                cleaned_data["student_number"] = f"STU-{user.pk}"
        return cleaned_data


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    form = StudentAdminForm
    list_display = ("student_number", "full_name", "user", "is_active")
    search_fields = ("student_number", "full_name", "user__username")
    list_filter = ("is_active",)
    # Keep embeddings out of forms; camera enrollment manages biometric data.
    exclude = ("face_embedding",)

    def save_model(self, request, obj, form, change):
        if obj.user:
            obj.full_name = obj.user.get_full_name().strip() or obj.user.get_username()
            if not obj.student_number:
                obj.student_number = f"STU-{obj.user_id}"
        super().save_model(request, obj, form, change)
        write_audit(request, "updated" if change else "created", obj)

    def delete_model(self, request, obj):
        write_audit(request, "deleted", obj)
        super().delete_model(request, obj)

@admin.register(AttendanceSession)
class AttendanceSessionAdmin(admin.ModelAdmin):
    list_display = ("title", "course", "starts_at", "ends_at", "is_active")
    list_filter = ("is_active", "course")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        write_audit(request, "updated" if change else "created", obj)

    def delete_model(self, request, obj):
        write_audit(request, "deleted", obj)
        super().delete_model(request, obj)


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "is_active")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        write_audit(request, "updated" if change else "created", obj)

    def delete_model(self, request, obj):
        write_audit(request, "deleted", obj)
        super().delete_model(request, obj)
@admin.register(AttendanceRecord)
class AttendanceRecordAdmin(admin.ModelAdmin):
    list_display = ("student", "session", "marked_at", "status", "similarity")
    list_filter = ("status", "session__course", "marked_at")
    search_fields = ("student__full_name", "student__student_number")
    list_editable = ("status",)

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        write_audit(request, "updated" if change else "created", obj)

    def delete_model(self, request, obj):
        write_audit(request, "deleted", obj)
        super().delete_model(request, obj)

@admin.register(AdminAuditLog)
class AdminAuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "actor", "action", "target_type", "target_id")
    list_filter = ("action", "target_type", "created_at")
    search_fields = ("actor__username", "action", "target_type", "target_id")
    readonly_fields = ("actor", "action", "target_type", "target_id", "details", "created_at")
