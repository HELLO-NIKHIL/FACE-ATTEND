# FACE-ATTEND

A college attendance web app built with Django, PostgreSQL, FaceNet, and MediaPipe. Students use a camera kiosk to mark attendance and can sign in to see their attendance history. Staff create courses and sessions, enroll students, and review or export reports.

This guide starts from a new computer. Follow the sections in order.

## 1. Install the required software

Install these before setting up the project:

1. **Python 3.11**: <https://www.python.org/downloads/>. On Windows, select **Add Python to PATH** during installation.
2. **PostgreSQL**: <https://www.postgresql.org/download/>. The installer includes pgAdmin, a graphical tool for managing PostgreSQL.
3. **Git** (only if downloading with Git): <https://git-scm.com/downloads>.

During PostgreSQL installation, remember the password you set for the `postgres` database administrator account.

## 2. Get the project folder

Either download the project from GitHub and extract it, or use Terminal/PowerShell:

```bash
git clone <PROJECT_GITHUB_URL>
cd FACE-ATTEND
```

If you downloaded a ZIP, extract it, open Terminal/PowerShell in the extracted project folder, and continue there. This folder must contain `manage.py`.

## 3. Create the PostgreSQL database

Open **pgAdmin** and connect to your local PostgreSQL server using the `postgres` password from installation.

1. Right-click **Login/Group Roles → Create → Login/Group Role**.
2. Set the name to `attendance_app`. On the **Definition** tab, choose a password and remember it. On **Privileges**, enable **Can login**. Save.
3. Right-click **Databases → Create → Database**. Name it `attendance_system`, set owner to `attendance_app`, and save.

Use the password you chose in the next step. Do not use the `postgres` administrator password as the app password.

## 4. Configure the project

Copy `.env.example` to a new file named `.env` in the project folder. Open `.env` in a text editor and replace the sample database password and secret key. Example:

```env
DATABASE_URL=postgresql+psycopg://attendance_app:YOUR_DATABASE_PASSWORD@localhost:5432/attendance_system
DJANGO_SECRET_KEY=put-a-long-random-private-value-here
DJANGO_DEBUG=true
DJANGO_TIME_ZONE=Asia/Kolkata
```

Replace `YOUR_DATABASE_PASSWORD` with the password chosen for `attendance_app`. Avoid special characters in that password unless they are URL-encoded. Keep `.env` private; never upload it or share its passwords. The `.env.example` file is only a template.

## 5. Install Python packages

Run commands from the project folder.

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The FaceNet package may download pretrained model weights the first time it runs; an internet connection may be needed for the first face enrollment or kiosk use.

## 6. Prepare the database and create the first staff account

With the virtual environment active, run:

```bash
python manage.py migrate
python manage.py createsuperuser
```

Choose a username, email, and password when prompted. This account is your first administrator/staff login. The password is stored securely by Django; it is not saved in the project files.

## 7. Start the website

```bash
python manage.py runserver
```

Keep this Terminal/PowerShell window open while using the site. In your browser open:

- Attendance kiosk: <http://127.0.0.1:8000/>
- Unified login: <http://127.0.0.1:8000/student/login/>
- Staff dashboard: <http://127.0.0.1:8000/portal/admin/>
- Django data management: <http://127.0.0.1:8000/admin/>

To stop the development server, return to the terminal and press **Ctrl+C**.

## 8. First-time setup and daily use

### Sign in as staff

Open <http://127.0.0.1:8000/portal/admin/> and sign in with the superuser created in Step 6. Staff can also open Django data management at <http://127.0.0.1:8000/admin/>. The custom dashboard is for sessions/reports; Django Admin manages the underlying records.

### Add a course

In the custom dashboard click **Manage data**. Under **Attendance → Courses**, add the course code and name. Return to the custom dashboard.

### Start an attendance session

On the dashboard, use **Start today's session**. Choose the course, enter a session name, set the time attendance closes, and optionally choose **Mark late from**. The session starts immediately. Attendance captured at or after the cutoff is automatically marked Late; otherwise it is Present.

### Enroll a student

From the dashboard click **Enroll student**. Enter the student's roll number and name, allow camera access, capture their face, and save. This creates the attendance/face profile, but does not create a login account.

### Give a student a portal login

In Django Admin go to **Authentication and Authorization → Users → Add user**. Choose **Student**, then set the username and password. Next go to **Attendance → Students**, edit the enrolled student's profile, select the new user in the **User** field, and save. The student can then sign in at <http://127.0.0.1:8000/student/login/>. Student accounts do not need Staff status.

### Mark attendance

Open <http://127.0.0.1:8000/>. Select the active session, click **Start attendance**, allow the browser to use the camera, and follow the prompt (blink, turn your head, or open your mouth). A student can be recorded once per session. If the session does not appear, staff must start an open session from the dashboard.

### View and export reports

The staff dashboard lets staff filter attendance by course, session, dates, and student. Use **Export → Download CSV** to download a spreadsheet-friendly report or **Print** for a printable page. Attendance status can be corrected in Django Admin under **Attendance → Attendance records**. Audit activity is under **Attendance → Admin audit logs**.

## Troubleshooting

- **Database connection error:** Check PostgreSQL is running, the database and login role exist, and `.env` has the correct credentials/database name.
- **Camera permission does not appear:** Click the kiosk button first. Use `127.0.0.1` or `localhost` for local development, then allow camera access in browser site settings.
- **Start attendance is unavailable or no session appears:** A staff member must start a session whose current time is between its start and end times.
- **Student login is rejected:** Create a regular Student user and link it to the correct record in **Attendance → Students**.
- **Port 8000 is already in use:** Stop the other server with Ctrl+C, or run `python manage.py runserver 8001` and open `http://127.0.0.1:8001/`.

## Keep private before uploading

Do not upload `.env`, database backups/dumps, real student data, or face data. `.gitignore` excludes common local files, but check carefully before uploading a folder manually. Users and passwords live in PostgreSQL; someone installing their own copy should run migrations and create their own superuser.

## Technical overview

- Django handles pages, authentication, permissions, and database models.
- PostgreSQL stores accounts, student profiles, courses, sessions, and attendance.
- `facenet-pytorch` MTCNN detects faces and FaceNet generates face embeddings; MediaPipe performs the liveness checks.
- The supported app runs with `python manage.py runserver`; it does not require Streamlit.
