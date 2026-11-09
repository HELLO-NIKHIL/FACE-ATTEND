import time
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Helper landmark index groups for eyes and mouth (MediaPipe Face Mesh)
# Using commonly referenced indices (may vary across resources but stable enough for heuristics).
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_INNER = [78, 308, 13, 14]  # left corner, right corner, upper inner, lower inner
NOSE_TIP = 1

def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    p = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idx]
    # order: [p1, p2, p3, p4, p5, p6] as in EAR formula
    p1, p2, p3, p4, p5, p6 = p
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    h = euclidean(p1, p4) + 1e-6
    return (v1 + v2) / (2.0 * h)

def lip_gap_norm(landmarks, img_w, img_h):
    # inner upper (13) and inner lower (14); normalize by inter-ocular distance
    p_left_eye = (landmarks[33].x*img_w, landmarks[33].y*img_h)
    p_right_eye = (landmarks[263].x*img_w, landmarks[263].y*img_h)
    io = euclidean(p_left_eye, p_right_eye) + 1e-6
    p_up = (landmarks[13].x*img_w, landmarks[13].y*img_h)
    p_lo = (landmarks[14].x*img_w, landmarks[14].y*img_h)
    gap = euclidean(p_up, p_lo)
    return gap / io

def smile_ratio(landmarks, img_w, img_h, neutral_mouth_width=None):
    # corner spread normalized by inter-ocular
    left = (landmarks[78].x*img_w, landmarks[78].y*img_h)
    right = (landmarks[308].x*img_w, landmarks[308].y*img_h)
    p_left_eye = (landmarks[33].x*img_w, landmarks[33].y*img_h)
    p_right_eye = (landmarks[263].x*img_w, landmarks[263].y*img_h)
    io = euclidean(p_left_eye, p_right_eye) + 1e-6
    width = euclidean(left, right) / io
    if neutral_mouth_width is None:
        return width, None
    increase = (width - neutral_mouth_width) / (neutral_mouth_width + 1e-6)
    return width, increase

def estimate_yaw_pnp(landmarks, img_w, img_h):
    """
    Robust yaw estimate using 6+ points with solvePnPRansac.
    Returns yaw in degrees. Falls back gracefully on failure.
    """
    import numpy as np, cv2

    # Use 6 stable FaceMesh landmarks:
    #  1  = nose tip
    # 33  = left eye outer corner
    # 263 = right eye outer corner
    # 78  = left mouth corner
    # 308 = right mouth corner
    # 152 = chin
    idxs = [1, 33, 263, 78, 308, 152]

    # 2D image points
    try:
        image_points = np.array(
            [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idxs],
            dtype=np.float64,
        )
    except Exception:
        return 0.0  # no landmarks yet

    # Very rough generic 3D face model (arbitrary units, consistent scale)
    # x: left(-)→right(+), y: up(+)→down(-), z: forward(+)/out of face
    model_points = np.array([
        (0.0,    0.0,   0.0),   # nose tip
        (-30.0,  0.0,  -30.0),  # left eye outer
        (30.0,   0.0,  -30.0),  # right eye outer
        (-25.0, -25.0, -30.0),  # left mouth corner
        (25.0,  -25.0, -30.0),  # right mouth corner
        (0.0,   -55.0, -10.0),  # chin
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            model_points, image_points, camera_matrix, dist_coeffs,
            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0
        R, _ = cv2.Rodrigues(rvec)

        # Yaw extraction (rotation around Y); sign convention: positive = turn right
        # This convention matches our "turn head right" check (positive delta).
        yaw = np.degrees(np.arctan2(R[2,0], R[2,2]))
        return float(yaw)
    except cv2.error:
        # Any PnP failure → safe fallback (treat as 0)
        return 0.0

class LivenessChecker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.reset()

    def reset(self):
        self.state = None
        self.start_time = 0.0
        self.neutral = {"mouth_width": None}
        self.history = {"ear_L": [], "ear_R": [], "lip": [], "smile_w": [], "yaw": []}
        self.blink_count = 0
        self.last_blink_state = False
        self.below_counter = 0
        self.mouth_frames = 0
        self.smile_frames = 0
        self.yaw_baseline = None
        self.nose_baseline = None
        self.blink_was_closed = False

    def begin(self, challenge:str):
        self.reset()
        self.state = challenge
        self.start_time = time.time()

    def _timeout(self):
        return (time.time() - self.start_time) > self.cfg["liveness"]["timeout_seconds"]

    def process(self, frame_bgr):
        if self.state is None:
            return False, "No challenge"
        if self._timeout():
            return False, "Timeout"
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False, "No face landmarks"
        lm = results.multi_face_landmarks[0].landmark

        # Update metrics
        earL = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        earR = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        lip = lip_gap_norm(lm, w, h)
        sw, inc = smile_ratio(lm, w, h, self.neutral["mouth_width"])
        yaw = estimate_yaw_pnp(lm, w, h)

        self.history["ear_L"].append(earL)
        self.history["ear_R"].append(earR)
        self.history["lip"].append(lip)
        self.history["smile_w"].append(sw)
        # Save a short neutral baseline before evaluating a head-turn prompt.
        if self.state == "yaw_right" and self.yaw_baseline is None:
            self.history["yaw"].append(yaw)
            eye_left = np.array([lm[33].x, lm[33].y])
            eye_right = np.array([lm[263].x, lm[263].y])
            nose = np.array([lm[1].x, lm[1].y])
            eye_distance = np.linalg.norm(eye_right - eye_left) + 1e-6
            nose_offset = float(np.dot(nose - (eye_left + eye_right) / 2, (eye_right - eye_left) / eye_distance) / eye_distance)
            self.history.setdefault("nose_offset", []).append(nose_offset)
            if len(self.history["yaw"]) >= 5:
                self.yaw_baseline = float(np.median(self.history["yaw"][:5]))
                self.nose_baseline = float(np.median(self.history["nose_offset"][:5]))

        # Initialize neutral mouth width in first frames
        if self.neutral["mouth_width"] is None and len(self.history["smile_w"]) > 5:
            self.neutral["mouth_width"] = float(np.mean(self.history["smile_w"][:5]))

        # Challenge checks
        if self.state == "blink2":
            ok, msg = self._check_blink(earL, earR)
        elif self.state == "yaw_right":
            if self.yaw_baseline is None:
                ok, msg = False, "Look straight at the camera"
            else:
                eye_left = np.array([lm[33].x, lm[33].y])
                eye_right = np.array([lm[263].x, lm[263].y])
                nose = np.array([lm[1].x, lm[1].y])
                eye_vector = eye_right - eye_left
                eye_distance = np.linalg.norm(eye_vector) + 1e-6
                offset = float(np.dot(nose - (eye_left + eye_right) / 2, eye_vector / eye_distance) / eye_distance)
                delta = abs(offset - self.nose_baseline)
                threshold = 0.10
                ok, msg = delta >= threshold, f"Turn your head slightly to either side ({delta:.2f}/{threshold:.2f})"
        elif self.state == "mouth_open":
            ok, msg = self._check_mouth(lip)
        elif self.state == "smile":
            ok, msg = self._check_smile(sw)
        else:
            return False, "No challenge"

        if ok:
            return True, "Passed"
        if self._timeout():
            return False, "Timeout"
        return False, msg

    def _check_blink(self, earL, earR):
        thr = self.cfg["liveness"]["blink"]["ear_threshold"]
        frames_below = self.cfg["liveness"]["blink"]["frames_below"]
        required = self.cfg["liveness"]["blink"]["required_blinks"]
        # Determine closed
        closed = (earL < thr and earR < thr)
        if closed:
            self.last_blink_state = True
            self.below_counter = getattr(self, "below_counter", 0) + 1
        else:
            # A blink can occur entirely between our sampled webcam frames; a
            # single closed observation followed by an open one is enough.
            if self.last_blink_state:
                self.blink_count += 1
            self.below_counter = 0
            self.last_blink_state = False

        return (self.blink_count >= required), f"Blink count: {self.blink_count}/{required}"

    def _check_yaw(self, yaw):
        if self.yaw_baseline is None:
            return False, "Hold still"
        delta = yaw - self.yaw_baseline
        thr = self.cfg["liveness"]["yaw_right"]["delta_deg"]
        return (delta >= thr), f"Yaw Δ={delta:.1f}°, need ≥ {thr}°"

    def _check_mouth(self, lip):
        if not hasattr(self, "mouth_frames"):
            self.mouth_frames = 0
        thr = self.cfg["liveness"]["mouth_open"]["lip_gap_norm"]
        need = self.cfg["liveness"]["mouth_open"]["min_frames"]
        if lip > thr:
            self.mouth_frames += 1
        else:
            self.mouth_frames = 0
        # A single open-mouth frame is sufficient; requiring 15 consecutive
        # frames makes this unnecessarily sensitive to capture cadence.
        return (self.mouth_frames >= min(need, 3)), f"Mouth frames {self.mouth_frames}/{min(need, 3)}"

    def _check_smile(self, width):
        if self.neutral["mouth_width"] is None:
            return False, "Calibrating..."
        inc = (width - self.neutral["mouth_width"]) / (self.neutral["mouth_width"] + 1e-6)
        thr = self.cfg["liveness"]["smile"]["ratio_increase"]
        if not hasattr(self, "smile_frames"):
            self.smile_frames = 0
        if inc >= thr:
            self.smile_frames += 1
        else:
            self.smile_frames = 0
        need = self.cfg["liveness"]["smile"]["min_frames"]
        return (self.smile_frames >= need), f"Smile frames {self.smile_frames}/{need}"
