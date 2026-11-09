import argparse, os, sys, cv2, random, time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config
from src.recognition import FaceRecognizer
from src.templates import TemplateStore
from src.liveness import LivenessChecker
from src.attendance import AttendanceLogger
from src.antireplay import is_replay_frame

# Available randomized prompts
CHALLENGES = ["blink2", "yaw_right", "mouth_open", "smile"]


def overlay_text(img, text, y=30, color=(0, 255, 0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    parser = argparse.ArgumentParser(description="Run attendance with FaceNet + randomized liveness.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Load components
    rec = FaceRecognizer(min_face_size=cfg["recognition"]["min_face_size"])
    tm = TemplateStore(cfg["storage"]["template_path"])
    gallery = tm.get_gallery()
    if len(gallery) == 0:
        print("[WARN] No enrolled users found. Run scripts/enroll.py first.")

    logger = AttendanceLogger(cfg["logging"]["dir"], cooldown_seconds=cfg["logging"]["cooldown_seconds"])
    live = LivenessChecker(cfg)

    # Camera
    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    # State
    state = "IDLE"
    current = {"name": None, "pid": None, "score": 0.0, "challenge": None}
    sim_thr = cfg["recognition"]["similarity_threshold"]
    display_secs = cfg.get("ui", {}).get("display_seconds", 2.5)
    display_start = 0.0
    display_msg = ""

    print("[INFO] Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break

        view = frame.copy()

        # -------- IDLE / FAIL / PASSED: detect & try to identify --------
        if state in ["IDLE", "FAIL", "PASSED"]:
            emb, bbox = rec.embed_face(frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if emb is not None and len(gallery) > 0:
                name, score, pid = rec.identify(emb, gallery)
                overlay_text(view, f"Match: {name} ({score:.2f})", 30, (0, 255, 255))

                if score >= sim_thr:
                    # If this person is already marked today, show message and do NOT start challenge
                    if logger.already_marked_today(pid):
                        overlay_text(view, f"{name}: already marked today", 60, (0, 0, 255))
                        state = "DISPLAY_FAIL"
                        display_msg = "Already marked today"
                        display_start = time.time()
                    else:
                        state = "MATCHED"
                        current.update({"name": name, "pid": pid, "score": float(score)})
                else:
                    overlay_text(view, f"Below threshold ({sim_thr})", 60, (0, 200, 200))
            else:
                overlay_text(view, "Detecting face...", 30, (200, 200, 200))

        # -------- MATCHED: pick a random challenge and begin --------
        if state == "MATCHED":
            ch = random.choice(CHALLENGES)
            current["challenge"] = ch
            live.begin(ch)
            state = "CHALLENGE"

        # -------- CHALLENGE: run liveness, check anti-replay, decide --------
        if state == "CHALLENGE":
            ok_ch, msg = live.process(frame)

            # Anti-replay check (full-frame or you can pass bbox if you cache it)
            suspect, why = is_replay_frame(frame, None, cfg)
            if suspect:
                overlay_text(view, f"Replay suspected ({why}). Try again.", 30, (0, 0, 255))
                state = "DISPLAY_FAIL"
                display_msg = "Replay suspected ❌"
                display_start = time.time()
            else:
                overlay_text(
                    view,
                    f"Do: {current['challenge']} | {msg}",
                    30,
                    (0, 255, 0) if ok_ch else (0, 165, 255),
                )

                if ok_ch:
                    # Try to mark; logger enforces cooldown AND one-mark-per-day
                    marked = logger.mark(
                        current["pid"], current["name"], current["score"], current["challenge"], True, "camera0"
                    )
                    if marked:
                        state = "DISPLAY_PASS"
                        display_msg = "Attendance marked ✅"
                    else:
                        # Either cooldown or already-marked-today
                        reason = (
                            "Already marked today" if logger.already_marked_today(current["pid"]) else "Cooldown active"
                        )
                        state = "DISPLAY_FAIL"
                        display_msg = reason
                    display_start = time.time()
                elif msg == "Timeout":
                    state = "DISPLAY_FAIL"
                    display_msg = "Challenge failed ❌ Try again"
                    display_start = time.time()

        # -------- DISPLAY_*: keep message on screen for a few seconds --------
        if state == "DISPLAY_PASS":
            overlay_text(view, display_msg, 60, (0, 255, 0))
            overlay_text(view, f"{current['name']}", 90, (255, 255, 255))
            if (time.time() - display_start) >= display_secs:
                state = "IDLE"
                current = {"name": None, "pid": None, "score": 0.0, "challenge": None}

        if state == "DISPLAY_FAIL":
            overlay_text(view, display_msg, 60, (0, 0, 255))
            if current["name"]:
                overlay_text(view, f"{current['name']}", 90, (255, 255, 255))
            if (time.time() - display_start) >= display_secs:
                state = "IDLE"
                current = {"name": None, "pid": None, "score": 0.0, "challenge": None}

        # Draw window
        cv2.imshow("Attendance", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
