
import argparse, os, sys, cv2, numpy as np
from statistics import mean

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config
from src.recognition import FaceRecognizer
from src.templates import TemplateStore

def main():
    parser = argparse.ArgumentParser(description="Enroll a new user (capture N shots, average embeddings).")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", required=True, help="Person name to enroll")
    parser.add_argument("--shots", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    rec = FaceRecognizer(min_face_size=cfg["recognition"]["min_face_size"])
    tm = TemplateStore(cfg["storage"]["template_path"])

    print(f"[INFO] Enrolling '{args.name}' — press SPACE to capture, ESC to cancel.")
    collected = []
    while len(collected) < args.shots:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        emb, bbox = rec.embed_face(frame)
        view = frame.copy()
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(view, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(view, f"{args.name}: {len(collected)}/{args.shots} shots", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
        cv2.imshow("Enroll", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Cancelled.")
            cap.release(); cv2.destroyAllWindows()
            return
        if key == 32:  # SPACE
            if emb is not None:
                collected.append(emb)
                print(f"Captured {len(collected)}/{args.shots}")
            else:
                print("No face detected; try again.")

    cap.release(); cv2.destroyAllWindows()
    if len(collected) < args.shots:
        print("Not enough shots; exiting.")
        return

    avg = np.mean(np.stack(collected, axis=0), axis=0)
    avg = (avg / (np.linalg.norm(avg) + 1e-12)).astype(np.float32)
    person = tm.add_person(args.name, avg.tolist())
    print(f"[OK] Enrolled: {person['name']} (id={person['person_id']})")

if __name__ == "__main__":
    main()
