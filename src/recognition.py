import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognizer:
    def __init__(self, device:str=None, min_face_size:int=80):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=False, device=self.device, image_size=160, margin=0, min_face_size=min_face_size)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def _to_rgb(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def embed_face(self, frame_bgr):
        """Detects, aligns, and embeds the most prominent face. Returns (embedding_512, bbox) or (None, None)."""
        img_rgb = self._to_rgb(frame_bgr)
        face = self.mtcnn(img_rgb)  # returns 160x160 tensor (C,H,W) normalized internally
        if face is None:
            return None, None
        with torch.no_grad():
            emb = self.model(face.unsqueeze(0).to(self.device)).cpu().numpy()[0]  # (512,)
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-12
        emb = (emb / norm).astype(np.float32)
        # best bbox via detect
        boxes, probs = self.mtcnn.detect(img_rgb)
        bbox = None
        if boxes is not None and len(boxes) > 0:
            i = int(np.argmax(probs))
            bbox = boxes[i].astype(int)  # x1,y1,x2,y2
        return emb, bbox

    @staticmethod
    def cosine(u:np.ndarray, v:np.ndarray) -> float:
        return float(np.dot(u, v))

    def identify(self, emb:np.ndarray, gallery:list):
        """gallery: list of (name, emb_list, person_id). Returns (best_name, best_score, person_id)."""
        if emb is None or gallery is None or len(gallery) == 0:
            return None, 0.0, None
        sims = []
        for name, gemb_list, pid in gallery:
            gemb = np.array(gemb_list, dtype=np.float32)
            sims.append((name, self.cosine(emb, gemb), pid))
        best = max(sims, key=lambda x: x[1])
        return best  # (name, score, pid)
