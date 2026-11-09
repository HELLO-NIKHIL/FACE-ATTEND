import cv2, numpy as np
from math import degrees, atan2

def _count_long_axis_aligned_lines(gray, min_len=120, angle_tol=10):
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=min_len, maxLineGap=8)
    if lines is None: return 0
    cnt = 0
    for x1,y1,x2,y2 in lines[:,0]:
        ang = abs(degrees(atan2(y2-y1, x2-x1)))
        # near 0° (horizontal) or 90° (vertical)
        if (ang <= angle_tol) or (abs(ang-90) <= angle_tol):
            cnt += 1
    return cnt

def _has_moire(gray, peak_ratio=5.0):
    # FFT magnitude peaks away from center
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    # mask out low-freq center
    cx, cy = w//2, h//2
    mask = np.ones_like(mag, dtype=bool)
    cv2.circle(mask, (cx, cy), int(min(h,w)*0.12), 0, -1)
    ring = mag[mask]
    if ring.size < 100: return False
    top = np.percentile(ring, 99.5)
    med = np.median(ring)
    return (top / (med + 1e-6)) >= peak_ratio

def is_replay_frame(frame_bgr, bbox, cfg) -> (bool, str):
    """
    Cheap heuristics:
      1) Many long axis-aligned lines (likely screen edges/bezels)
      2) Optional moiré peaks in FFT
    Returns (is_replay, reason)
    """
    if not cfg.get("anti_replay", {}).get("enabled", True):
        return False, "disabled"

    x1,y1,x2,y2 = bbox if bbox is not None else (0,0,frame_bgr.shape[1], frame_bgr.shape[0])
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame_bgr.shape[1],x2), min(frame_bgr.shape[0],y2)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = frame_bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    se = cfg["anti_replay"]["screen_edge"]
    long_lines = _count_long_axis_aligned_lines(gray, se["min_length"], se["angle_tol_deg"])
    if long_lines >= se["min_long_lines"]:
        return True, f"screen-edges:{long_lines}"

    if cfg["anti_replay"]["moire"]["enabled"]:
        if _has_moire(gray, cfg["anti_replay"]["moire"]["peak_ratio"]):
            return True, "moire-peaks"

    return False, "clear"
