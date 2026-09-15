from typing import Tuple
import cv2
import numpy as np


def to_u8_normalized(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return ((x - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)


def resize_keep_aspect(
    frame_bgr: np.ndarray, short_side: int
) -> Tuple[np.ndarray, float, float]:
    """Resize frame so its shorter dimension equals short_side.
    Returns (resized, sx, sy) where sx/sy are the scale factors."""
    h, w = frame_bgr.shape[:2]
    if h < w:
        new_h, new_w = short_side, int(round(w * (short_side / h)))
    else:
        new_w, new_h = short_side, int(round(h * (short_side / w)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w / w, new_h / h


def map_bbox_proc_to_orig(
    bbox: Tuple[int, int, int, int], sx: float, sy: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ox1 = int(np.clip(round(x1 / sx), 0, W - 1))
    oy1 = int(np.clip(round(y1 / sy), 0, H - 1))
    ox2 = int(np.clip(round(x2 / sx), 0, W - 1))
    oy2 = int(np.clip(round(y2 / sy), 0, H - 1))
    return ox1, oy1, ox2, oy2


def iou_xyxy(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(1e-6, area_a + area_b - inter))


def depth_to_proximity(depth: np.ndarray, invert: bool = True) -> np.ndarray:
    lo = float(np.percentile(depth, 5))
    hi = float(np.percentile(depth, 95))
    dn = np.clip((depth - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    prox = (1.0 - dn) if invert else dn
    return prox.astype(np.float32)


def median_in_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = img[y1:y2, x1:x2]
    return float(np.median(patch)) if patch.size else 0.0


def frac_close_in_bbox(
    prox: np.ndarray, bbox: Tuple[int, int, int, int], threshold: float
) -> float:
    x1, y1, x2, y2 = bbox
    h, w = prox.shape[:2]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = prox[y1:y2, x1:x2]
    return float((patch >= threshold).mean()) if patch.size else 0.0


def apply_border_ignore(prox: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Zero out a border band of the proximity map to reduce false positives at edges."""
    h, w = prox.shape[:2]
    b = int(round(frac * min(h, w)))
    prox2 = prox.copy()
    prox2[:b, :] = 0
    prox2[-b:, :] = 0
    prox2[:, :b] = 0
    prox2[:, -b:] = 0
    return prox2
