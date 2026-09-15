from __future__ import annotations
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from core.utils import (
    depth_to_proximity,
    median_in_bbox,
    frac_close_in_bbox,
    iou_xyxy,
    apply_border_ignore,
)
from apps.visioncane.config import (
    DEFAULT_YOLO_WEIGHTS,
    PROC_SHORT_SIDE,
    DEVICE,
    FOV_DEG,
    INVERT_DEFAULT,
    YOLO_CONF,
    MAX_DETS,
    BORDER_IGNORE_FRAC,
    BORDER_PENALTY_FRAC,
    BOTTOM_GAMMA,
    BOTTOM_MIN_MULT,
    BOTTOM_MAX_MULT,
    CLOSE_COLOR_THRES,
    GLOBAL_CLOSE_MIN_FRAC,
    BBOX_CLOSE_MIN_FRAC,
    IOU_LOCK_THRES,
    MAX_LOST_FRAMES,
    CLOSE_QUANTILE,
    MIN_AREA_FRAC,
    MORPH_KERNEL,
    MIN_INTERVAL,
    MAX_INTERVAL,
    MIN_VOL,
    MAX_VOL,
    PITCH_FAR,
    PITCH_NEAR,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixel_to_pan(x_px: float, w: int, fov_deg: float) -> float:
    cx = (w - 1) * 0.5
    fov = math.radians(fov_deg)
    f = (w * 0.5) / max(1e-6, math.tan(fov * 0.5))
    angle = math.atan((x_px - cx) / max(1e-6, f))
    return float(np.clip(angle / (fov * 0.5), -1.0, 1.0))


def _map_prox_to_audio(prox: float) -> Tuple[float, float, float]:
    prox = float(np.clip(prox, 0.0, 1.0))
    interval = float(np.clip(MAX_INTERVAL - prox * (MAX_INTERVAL - MIN_INTERVAL), MIN_INTERVAL, MAX_INTERVAL))
    volume = float(np.clip(MIN_VOL + prox * (MAX_VOL - MIN_VOL), MIN_VOL, MAX_VOL))
    pitch = float(np.clip(PITCH_FAR + prox * (PITCH_NEAR - PITCH_FAR), PITCH_FAR, PITCH_NEAR))
    return interval, pitch, volume


def _border_factor(cx: float, cy: float, w: int, h: int) -> float:
    mx, my = BORDER_PENALTY_FRAC * w, BORDER_PENALTY_FRAC * h
    fx = np.clip((min(cx, w - 1 - cx)) / max(1e-6, mx), 0.0, 1.0)
    fy = np.clip((min(cy, h - 1 - cy)) / max(1e-6, my), 0.0, 1.0)
    return float((fx * fy) ** 2)


def _bottom_bias(y2: float, h: int) -> float:
    b = float(np.clip(y2 / max(1.0, h), 0.0, 1.0)) ** BOTTOM_GAMMA
    return float(BOTTOM_MIN_MULT + (BOTTOM_MAX_MULT - BOTTOM_MIN_MULT) * b)


def _global_close_exists(prox: np.ndarray) -> bool:
    prox2 = apply_border_ignore(prox, frac=BORDER_IGNORE_FRAC)
    return float((prox2 >= CLOSE_COLOR_THRES).mean()) >= GLOBAL_CLOSE_MIN_FRAC


def _find_depth_blob_fallback(
    prox: np.ndarray,
) -> Optional[Tuple[int, int, int, int, float]]:
    h, w = prox.shape[:2]
    prox2 = apply_border_ignore(prox, frac=BORDER_IGNORE_FRAC)

    q = float(np.quantile(prox2, CLOSE_QUANTILE))
    mask = ((prox2 >= q).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * (h * w)
    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bb = (x, y, x + bw, y + bh)
        medp = median_in_bbox(prox2, bb)
        score = (medp ** 2.2) * (area ** 0.9) * _bottom_bias(bb[3], h)
        if score > best_score:
            best_score, best = score, (bb[0], bb[1], bb[2], bb[3], medp)
    return best


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class VisionCaneProcessor:
    """
    Fuses YOLO detections with depth proximity to select the most dangerous
    obstacle and return directional audio cue parameters.

    process() is pure computation — no I/O, no audio playback.
    """

    def __init__(
        self,
        yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
        proc_short: int = PROC_SHORT_SIDE,
        invert: bool = INVERT_DEFAULT,
    ):
        self.yolo = YOLO(yolo_weights)
        self.proc_short = proc_short
        self.invert = invert
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self._lost = 0

    def process(self, proc_bgr: np.ndarray, depth: np.ndarray) -> dict:
        """
        Args:
            proc_bgr: BGR frame at proc_short resolution
            depth:    float32 H×W depth array (same size as proc_bgr)
        Returns:
            {present, pan, interval, volume, pitch}
        """
        H, W = proc_bgr.shape[:2]
        prox = depth_to_proximity(depth, invert=self.invert)
        prox2 = apply_border_ignore(prox, frac=BORDER_IGNORE_FRAC)
        close_ok = _global_close_exists(prox)

        det = self.yolo.predict(
            proc_bgr, conf=YOLO_CONF, verbose=False, device=DEVICE, max_det=MAX_DETS
        )[0]
        boxes = det.boxes
        chosen = None

        if boxes is not None and len(boxes) > 0 and close_ok:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # Try to lock onto previously tracked box
            if self._prev_bbox is not None:
                best_iou, best_bb = 0.0, None
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    bb = (int(x1), int(y1), int(x2), int(y2))
                    iou = iou_xyxy(self._prev_bbox, bb)
                    if iou > best_iou:
                        best_iou, best_bb = iou, bb
                if best_bb is not None and best_iou >= IOU_LOCK_THRES:
                    if frac_close_in_bbox(prox2, best_bb, CLOSE_COLOR_THRES) >= BBOX_CLOSE_MIN_FRAC:
                        chosen = (*best_bb, median_in_bbox(prox2, best_bb))

            # No lock — pick highest-scoring detection
            if chosen is None:
                best_score, best = -1.0, None
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    bb = (int(x1), int(y1), int(x2), int(y2))
                    if bb[2] <= bb[0] or bb[3] <= bb[1]:
                        continue
                    if frac_close_in_bbox(prox2, bb, CLOSE_COLOR_THRES) < BBOX_CLOSE_MIN_FRAC:
                        continue
                    medp = median_in_bbox(prox2, bb)
                    area_frac = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1])) / (W * H)
                    cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
                    score = (
                        (medp ** 2.6)
                        * ((area_frac + 1e-6) ** 0.6)
                        * ((float(c) + 1e-6) ** 0.6)
                        * _border_factor(cx, cy, W, H)
                        * _bottom_bias(bb[3], H)
                    )
                    if score > best_score:
                        best_score, best = score, (*bb, medp)
                chosen = best

        # Fallback: pure depth blob if YOLO found nothing
        if chosen is None and close_ok:
            fb = _find_depth_blob_fallback(prox)
            if fb is not None:
                chosen = fb

        if chosen is None:
            self._lost += 1
            if self._lost > MAX_LOST_FRAMES:
                self._prev_bbox = None
            return {"present": False, "pan": 0.0, "interval": MAX_INTERVAL, "volume": 0.0, "pitch": PITCH_FAR}

        self._lost = 0
        x1, y1, x2, y2, medp = chosen
        self._prev_bbox = (int(x1), int(y1), int(x2), int(y2))

        cx = 0.5 * (x1 + x2)
        pan = _pixel_to_pan(cx, W, FOV_DEG)
        interval, pitch, volume = _map_prox_to_audio(medp)

        return {"present": True, "pan": pan, "interval": interval, "volume": volume, "pitch": pitch}
