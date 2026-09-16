from __future__ import annotations
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from core.utils import (
    median_in_bbox,
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
    PCTL_LOW,
    PCTL_HIGH,
    REF_EMA_ALPHA,
    MIN_BG_FRAC,
    PROX_CLIP,
    AREA_NEAR_MIN,
    AREA_NEAR_MAX,
    URGENCY_RISE,
    URGENCY_FALL,
    COAST_DECAY,
    COAST_MIN,
    BORDER_IGNORE_FRAC,
    BORDER_PENALTY_FRAC,
    BOTTOM_GAMMA,
    BOTTOM_MIN_MULT,
    BOTTOM_MAX_MULT,
    METRIC_DEPTH,
    RELEASE_DISTANCE_M,
    CRITICAL_DISTANCE_M,
    GROUND_FIT_TOP,
    GROUND_ROW_PCTL,
    GROUND_TOLERANCE,
    GROUND_MIN_DROP_M,
    GROUND_MAX_RESIDUAL,
    MIN_TARGET_PROX_METRIC,
    RETAIN_TARGET_PROX_METRIC,
    FALLBACK_MIN_PROX_METRIC,
    PROX_FAR_ANCHOR,
    PROX_NEAR_ANCHOR,
    MIN_TARGET_PROX_RELATIVE,
    RETAIN_TARGET_PROX_RELATIVE,
    FALLBACK_MIN_PROX_RELATIVE,
    SIZE_FUSION_MIN,
    FLOOR_IGNORE_FRAC,
    IOU_LOCK_THRES,
    MAX_LOST_FRAMES,
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


def _ground_mask(metres: np.ndarray) -> np.ndarray:
    """True where a pixel's distance matches the floor receding at that image row.

    The floor is always among the nearest surfaces in view, so without suppressing it
    every frame reports an obstacle underfoot. Rather than assume a camera height and
    tilt, the floor is measured from the frame itself: across the lower rows its
    distance falls steadily toward the bottom of the image, so a line is fitted to the
    per-row distance and pixels lying on that line are treated as floor.

    Two guards keep it from erasing real obstacles. Each row is summarised by an upper
    percentile, so anything nearer than the floor in that row (i.e. standing on it) sits
    below the fit and survives. And the fit is used only when it actually looks like a
    receding floor — a wall, or an obstacle looming across the lower frame, gives a flat
    slope, which disables suppression and lets the alarm through.
    """
    h, w = metres.shape[:2]
    y0 = int(GROUND_FIT_TOP * h)
    rows = np.arange(y0, h, dtype=np.float32)
    if rows.size < 8:
        return np.zeros(metres.shape, dtype=bool)

    per_row = np.percentile(metres[y0:, :], GROUND_ROW_PCTL, axis=1).astype(np.float32)

    slope, intercept = np.polyfit(rows, per_row, 1)
    expected = slope * rows + intercept
    spread = float(np.percentile(per_row, 90) - np.percentile(per_row, 10))
    residual = float(np.median(np.abs(per_row - expected)))

    # A floor recedes: distance must fall by a real margin toward the bottom of frame,
    # and the rows must actually follow the line rather than scatter around it.
    drop = float(expected[0] - expected[-1])
    if drop < GROUND_MIN_DROP_M or residual > GROUND_MAX_RESIDUAL * max(spread, 1e-6):
        return np.zeros(metres.shape, dtype=bool)

    mask = np.zeros(metres.shape, dtype=bool)
    band = metres[y0:, :]
    exp2d = expected[:, None]
    mask[y0:, :] = np.abs(band - exp2d) <= GROUND_TOLERANCE * np.maximum(exp2d, 1e-6)
    return mask


def _find_depth_blob_fallback(
    prox: np.ndarray,
    min_prox: float,
) -> Optional[Tuple[int, int, int, int, float]]:
    """Nearest protruding depth blob, used when the detector recognises nothing.

    Walls, poles and unlabelled clutter still need to raise an alarm. The ground
    directly in front of the camera is excluded: it is always among the nearest
    surfaces in view, so without that cut every frame would report an obstacle.
    """
    h, w = prox.shape[:2]
    prox2 = apply_border_ignore(prox, frac=BORDER_IGNORE_FRAC)
    prox2[int((1.0 - FLOOR_IGNORE_FRAC) * h):, :] = 0.0

    mask = ((prox2 >= min_prox).astype(np.uint8) * 255)
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
        metric: bool = METRIC_DEPTH,
    ):
        self.yolo = YOLO(yolo_weights)
        self.proc_short = proc_short
        self.invert = invert
        self.metric = metric
        self._min_target = MIN_TARGET_PROX_METRIC if metric else MIN_TARGET_PROX_RELATIVE
        self._retain_target = RETAIN_TARGET_PROX_METRIC if metric else RETAIN_TARGET_PROX_RELATIVE
        self._fallback_min = FALLBACK_MIN_PROX_METRIC if metric else FALLBACK_MIN_PROX_RELATIVE
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self._lost = 0
        # Reference depth scale (EMA over background pixels) and smoothed output.
        self._ref_lo: Optional[float] = None
        self._ref_hi: Optional[float] = None
        self._urgency = 0.0
        self._last_pan = 0.0

    # ------------------------------------------------------------------
    # Proximity
    # ------------------------------------------------------------------

    def _proximity_map(self, depth: np.ndarray) -> np.ndarray:
        """Depth -> proximity in 0-1, where 1.0 is maximum danger.

        In metric mode the depth is in metres, so danger simply ramps from
        RELEASE_DISTANCE_M down to CRITICAL_DISTANCE_M, and anything beyond the release
        distance is exactly zero — the cane stays silent with nothing near. The ground
        plane is suppressed separately, or the floor itself would alarm every frame.

        In relative mode there are no absolute distances, so proximity is measured
        against the scene itself: percentiles over the background (everything outside
        the locked target), smoothed across frames. Without excluding the target, an
        obstacle that grows to fill the frame ends up defining the very scale used to
        judge it, and its proximity collapses just as it becomes most dangerous.
        """
        if self.metric:
            span = max(1e-6, RELEASE_DISTANCE_M - CRITICAL_DISTANCE_M)
            metres = depth.astype(np.float32)
            prox = np.clip((RELEASE_DISTANCE_M - metres) / span, 0.0, 1.0)
            prox[_ground_mask(metres)] = 0.0
            return prox.astype(np.float32)

        d = -depth.astype(np.float32) if self.invert else depth.astype(np.float32)

        sample = d
        if self._prev_bbox is not None:
            x1, y1, x2, y2 = self._prev_bbox
            keep = np.ones(d.shape, dtype=bool)
            keep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = False
            if keep.mean() >= MIN_BG_FRAC:
                sample = d[keep]

        lo = float(np.percentile(sample, PCTL_LOW))
        hi = float(np.percentile(sample, PCTL_HIGH))
        if hi - lo < 1e-6:
            hi = lo + 1e-6

        if self._ref_lo is None:
            self._ref_lo, self._ref_hi = lo, hi
        else:
            self._ref_lo += REF_EMA_ALPHA * (lo - self._ref_lo)
            self._ref_hi += REF_EMA_ALPHA * (hi - self._ref_hi)

        prox = (d - self._ref_lo) / max(1e-6, self._ref_hi - self._ref_lo)
        return np.clip(prox, 0.0, PROX_CLIP).astype(np.float32)

    def _danger(self, medp: float) -> float:
        """Median proximity -> 0-1 danger term."""
        if self.metric:
            return float(np.clip(medp, 0.0, 1.0))
        span = max(1e-6, PROX_NEAR_ANCHOR - PROX_FAR_ANCHOR)
        return float(np.clip((medp - PROX_FAR_ANCHOR) / span, 0.0, 1.0))

    def _update_urgency(self, medp: float, bbox: Tuple[int, int, int, int], W: int, H: int) -> float:
        """Fuse depth proximity with apparent size, then smooth asymmetrically.

        Once an obstacle is nearer than anything in the background its proximity
        saturates, but its apparent size keeps growing — so size carries the cue the
        rest of the way in. Urgency rises quickly and decays slowly, so one bad frame
        cannot report a looming obstacle as distant.
        """
        x1, y1, x2, y2 = bbox
        depth_term = self._danger(medp)
        target = depth_term
        if not self.metric and depth_term >= SIZE_FUSION_MIN:
            area_frac = max(0, x2 - x1) * max(0, y2 - y1) / float(max(1, W * H))
            size_term = (area_frac - AREA_NEAR_MIN) / max(1e-6, AREA_NEAR_MAX - AREA_NEAR_MIN)
            target = max(depth_term, float(np.clip(size_term, 0.0, 1.0)))

        alpha = URGENCY_RISE if target > self._urgency else URGENCY_FALL
        self._urgency += alpha * (target - self._urgency)
        return self._urgency

    def _coast(self) -> dict:
        """Fade the previous cue across a short detection gap rather than going silent."""
        self._urgency *= COAST_DECAY
        if self._urgency < COAST_MIN:
            self._urgency = 0.0
            return {"present": False, "pan": 0.0, "interval": MAX_INTERVAL,
                    "volume": 0.0, "pitch": PITCH_FAR}
        interval, pitch, volume = _map_prox_to_audio(self._urgency)
        return {"present": True, "pan": self._last_pan, "interval": interval,
                "volume": volume, "pitch": pitch}

    def process(self, proc_bgr: np.ndarray, depth: np.ndarray) -> dict:
        """
        Args:
            proc_bgr: BGR frame at proc_short resolution
            depth:    float32 H×W depth array (same size as proc_bgr)
        Returns:
            {present, pan, interval, volume, pitch}
        """
        H, W = proc_bgr.shape[:2]
        # `prox` measures the target itself; `prox2` (edges zeroed) is only for
        # scene-wide gating and the blob fallback. Measuring a target on prox2 would
        # penalise exactly the close obstacles that spill past the frame edges.
        prox = self._proximity_map(depth)

        det = self.yolo.predict(
            proc_bgr, conf=YOLO_CONF, verbose=False, device=DEVICE, max_det=MAX_DETS
        )[0]
        boxes = det.boxes
        chosen = None

        if boxes is not None and len(boxes) > 0:
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
                    medp = median_in_bbox(prox, best_bb)
                    # Holding a lock is deliberately easier than acquiring one.
                    if medp >= self._retain_target:
                        chosen = (*best_bb, medp)

            # No lock — pick highest-scoring detection
            if chosen is None:
                best_score, best = -1.0, None
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    bb = (int(x1), int(y1), int(x2), int(y2))
                    if bb[2] <= bb[0] or bb[3] <= bb[1]:
                        continue
                    medp = median_in_bbox(prox, bb)
                    if medp < self._min_target:
                        continue
                    area_frac = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1])) / (W * H)
                    cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
                    score = (
                        (self._danger(medp) ** 2.6)
                        * ((area_frac + 1e-6) ** 0.6)
                        * ((float(c) + 1e-6) ** 0.6)
                        * _border_factor(cx, cy, W, H)
                        * _bottom_bias(bb[3], H)
                    )
                    if score > best_score:
                        best_score, best = score, (*bb, medp)
                chosen = best

        # Fallback: pure depth blob if YOLO found nothing
        if chosen is None:
            fb = _find_depth_blob_fallback(prox, self._fallback_min)
            if fb is not None:
                chosen = fb

        if chosen is None:
            self._lost += 1
            if self._lost > MAX_LOST_FRAMES:
                self._prev_bbox = None
            return self._coast()

        self._lost = 0
        x1, y1, x2, y2, medp = chosen
        bbox = (int(x1), int(y1), int(x2), int(y2))
        self._prev_bbox = bbox

        cx = 0.5 * (x1 + x2)
        pan = _pixel_to_pan(cx, W, FOV_DEG)
        self._last_pan = pan
        urgency = self._update_urgency(medp, bbox, W, H)
        interval, pitch, volume = _map_prox_to_audio(urgency)

        return {"present": True, "pan": pan, "interval": interval, "volume": volume, "pitch": pitch}
