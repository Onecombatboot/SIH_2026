from __future__ import annotations
import os
import random
import time
import urllib.request
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

from apps.neuromap.config import (
    PROC_SHORT_SIDE,
    CUE_DURATION,
    RESPONSE_WINDOW,
    FEEDBACK_DURATION,
    UNLOCK_ACCURACY,
    UNLOCK_ROUNDS,
    ZONE_CONFIGS,
)

# ---------------------------------------------------------------------------
# Face mesh connection indices for drawing overlays
# ---------------------------------------------------------------------------
_FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
    400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10
]
_LEFT_EYE  = [33,7,163,144,145,153,154,155,133,33]
_RIGHT_EYE = [362,382,381,380,374,373,390,249,263,362]
_NOSE      = [168,6,197,195,5,4,1]
_LIPS      = [61,146,91,181,84,17,314,405,321,375,291,61]

_MODEL_PATH = os.path.join("assets", "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Game states
CUE      = "cue"
RESPONSE = "response"
FEEDBACK = "feedback"


class NeuroMapProcessor:
    """
    Gamified spatial awareness trainer for TBI rehabilitation.

    Uses MediaPipe FaceLandmarker (Tasks API) for accurate head pose estimation.
    Draws the full face mesh + a direction arrow on each frame so the
    therapist/patient can see what the system is detecting.

    process() is called every frame.  Returns:
        {
          "annotated_bgr": np.ndarray,
          "cue":           dict,
          "game":          dict,
        }
    """

    def __init__(self, proc_short: int = PROC_SHORT_SIDE):
        self.proc_short = proc_short

        # Download model on first run
        if not os.path.exists(_MODEL_PATH):
            print(f"[NeuroMap] Downloading face landmarker model to {_MODEL_PATH} ...")
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("[NeuroMap] Download complete.")

        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=_mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)

        # Game state
        self._state        = CUE
        self._level        = 1
        self._target_zone  = 0
        self._state_ts     = time.time()
        self._score        = 0
        self._total        = 0
        self._streak       = 0
        self._correct: Optional[bool] = None
        self._patient_zone: Optional[int] = None
        self._recent: list[bool] = []

        self._start_next_round()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cfg(self) -> dict:
        return ZONE_CONFIGS[self._level]

    def _start_next_round(self):
        cfg = self._cfg()
        self._target_zone  = random.randint(0, cfg["n"] - 1)
        self._state        = CUE
        self._state_ts     = time.time()
        self._correct      = None
        self._patient_zone = None

    def _try_level_up(self):
        if self._level >= 3 or len(self._recent) < UNLOCK_ROUNDS:
            return
        if sum(self._recent[-UNLOCK_ROUNDS:]) / UNLOCK_ROUNDS >= UNLOCK_ACCURACY:
            self._level  += 1
            self._recent  = []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, frame_bgr: np.ndarray):
        """Run face landmarker. Returns (yaw, pitch, landmarks) or (None, None, None)."""
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None, None, None

        landmarks = result.face_landmarks[0]

        # Head pose from transformation matrix (preferred)
        if result.facial_transformation_matrixes:
            mat    = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
            R      = mat[:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
            yaw    = float(angles[1])
            pitch  = float(angles[0])
        else:
            # Fallback: estimate yaw from eye-to-nose lateral offset
            nose   = landmarks[4]
            l_eye  = landmarks[33]
            r_eye  = landmarks[263]
            eye_cx = (l_eye.x + r_eye.x) / 2.0
            eye_w  = abs(r_eye.x - l_eye.x) + 1e-6
            yaw    = float((nose.x - eye_cx) / eye_w * 90.0)
            pitch  = float((0.45 - nose.y) * 80.0)

        return yaw, pitch, landmarks

    def _yaw_pitch_to_zone(self, yaw: float, pitch: float) -> int:
        cfg = self._cfg()
        if self._level <= 2:
            zone = 0
            for thresh in cfg["yaw_bins"]:
                if yaw >= thresh:
                    zone += 1
            return zone
        else:
            col = sum(1 for t in cfg["yaw_bins"]   if yaw   >= t)
            row = sum(1 for t in cfg["pitch_bins"] if pitch <= t)
            return col * 3 + row

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_poly(self, frame, landmarks, indices, color, thickness=1):
        h, w = frame.shape[:2]
        pts = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, thickness)

    def _annotate(
        self,
        frame: np.ndarray,
        yaw: Optional[float],
        pitch: Optional[float],
        landmarks,
    ) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]
        cfg  = self._cfg()
        n    = cfg["n"]

        # ---- face mesh overlay ----
        if landmarks is not None:
            mesh_color = (0, 210, 160)
            self._draw_poly(out, landmarks, _FACE_OVAL, mesh_color, 1)
            self._draw_poly(out, landmarks, _LEFT_EYE,  mesh_color, 1)
            self._draw_poly(out, landmarks, _RIGHT_EYE, mesh_color, 1)
            self._draw_poly(out, landmarks, _NOSE,      mesh_color, 1)
            self._draw_poly(out, landmarks, _LIPS,      mesh_color, 1)

            # Direction arrow from nose tip
            if yaw is not None and pitch is not None:
                nose = landmarks[4]
                nx, ny   = int(nose.x * w), int(nose.y * h)
                arrow_len = 70
                dx = int(np.sin(np.radians(yaw))   *  arrow_len)
                dy = int(np.sin(np.radians(pitch))  * -arrow_len)
                cv2.arrowedLine(out, (nx, ny), (nx + dx, ny + dy), (50, 150, 255), 3, tipLength=0.35)

        # ---- zone grid ----
        if self._level <= 2:
            YAW_MAX = 45.0
            bounds  = [0] + [
                int(np.clip((t / YAW_MAX + 1.0) * 0.5 * w, 0, w))
                for t in cfg["yaw_bins"]
            ] + [w]

            for i in range(n):
                x1, x2   = bounds[i], bounds[i + 1]
                is_target = (i == self._target_zone) and (self._state == RESPONSE)
                is_here   = (i == self._patient_zone)

                overlay = out.copy()
                cv2.rectangle(overlay, (x1, 0), (x2, h),
                              (0, 200, 0) if is_target else (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.28 if is_target else 0.08, out, 1 - (0.28 if is_target else 0.08), 0, out)

                if i > 0:
                    cv2.line(out, (x1, 0), (x1, h), (160, 160, 160), 1)

                label_color = (0, 255, 255) if is_here else (220, 220, 220)
                cv2.putText(out, cfg["labels"][i], (x1 + 6, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2)

        else:   # 3×3
            cw, ch = w // 3, h // 3
            for col in range(3):
                for row in range(3):
                    zi  = col * 3 + row
                    x1, y1 = col * cw, row * ch
                    x2, y2 = x1 + cw, y1 + ch
                    is_target = (zi == self._target_zone) and (self._state == RESPONSE)
                    is_here   = (zi == self._patient_zone)
                    overlay   = out.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  (0, 200, 0) if is_target else (50, 50, 50), -1)
                    cv2.addWeighted(overlay, 0.28 if is_target else 0.08, out, 1 - (0.28 if is_target else 0.08), 0, out)
                    cv2.putText(out, cfg["labels"][zi], (x1 + 4, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 255) if is_here else (220, 220, 220), 2)

            for i in range(1, 3):
                cv2.line(out, (i * cw, 0), (i * cw, h), (160, 160, 160), 1)
                cv2.line(out, (0, i * ch), (w, i * ch), (160, 160, 160), 1)

        # ---- CUE prompt ----
        if self._state == CUE:
            cv2.putText(out, f"LISTEN  \u2192  {cfg['labels'][self._target_zone]}",
                        (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 220, 0), 3)

        # ---- RESPONSE timer bar ----
        if self._state == RESPONSE:
            elapsed  = time.time() - self._state_ts
            fraction = max(0.0, 1.0 - elapsed / RESPONSE_WINDOW)
            cv2.rectangle(out, (0, h - 8), (int(w * fraction), h),
                          (0, 200, 0) if fraction > 0.4 else (0, 80, 255), -1)

        # ---- FEEDBACK ----
        if self._state == FEEDBACK and self._correct is not None:
            text  = "CORRECT!" if self._correct else "WRONG"
            color = (0, 230, 0)  if self._correct else (0, 0, 230)
            cv2.putText(out, text, (w // 2 - 90, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)

        # ---- head pose readout ----
        if yaw is not None:
            zone_label = cfg["labels"][self._patient_zone] if self._patient_zone is not None else "?"
            pose_text  = f"Yaw {yaw:+.1f}\u00b0  Pitch {pitch:+.1f}\u00b0  \u2192  {zone_label}"
            cv2.putText(out, pose_text, (8, h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        else:
            cv2.putText(out, "No face detected", (8, h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2)

        # ---- HUD ----
        hud = f"Score {self._score}/{self._total}   Streak {self._streak}   Level {self._level}"
        cv2.putText(out, hud, (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, proc_bgr: np.ndarray, depth=None) -> dict:
        now     = time.time()
        elapsed = now - self._state_ts

        # Mirror the frame so the patient sees a selfie-style reflection.
        # Detection runs on the flipped frame so zone mapping is naturally mirrored.
        proc_bgr = cv2.flip(proc_bgr, 1)

        yaw, pitch, landmarks = self._detect(proc_bgr)

        # Horizontal flip inverts pitch sign due to coordinate handedness change.
        if pitch is not None:
            pitch = -pitch

        if yaw is not None:
            self._patient_zone = self._yaw_pitch_to_zone(yaw, pitch or 0.0)

        cfg = self._cfg()
        cue = {"present": False, "pan": 0.0, "interval": 1.0, "volume": 0.0, "pitch": 440.0}

        if self._state == CUE:
            z   = self._target_zone
            cue = {"present": True, "pan": cfg["pans"][z], "interval": 0.15,
                   "volume": 0.85, "pitch": cfg["pitches"][z]}
            if elapsed >= CUE_DURATION:
                self._state    = RESPONSE
                self._state_ts = now

        elif self._state == RESPONSE:
            z   = self._target_zone
            cue = {"present": True, "pan": cfg["pans"][z], "interval": 0.35,
                   "volume": 0.40, "pitch": cfg["pitches"][z]}
            if self._patient_zone is not None and self._patient_zone == self._target_zone:
                self._correct  = True
                self._score   += 1
                self._streak  += 1
                self._total   += 1
                self._recent.append(True)
                self._try_level_up()
                self._state    = FEEDBACK
                self._state_ts = now
            elif elapsed >= RESPONSE_WINDOW:
                self._correct  = False
                self._streak   = 0
                self._total   += 1
                self._recent.append(False)
                self._state    = FEEDBACK
                self._state_ts = now

        elif self._state == FEEDBACK:
            if elapsed >= FEEDBACK_DURATION:
                self._start_next_round()

        annotated = self._annotate(proc_bgr, yaw, pitch, landmarks)
        accuracy  = (self._score / self._total) if self._total else 0.0

        game = {
            "state":              self._state,
            "level":              self._level,
            "n_zones":            cfg["n"],
            "target_zone":        self._target_zone,
            "target_label":       cfg["labels"][self._target_zone],
            "patient_zone":       self._patient_zone,
            "score":              self._score,
            "total":              self._total,
            "streak":             self._streak,
            "accuracy":           round(accuracy, 2),
            "correct":            self._correct,
            "response_remaining": round(max(0.0, RESPONSE_WINDOW - elapsed), 2)
                                   if self._state == RESPONSE else 0.0,
            "current_yaw":        round(yaw,   1) if yaw   is not None else None,
            "current_pitch":      round(pitch, 1) if pitch is not None else None,
        }

        return {"annotated_bgr": annotated, "cue": cue, "game": game}
