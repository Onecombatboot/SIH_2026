import os
import cv2


class VideoFile:
    """Drop-in replacement for Camera that reads from a video file.
    Loops by default so the stream never runs dry."""

    def __init__(self, path: str, loop: bool = True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop = loop

    def read(self):
        """Returns the next BGR frame, or None on unrecoverable failure."""
        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
