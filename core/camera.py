import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = None, height: int = None):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """Returns a BGR frame or None on failure."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
