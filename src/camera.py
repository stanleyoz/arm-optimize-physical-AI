import cv2
import threading
from collections import deque
from queue import Queue, Empty


class Camera:
    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self._cap = None
        self._running = False
        self._thread = None
        self._frame_queue: Queue = Queue(maxsize=2)

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._open()
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            self._frame_queue.put(frame)

    def start(self):
        if self._running:
            return
        self._running = True
        self._open()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read(self) -> cv2.typing.MatLike | None:
        try:
            return self._frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()
