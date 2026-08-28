from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Returns (ok, frame_bgr, timestamp_seconds)."""

    @abstractmethod
    def close(self) -> None:
        ...

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame, ts = self.read()
        if not ok:
            raise StopIteration
        return frame, ts


class VideoFileSource(FrameSource):
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"could not open video file: {path}")

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return True, frame, ts

    def close(self) -> None:
        self.cap.release()


class LiveCameraSource(FrameSource):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "live camera SDK/protocol not yet determined -- see context.md"
        )

    def read(self):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
