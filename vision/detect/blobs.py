from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]
    area: int
    intensity: float


def arena_mask(gray: np.ndarray) -> np.ndarray:
    """Largest bright region of a lit frame is the arena floor; everything
    outside it (crowd, lights, walls) generates motion we never want."""
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, np.uint8)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def mask_polygon(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)


class BlobDetector:
    def __init__(self, mask: np.ndarray, min_area: int = 400, max_area: int = 25000):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=False)
        self.mask = mask
        self.min_area = min_area
        self.max_area = max_area
        self.k_open = np.ones((3, 3), np.uint8)
        self.k_close = np.ones((9, 9), np.uint8)

    def apply(self, gray: np.ndarray, learning_rate: float = -1.0):
        fg = self.bg.apply(gray, learningRate=learning_rate)
        fg = cv2.bitwise_and(fg, self.mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)

        n, _, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
        dets = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not self.min_area <= area <= self.max_area:
                continue
            x, y, w, h = (int(v) for v in stats[i, :4])
            dets.append(Detection(float(cent[i][0]), float(cent[i][1]),
                                  (x, y, w, h), area,
                                  float(gray[y:y + h, x:x + w].mean())))
        return dets, fg
