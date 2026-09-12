import json
from pathlib import Path

import cv2
import numpy as np


def compute_homography(pixel_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """pixel_pts, world_pts: Nx2 arrays, N >= 4, corresponding points.
    world_pts are arena-floor coordinates in meters. Returns the 3x3
    homography mapping pixel -> world."""
    if len(pixel_pts) < 4 or len(pixel_pts) != len(world_pts):
        raise ValueError("need >= 4 pixel/world point pairs of equal length")
    H, _ = cv2.findHomography(np.asarray(pixel_pts, dtype=np.float64),
                               np.asarray(world_pts, dtype=np.float64))
    if H is None:
        raise ValueError("homography computation failed -- check point correspondences")
    return H


def pixel_to_world(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    pt = np.array([[point]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world_to_pixel(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    return pixel_to_world(np.linalg.inv(H), point)


def save(H: np.ndarray, path: str) -> None:
    Path(path).write_text(json.dumps(H.tolist()))


def load(path: str) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text()))
