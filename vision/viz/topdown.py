from collections import deque

import cv2
import numpy as np

from calib.homography import pixel_to_world


class TopDown:
    """Renders tracked positions on the flattened arena floor. Extent is taken
    from the arena outline itself so it works with whatever origin the
    homography was calibrated against."""

    def __init__(self, H: np.ndarray, arena_polygon: np.ndarray,
                 size: int = 600, trail: int = 200):
        self.H = H
        world = cv2.perspectiveTransform(
            arena_polygon.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        lo, hi = world.min(axis=0), world.max(axis=0)
        pad = 0.05 * (hi - lo).max()
        self.lo, self.hi = lo - pad, hi + pad
        self.scale = size / (self.hi - self.lo).max()
        self.size = size
        self.outline = np.array([self._to_canvas(p) for p in world], np.int32)
        self.trails: dict[str, deque] = {}
        self.trail_len = trail

    def _to_canvas(self, world_pt) -> tuple[int, int]:
        x = (world_pt[0] - self.lo[0]) * self.scale
        y = (world_pt[1] - self.lo[1]) * self.scale
        return int(x), int(y)

    def world_of(self, pixel_pt) -> tuple[float, float]:
        return pixel_to_world(self.H, pixel_pt)

    def render(self, positions: dict[str, tuple[float, float]],
               colors: dict[str, tuple[int, int, int]]) -> np.ndarray:
        canvas = np.full((self.size, self.size, 3), 24, np.uint8)
        cv2.polylines(canvas, [self.outline], True, (70, 70, 70), 2)

        for label, world in positions.items():
            trail = self.trails.setdefault(label, deque(maxlen=self.trail_len))
            trail.append(self._to_canvas(world))
            color = colors[label]
            pts = list(trail)
            for k in range(1, len(pts)):
                fade = k / len(pts)
                cv2.line(canvas, pts[k - 1], pts[k],
                         tuple(int(c * fade) for c in color), 2)
            cv2.circle(canvas, pts[-1], 7, color, -1)
            cv2.putText(canvas, f"{label} ({world[0]:.1f}, {world[1]:.1f})m",
                        (pts[-1][0] + 10, pts[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return canvas
