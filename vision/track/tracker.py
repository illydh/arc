import cv2
import numpy as np

from detect.blobs import Detection

TEMPLATE_SCORE = 0.7
SEARCH_MARGIN = 60
# a template match is only trusted this far from the last real detection --
# without the leash it random-walks across the floor over a few seconds
ANCHOR_RADIUS = 120.0


class Track:
    def __init__(self, label: str, x: float, y: float, half: int = 45):
        self.label = label
        self.stale = 0
        self.source = "init"
        self.intensity = None
        self.area = None
        self.template = None
        self.anchor = (x, y)
        self.half = half
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.pred = (x, y)

    def predict(self, dt: float) -> tuple[float, float]:
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.diag(
            [1.0, 1.0, 400.0, 400.0]).astype(np.float32) * dt
        p = self.kf.predict()
        self.pred = (float(p[0, 0]), float(p[1, 0]))
        return self.pred

    def correct(self, x: float, y: float, noise: float) -> None:
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise
        self.kf.correct(np.array([[x], [y]], np.float32))

    def coast(self) -> None:
        # an unmeasured track holding its last velocity leaves the arena within
        # a second at 98fps, so bleed the velocity off instead
        self.kf.statePost[2:] *= 0.8

    def observe(self, det: Detection) -> None:
        self.intensity = (det.intensity if self.intensity is None
                          else 0.9 * self.intensity + 0.1 * det.intensity)
        self.area = (float(det.area) if self.area is None
                     else 0.9 * self.area + 0.1 * det.area)

    def grab_template(self, gray: np.ndarray) -> None:
        x, y = (int(v) for v in self.position)
        h, w = gray.shape
        x0, x1 = max(0, x - self.half), min(w, x + self.half)
        y0, y1 = max(0, y - self.half), min(h, y + self.half)
        if x1 - x0 > 10 and y1 - y0 > 10:
            self.template = gray[y0:y1, x0:x1].copy()
            self.anchor = (float(x), float(y))

    def match_template(self, gray: np.ndarray):
        """A robot that stops moving vanishes from a motion-only detector, so
        fall back to appearance around the predicted position."""
        if self.template is None:
            return None
        if np.hypot(self.pred[0] - self.anchor[0],
                    self.pred[1] - self.anchor[1]) > ANCHOR_RADIUS:
            return None
        th, tw = self.template.shape
        h, w = gray.shape
        cx, cy = self.pred
        x0 = max(0, int(cx) - tw // 2 - SEARCH_MARGIN)
        y0 = max(0, int(cy) - th // 2 - SEARCH_MARGIN)
        x1 = min(w, int(cx) + tw // 2 + SEARCH_MARGIN)
        y1 = min(h, int(cy) + th // 2 + SEARCH_MARGIN)
        window = gray[y0:y1, x0:x1]
        if window.shape[0] < th or window.shape[1] < tw:
            return None
        res = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < TEMPLATE_SCORE:
            return None
        return x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2

    @property
    def position(self) -> tuple[float, float]:
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])


def _in_bbox(point, bbox, pad: int = 0) -> bool:
    x, y, w, h = bbox
    return (x - pad <= point[0] <= x + w + pad
            and y - pad <= point[1] <= y + h + pad)


class RobotTracker:
    """Exactly two tracks with fixed, user-assigned identities. Tracks are
    never created or destroyed -- a robot that stops being detected is coasted
    and reacquired, never renumbered."""

    def __init__(self, seeds: dict[str, tuple[float, float]], mask: np.ndarray,
                 gate: float = 70.0):
        self.tracks = [Track(label, x, y) for label, (x, y) in seeds.items()]
        self.mask = mask
        self.gate = gate

    def _in_arena(self, point) -> bool:
        x, y = int(point[0]), int(point[1])
        h, w = self.mask.shape
        return 0 <= x < w and 0 <= y < h and self.mask[y, x] > 0

    def _assign(self, dets: list[Detection]) -> dict[int, Detection]:
        # a stale track has drifted from the truth (or latched onto scenery).
        # There are only ever two robots, so an unclaimed robot-sized blob is
        # almost certainly the missing one: widen the gate without limit until
        # the track can reach it. The healthy track still claims its own blob
        # first because assignment takes the globally cheapest pair each round.
        cost = np.full((len(self.tracks), len(dets)), np.inf)
        for i, t in enumerate(self.tracks):
            gate = self.gate * (1 + 0.5 * t.stale)
            for j, d in enumerate(dets):
                dist = float(np.hypot(t.pred[0] - d.cx, t.pred[1] - d.cy))
                if dist > gate and not _in_bbox(t.pred, d.bbox):
                    continue
                cost[i, j] = dist
                if t.intensity is not None:
                    cost[i, j] += 0.5 * abs(t.intensity - d.intensity)
                if t.area is not None:
                    # smoke clouds and debris showers run several times a
                    # robot's footprint; distance alone happily follows them
                    cost[i, j] += 30.0 * abs(np.log(d.area / t.area))

        assigned: dict[int, Detection] = {}
        while dets:
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            if not np.isfinite(cost[i, j]):
                break
            assigned[int(i)] = dets[int(j)]
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        return assigned

    def update(self, dets: list[Detection], gray: np.ndarray, dt: float) -> None:
        for t in self.tracks:
            t.predict(dt)
        assigned = self._assign(dets)

        for i, t in enumerate(self.tracks):
            det = assigned.get(i)
            if det is not None:
                t.correct(det.cx, det.cy, noise=4.0)
                t.observe(det)
                t.grab_template(gray)
                t.stale = 0
                t.source = "blob"
                continue

            t.stale += 1
            # clinch: the other robot's blob has swallowed this one. Follow the
            # merged blob weakly rather than template-searching, which would
            # otherwise walk off onto floor texture.
            clinch = next((d for k, d in assigned.items()
                           if k != i and _in_bbox(t.pred, d.bbox, pad=30)), None)
            if clinch is not None:
                t.correct(clinch.cx, clinch.cy, noise=400.0)
                t.source = "clinch"
                continue

            hit = t.match_template(gray)
            if hit is not None and self._in_arena(hit):
                t.correct(hit[0], hit[1], noise=25.0)
                t.source = "template"
            else:
                t.coast()
                t.source = "coast"

    def state(self) -> dict[str, dict]:
        return {t.label: {"position": t.position, "velocity": t.velocity,
                          "stale": t.stale, "source": t.source}
                for t in self.tracks}
