import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from calib.homography import compute_homography, load, save
from capture.source import VideoFileSource
from detect.blobs import BlobDetector, arena_mask, mask_polygon
from track.tracker import RobotTracker
from viz.topdown import TopDown

CALIB_PATH = "calib/homography.json"
TILE_M = 4 * 0.3048  # arena is a 12x12 grid of 4ft tiles
LABELS = ["home", "opponent"]
COLORS = {"home": (0, 255, 0), "opponent": (0, 0, 255)}
WARMUP = 100
ENTER, ESC = 13, 27


def advance(src, n):
    frame = None
    for _ in range(n):
        ok, f, ts = src.read()
        if not ok:
            return frame, None
        frame, stamp = f, ts
    return frame, stamp


def pick_points(frame, window, prompts):
    """Collect one click per prompt, drawn as they land."""
    points = []

    def redraw():
        vis = frame.copy()
        for i, (x, y) in enumerate(points):
            color = COLORS.get(LABELS[i] if i < len(LABELS) else "", (0, 200, 255))
            cv2.circle(vis, (x, y), 10, color, 2)
            cv2.putText(vis, prompts[i], (x + 14, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(points) < len(prompts):
            cv2.putText(vis, f"click: {prompts[len(points)]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(vis, "enter = confirm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(vis, "u = undo   q = quit", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow(window, vis)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(prompts):
            points.append((x, y))
            redraw()

    cv2.setMouseCallback(window, on_click)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ESC):
            return None
        if key == ord("s"):
            return []
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ENTER and len(points) == len(prompts):
            return points


def seek_start(src, window):
    """Step through the clip to the frame tracking should begin from. The
    arena lights are off for the first half of every clip in data/."""
    frame, _ = advance(src, 1)
    while True:
        vis = frame.copy()
        cv2.putText(vis, "space=+1  n=+30  m=+300  enter=start here  q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ESC):
            return None
        if key == ENTER:
            return frame
        step = {ord(" "): 1, ord("n"): 30, ord("m"): 300}.get(key)
        if step:
            nxt, _ = advance(src, step)
            if nxt is None:
                return frame
            frame = nxt


def read_tile_span():
    """A zero span collapses the world rectangle to a point and a negative one
    silently mirrors the coordinate frame, so neither reaches the solver."""
    while True:
        raw = input("rectangle size in 4ft tiles (width,height): ")
        try:
            w, h = (float(v) for v in raw.split(","))
        except ValueError:
            print(f"could not parse {raw!r} as 'width,height' -- try again.")
            continue
        if w > 0 and h > 0:
            return w, h
        print("width and height must both be greater than 0 -- try again.")


def quick_calibrate(frame, window):
    """Four floor-grid points bounding a whole number of tiles is enough to
    flatten the floor; the arena's own corners are occluded in this footage."""
    prompts = ["grid corner: top-left", "top-right", "bottom-right", "bottom-left"]
    print("no calibration found. click 4 floor-grid intersections bounding a "
          "whole number of tiles, clockwise from top-left (s = skip, and run "
          "without the top-down view).")
    pts = pick_points(frame, window, prompts)
    if not pts:
        return None
    w, h = read_tile_span()
    world = np.array([(0, 0), (w * TILE_M, 0),
                      (w * TILE_M, h * TILE_M), (0, h * TILE_M)])
    try:
        H = compute_homography(np.array(pts, dtype=np.float64), world)
    except ValueError:
        # identical or collinear clicks. The tracking session cost a seek, a
        # warm-up and four clicks; don't discard it over the top-down view.
        print("those 4 clicks don't define a rectangle (identical or collinear "
              "points) -- continuing without the top-down view.")
        return None
    save(H, CALIB_PATH)
    print(f"saved {CALIB_PATH}")
    return H


def overlay(frame, tracker, dets, fps, n):
    vis = frame.copy()
    for d in dets:
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (120, 120, 120), 1)
    for t in tracker.tracks:
        p = tuple(int(v) for v in t.position)
        color = COLORS[t.label]
        cv2.circle(vis, p, 18, color, 2)
        cv2.putText(vis, f"{t.label} [{t.source}]", (p[0] + 22, p[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"frame {n}  {fps:.0f} fps  q=stop", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return vis


def main(video_path: str, out_path: str, start_frame: int):
    src = VideoFileSource(video_path)
    # the arena lights are off for roughly the first half of every clip in
    # data/, so starting from frame 0 means scrubbing through minutes of black
    src.seek(start_frame)
    window = "tracking"
    cv2.namedWindow(window)

    frame = seek_start(src, window)
    if frame is None:
        src.close()
        cv2.destroyAllWindows()
        return

    mask = arena_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    detector = BlobDetector(mask)
    for _ in range(WARMUP):
        nxt, ts = advance(src, 1)
        if nxt is None:
            break
        frame = nxt
        detector.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    seeds = pick_points(frame, window, ["home (Nemesis)", "opponent"])
    if not seeds:
        src.close()
        cv2.destroyAllWindows()
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker = RobotTracker(dict(zip(LABELS, seeds)), mask)
    for t in tracker.tracks:
        t.grab_template(gray)

    H = load(CALIB_PATH) if Path(CALIB_PATH).exists() else quick_calibrate(frame, window)
    top = TopDown(H, mask_polygon(mask)) if H is not None else None
    if top is not None:
        cv2.namedWindow("arena (top-down)")

    fields = ["frame", "time_s"]
    for label in LABELS:
        fields += [f"{label}_px_x", f"{label}_px_y",
                   f"{label}_x_m", f"{label}_y_m", f"{label}_source"]

    n = 0
    prev_ts = None
    fps = 0.0
    clock = time.perf_counter()
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        while True:
            ok, frame, ts = src.read()
            if not ok:
                break
            n += 1
            dt = 1.0 / 98.0 if prev_ts is None else max(ts - prev_ts, 1e-3)
            prev_ts = ts

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets, _ = detector.apply(gray, learning_rate=0.001)
            tracker.update(dets, gray, dt)
            state = tracker.state()

            row = {"frame": n, "time_s": round(ts, 4)}
            world = {}
            for label, s in state.items():
                px, py = s["position"]
                row[f"{label}_px_x"] = round(px, 2)
                row[f"{label}_px_y"] = round(py, 2)
                row[f"{label}_source"] = s["source"]
                if top is not None:
                    wx, wy = top.world_of((px, py))
                    world[label] = (wx, wy)
                    row[f"{label}_x_m"] = round(wx, 4)
                    row[f"{label}_y_m"] = round(wy, 4)
            writer.writerow(row)

            if n % 10 == 0:
                now = time.perf_counter()
                fps = 10.0 / (now - clock)
                clock = now
            cv2.imshow(window, overlay(frame, tracker, dets, fps, n))
            if top is not None:
                cv2.imshow("arena (top-down)", top.render(world, COLORS))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ESC):
                break

    src.close()
    cv2.destroyAllWindows()
    print(f"tracked {n} frames -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/track_fight.py <path-to-video> "
              "[out.csv] [start_frame]")
        sys.exit(1)
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else "tracks.csv",
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
