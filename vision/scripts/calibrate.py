import sys

sys.path.insert(0, ".")
import cv2
import numpy as np

from calib.homography import compute_homography, save
from capture.source import VideoFileSource

WINDOW = "calibrate (click reference points, q to finish, u to undo)"
TILE_SIZE_M = 4 * 0.3048  # confirmed arena tile size: 4ft square, 12x12 grid


def grab_frame(path: str, frame_index: int) -> np.ndarray:
    src = VideoFileSource(path)
    frame = None
    for i, (f, _ts) in enumerate(src):
        frame = f
        if i >= frame_index:
            break
    src.close()
    if frame is None:
        raise ValueError(f"could not read frame {frame_index} from {path}")
    return frame


def pick_points(frame: np.ndarray) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(display, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(WINDOW, display)

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
    cv2.destroyWindow(WINDOW)
    return points


def main(video_path: str, frame_index: int):
    frame = grab_frame(video_path, frame_index)
    print("click each reference point in the image, then press q.")
    print("press u to undo the last point.")
    pixel_pts = pick_points(frame)

    if len(pixel_pts) < 4:
        print(f"only {len(pixel_pts)} points selected, need >= 4. aborting.")
        return

    print(f"tile size: {TILE_SIZE_M:.4f}m. point 0 is the origin (0,0).")
    print("for each point after that, enter its tile offset (di,dj) from "
          "the PREVIOUS point -- fractional if not on a grid intersection.")
    world_pts = [(0.0, 0.0)]
    tile_pos = (0.0, 0.0)
    for i, (x, y) in enumerate(pixel_pts[1:], start=1):
        while True:
            raw = input(f"point {i} (pixel {x},{y}) -- tile offset from point {i - 1}, di,dj: ")
            parts = raw.strip().split(",")
            try:
                di, dj = (float(v.strip()) for v in parts)
                break
            except ValueError:
                print(f"could not parse {raw!r} as 'di,dj' -- try again.")
        tile_pos = (tile_pos[0] + di, tile_pos[1] + dj)
        world_pts.append((tile_pos[0] * TILE_SIZE_M, tile_pos[1] * TILE_SIZE_M))

    H = compute_homography(np.array(pixel_pts), np.array(world_pts))
    save(H, "calib/homography.json")
    print("saved calib/homography.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/calibrate.py <path-to-video> [frame_index]")
        sys.exit(1)
    video_path = sys.argv[1]
    frame_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(video_path, frame_index)
