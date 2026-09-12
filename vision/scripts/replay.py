import sys
import time

sys.path.insert(0, ".")
from capture.source import VideoFileSource


def main(path: str, max_frames: int = 500):
    src = VideoFileSource(path)
    n = 0
    t0 = time.perf_counter()
    for frame, ts in src:
        n += 1
        if n >= max_frames:
            break
    elapsed = time.perf_counter() - t0
    src.close()

    print(f"path: {path}")
    print(f"frames read: {n}")
    print(f"wall time: {elapsed:.3f}s")
    print(f"throughput: {n / elapsed:.1f} fps")
    print(f"last frame shape: {frame.shape}, last timestamp: {ts:.3f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/replay.py <path-to-video> [max_frames]")
        sys.exit(1)
    path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(path, max_frames)
