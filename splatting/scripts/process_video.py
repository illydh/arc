import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from nerfstudio.process_data.colmap_utils import colmap_to_json

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "video_inside_battlebox.mp4"
PROCESSED = ROOT / "data" / "processed"

# Scoring resolution for the blur pass. Small enough that decode dominates,
# large enough that motion blur still separates from sharp frames.
SCORE_W, SCORE_H = 320, 180

MATCHERS = {
    "sequential": "sequential_matcher",
    "vocab_tree": "vocab_tree_matcher",
    "exhaustive": "exhaustive_matcher",
}


def step(cmd):
    print("+ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    num, den = out[0].split("/")
    return int(out[1]), float(num) / float(den)


def blur_scores(video, start, duration, spacing):
    """Laplacian variance per candidate frame, decoded small and piped raw.

    ns-process-data has no blur rejection -- its min_blur_score is Polycam-only
    and video mode just uses ffmpeg's `thumbnail` filter -- and this is
    handheld phone footage, so sharpness selection is done here instead.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", f"select='not(mod(n\\,{spacing}))',scale={SCORE_W}:{SCORE_H},format=gray",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    nbytes = SCORE_W * SCORE_H
    scores = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            gray = np.frombuffer(buf, np.uint8).reshape(SCORE_H, SCORE_W)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return np.array(scores)


def pick_sharpest(scores, oversample):
    """Ordinal of the sharpest candidate in each window of `oversample`."""
    return [
        start + int(scores[start:start + oversample].argmax())
        for start in range(0, len(scores) - oversample + 1, oversample)
    ]


def extract(video, start, duration, spacing, keep, image_dir):
    """Extract every `spacing`-th frame, then keep only the chosen ordinals.

    ffmpeg's expression parser blows up on a select= built from a few hundred
    eq(n\\,i) terms, so frames cannot be picked individually. Extracting the
    regular candidate set and pruning afterwards is what scales.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"select='not(mod(n\\,{spacing}))'",
            "-fps_mode", "passthrough", "-q:v", "2",
            str(image_dir / "cand_%05d.jpg")]
    step(cmd)

    candidates = sorted(image_dir.glob("cand_*.jpg"))
    chosen = {candidates[i] for i in keep if i < len(candidates)}
    for path in candidates:
        if path not in chosen:
            path.unlink()
    for n, path in enumerate(sorted(chosen), start=1):
        path.rename(image_dir / f"frame_{n:05d}.jpg")
    return sorted(image_dir.glob("frame_*.jpg"))


def downscale(image_dir, levels):
    for i in range(1, levels + 1):
        target = Path(f"{image_dir}_{2 ** i}")
        target.mkdir(parents=True, exist_ok=True)
        step(["ffmpeg", "-v", "error", "-y", "-noautorotate",
              "-i", str(image_dir / "frame_%05d.jpg"),
              "-vf", f"scale=iw/{2 ** i}:ih/{2 ** i}", "-q:v", "2",
              str(target / "frame_%05d.jpg")])


def run_colmap(image_dir, colmap_dir, matcher, tree):
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    # COLMAP 4.2 renamed SiftExtraction/SiftMatching to FeatureExtraction/
    # FeatureMatching. nerfstudio 1.1.5 still emits the 3.x names, so
    # ns-process-data cannot drive this COLMAP build at all.
    step(["colmap", "feature_extractor",
          "--database_path", str(db), "--image_path", str(image_dir),
          "--ImageReader.single_camera", "1",
          "--ImageReader.camera_model", "OPENCV",
          "--FeatureExtraction.use_gpu", "0"])

    # Loop closure would help -- a walkthrough revisits the same corners and
    # plain sequential matching drifts -- but COLMAP 4.2 reads only faiss
    # vocab trees (it dropped flann in May 2025), and the tree nerfstudio
    # downloads is the legacy flann one. So it needs an explicit faiss tree.
    match = ["colmap", MATCHERS[matcher], "--database_path", str(db),
             "--FeatureMatching.use_gpu", "0"]
    if matcher == "sequential" and tree is not None:
        match += ["--SequentialMatching.loop_detection", "1",
                  "--SequentialMatching.vocab_tree_path", str(tree)]
    elif matcher == "vocab_tree":
        if tree is None:
            raise SystemExit("--matcher vocab_tree requires --vocab-tree PATH")
        match += ["--VocabTreeMatching.vocab_tree_path", str(tree)]
    step(match)

    step(["colmap", "mapper",
          "--database_path", str(db), "--image_path", str(image_dir),
          "--output_path", str(sparse),
          "--Mapper.ba_global_function_tolerance", "1e-6"])

    model = sparse / "0"
    if (model / "cameras.bin").exists():
        step(["colmap", "bundle_adjuster",
              "--input_path", str(model), "--output_path", str(model),
              "--BundleAdjustment.refine_principal_point", "1"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--start", type=float, default=0.0, help="segment start, seconds")
    ap.add_argument("--end", type=float, help="segment end, seconds")
    ap.add_argument("--matcher", default="sequential", choices=list(MATCHERS))
    ap.add_argument("--vocab-tree", type=Path,
                    help="faiss-format COLMAP vocab tree; enables loop detection")
    ap.add_argument("--downscales", type=int, default=3)
    ap.add_argument("--name")
    ap.add_argument("--video", type=Path, default=VIDEO)
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"no such video: {args.video}")

    name = args.name or f"{args.matcher}_{args.frames}"
    out = PROCESSED / name
    image_dir = out / "images"
    start_time = time.time()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    nb_frames, fps = probe(args.video)
    duration = (args.end - args.start) if args.end else None
    segment = int(duration * fps) if duration else nb_frames - int(args.start * fps)
    spacing = max(1, segment // (args.frames * args.oversample))
    print(f"{nb_frames} frames @ {fps:.2f} fps; segment {segment} frames "
          f"-> scoring every {spacing}th, keeping sharpest of each "
          f"{args.oversample}", flush=True)

    scores = blur_scores(args.video, args.start, duration, spacing)
    keep = pick_sharpest(scores, args.oversample)
    kept = scores[keep]
    print(f"scored {len(scores)} candidates, selecting {len(keep)}", flush=True)
    print(f"blur (Laplacian var): all median {np.median(scores):.1f}, "
          f"kept median {np.median(kept):.1f}, kept min {kept.min():.1f}", flush=True)

    written = extract(args.video, args.start, duration, spacing, keep, image_dir)
    print(f"extracted {len(written)} frames -> {image_dir}", flush=True)
    downscale(image_dir, args.downscales)

    model = run_colmap(image_dir, out / "colmap", args.matcher, args.vocab_tree)
    registered = colmap_to_json(model, out) if (model / "cameras.bin").exists() else 0
    elapsed = time.time() - start_time

    (out / "run.json").write_text(
        json.dumps(
            {
                "seconds": round(elapsed, 1),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "video": str(args.video),
                "segment_start": args.start,
                "segment_end": args.end,
                "matcher": args.matcher,
                "vocab_tree": str(args.vocab_tree) if args.vocab_tree else None,
                "source_frames": nb_frames,
                "segment_frames": segment,
                "spacing": spacing,
                "oversample": args.oversample,
                "extracted": len(written),
                "registered": registered,
                "blur_median_all": round(float(np.median(scores)), 2),
                "blur_median_kept": round(float(np.median(kept)), 2),
                "blur_min_kept": round(float(kept.min()), 2),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nregistered {registered}/{len(written)} in {elapsed / 60:.1f} min "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
