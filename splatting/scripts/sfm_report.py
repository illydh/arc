import json
import sys
from pathlib import Path

import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_images_binary,
    read_model,
)

# Gate thresholds. Stated up front so a run is judged against them rather than
# post-hoc; see splatting/mds/context.md.
MIN_RATIO_GO = 0.80
MIN_RATIO_NOGO = 0.40
MAX_REPROJ_GO = 1.0
MAX_GAP_FRAC_GO = 0.05
MIN_DOMINANCE = 0.5


def camera_centers(images):
    """COLMAP stores world->camera; the center is -R^T t."""
    return {
        im.name: -qvec2rotmat(im.qvec).T @ im.tvec
        for im in images.values()
    }


def unregistered_runs(extracted, registered):
    runs, start = [], None
    for i, name in enumerate(extracted):
        if name not in registered:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(extracted) - 1))
    return runs


def submodels(colmap_dir):
    sparse = colmap_dir / "sparse"
    if not sparse.is_dir():
        return []
    return sorted(
        (d for d in sparse.iterdir() if d.is_dir() and (d / "images.bin").exists()),
        key=lambda d: int(d.name) if d.name.isdigit() else 99,
    )


def verdict(n_models, ratio, dominance, reproj, gap_frac):
    """Fragmentation is judged by dominance, not by raw sub-model count.

    Counting sub-models rejected a run with one coherent 280-image model plus
    three 12-22 image orphans, which is not the failure the count was meant to
    catch (a walk splitting into comparable pieces). What matters is whether
    one model holds most of the registered frames.
    """
    if n_models == 0 or ratio < MIN_RATIO_NOGO or dominance < MIN_DOMINANCE:
        return "NO-GO"
    if (
        dominance >= 0.9
        and ratio >= MIN_RATIO_GO
        and reproj < MAX_REPROJ_GO
        and gap_frac <= MAX_GAP_FRAC_GO
    ):
        return "GO"
    return "MARGINAL"


def main(run_dir):
    run = Path(run_dir)
    image_dir = run / "images"
    if not image_dir.is_dir():
        sys.exit(f"no extracted images at {image_dir}")

    extracted = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    models = submodels(run / "colmap")
    if not models:
        print(f"run:        {run}")
        print(f"extracted:  {len(extracted)} frames")
        print("sub-models: 0 -- COLMAP produced no reconstruction")
        print("\nVERDICT: NO-GO")
        return 1

    print(f"run:        {run}")
    print(f"extracted:  {len(extracted)} frames")
    counts = [len(read_images_binary(d / "images.bin")) for d in models]
    # sparse/0 is the model colmap_to_json consumed, so it is the one that has
    # to be dominant -- not merely the largest one somewhere on disk.
    dominance = counts[0] / sum(counts)
    print(f"sub-models: {len(models)}" + ("  <- only sparse/0 is used downstream" if len(models) > 1 else ""))
    for d, c in zip(models, counts):
        print(f"  sparse/{d.name}: {c} registered")
    if len(models) > 1:
        print(f"dominance:  {dominance * 100:.1f}% of registered frames in the largest model")

    cameras, images, points = read_model(str(models[0]), ext=".bin")
    registered = {im.name for im in images.values()}
    ratio = len(registered) / len(extracted)

    errors = np.array([p.error for p in points.values()])
    reproj = float(errors.mean()) if len(errors) else float("inf")
    obs = np.array([int((im.point3D_ids != -1).sum()) for im in images.values()])
    track_len = np.array([len(p.image_ids) for p in points.values()])

    centers = camera_centers(images)
    ordered = np.array([centers[n] for n in extracted if n in centers])
    extent = ordered.max(axis=0) - ordered.min(axis=0)
    path = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())

    runs = unregistered_runs(extracted, registered)
    longest = max((b - a + 1 for a, b in runs), default=0)
    gap_frac = longest / len(extracted)

    cam = next(iter(cameras.values()))
    print(f"\ncamera:     {cam.model} {cam.width}x{cam.height}")
    print(f"registered: {len(registered)}/{len(extracted)}  ({ratio * 100:.1f}%)")
    print(f"3D points:  {len(points)}")
    print(f"reproj err: {reproj:.3f} px mean")
    print(f"obs/image:  {obs.mean():.0f} mean (min {obs.min()}, max {obs.max()})")
    print(f"track len:  {track_len.mean():.2f} images per point")

    # Scale is arbitrary: monocular SfM recovers geometry up to a global scale.
    print(f"\ntrajectory (arbitrary scale, no metric meaning yet):")
    print(f"  bbox:     {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}")
    print(f"  path len: {path:.2f}")
    print(f"  ratio path/bbox-diagonal: {path / np.linalg.norm(extent):.2f}")

    print(f"\nunregistered runs: {len(runs)} (longest {longest} frames = {gap_frac * 100:.1f}%)")
    for a, b in sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:5]:
        print(f"  {b - a + 1:>4} frames  {extracted[a]} .. {extracted[b]}")

    result = verdict(len(models), ratio, dominance, reproj, gap_frac)
    print(f"\nVERDICT: {result}")

    (run / "report.json").write_text(
        json.dumps(
            {
                "verdict": result,
                "submodels": len(models),
                "submodel_counts": counts,
                "dominance": round(dominance, 4),
                "extracted": len(extracted),
                "registered": len(registered),
                "ratio": round(ratio, 4),
                "points3d": len(points),
                "reproj_err_px": round(reproj, 4),
                "mean_obs_per_image": round(float(obs.mean()), 1),
                "mean_track_length": round(float(track_len.mean()), 3),
                "traj_bbox": [round(float(v), 4) for v in extent],
                "traj_path_len": round(path, 4),
                "longest_unregistered_run": longest,
                "unregistered_runs": len(runs),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if result != "NO-GO" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/sfm_report.py <processed-run-dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
