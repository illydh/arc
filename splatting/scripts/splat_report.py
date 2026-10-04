import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as Polygon
from plyfile import PlyData
from scipy.spatial import ConvexHull, cKDTree
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction

# Thresholds. Stated up front so a splat is judged against them rather than
# post-hoc; see splatting/mds/context.md. Lengths are in model units:
# nerfstudio scales the scene so the camera path fits inside radius 1.0.
REGION_MARGIN = 1.25
FREE_SPACE_R = 0.02
ISOLATED_K = 8
ISOLATED_FACTOR = 5.0
FAINT = 0.1
SOLID = 0.9
OVERSIZED = 0.1
NEEDLE = 100
FLOOR_THICK = 0.02
FLOOR_CELL = 0.05
FLOOR_TILT_MAX_DEG = 10
FLOOR_COVER_MIN = 0.5
PSNR_GO = 25
PSNR_NOGO = 18
SSIM_GO = 0.85
FLOATER_GO = 0.05
FLOATER_NOGO = 0.5

SH_C0 = 0.28209479177387814


def load_splat(path):
    with open(path, "rb") as f:
        header_len = f.read(16384).index(b"end_header\n") + len(b"end_header\n")
    v = PlyData.read(str(path))["vertex"].data
    size_ok = path.stat().st_size == header_len + len(v) * v.dtype.itemsize
    finite = all(np.isfinite(v[n]).all() for n in v.dtype.names)
    quat = np.linalg.norm(np.stack([v[f"rot_{i}"] for i in range(4)], axis=1), axis=1)
    return {
        "n": len(v),
        "size_ok": size_ok,
        "finite": finite,
        # splatfacto stores unnormalised quaternions and normalises at render
        # time, so only a zero-length one is a defect.
        "quat_ok": bool(quat.min() > 1e-6),
        "xyz": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64),
        "opacity": 1 / (1 + np.exp(-v["opacity"].astype(np.float64))),
        "scale": np.sort(np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)), axis=1),
        "rgb": np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1), 0, 1),
    }


def model_frame(run, dataset):
    """Training cameras and the SfM cloud, in the frame the splat is exported in."""
    meta = json.loads((dataset / "transforms.json").read_text())
    dp = json.loads((run / "dataparser_transforms.json").read_text())
    # dataparser_transforms.json maps raw COLMAP coordinates. transforms.json
    # and sparse_pc.ply already have applied_transform baked in, so undo it.
    applied = np.vstack([meta["applied_transform"], [0, 0, 0, 1]])
    m = np.vstack([dp["transform"], [0, 0, 0, 1]]) @ np.linalg.inv(applied)

    def place(p):
        return (p @ m[:3, :3].T + m[:3, 3]) * dp["scale"]

    # The dataparser sorts frames by filename before splitting train/eval.
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    names = [Path(f["file_path"]).name for f in frames]
    cams = place(np.array([f["transform_matrix"] for f in frames])[:, :3, 3])
    sp = PlyData.read(str(dataset / meta["ply_file_path"]))["vertex"].data
    sparse = place(np.stack([sp["x"], sp["y"], sp["z"]], axis=1).astype(np.float64))
    return names, cams, sparse


def eval_names(run, names):
    fraction = float(re.search(r"train_split_fraction: ([\d.]+)", (run / "config.yml").read_text()).group(1))
    return [names[i] for i in get_train_eval_split_fraction(names, fraction)[1]]


def signature(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float64)
    return (g - g.mean()) / (g.std() + 1e-9)


def score_pairs(pairs, dataset, held_out):
    """Per-view PSNR from side-by-side images (real photo left, render right).

    Each pair is named by matching its left half against the held-out frames,
    because the training log samples held-out views in random order.
    """
    thumbs = {n: signature(cv2.imread(str(dataset / "images_8" / n))) for n in held_out}
    views = {}
    for step, img in pairs:
        w = img.shape[1] // 2
        gt, pred = img[:, :w], img[:, w : 2 * w]
        sig = signature(gt)
        name = min(thumbs, key=lambda n: np.mean((thumbs[n] - sig) ** 2))
        if name not in views:
            mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2) / 255**2
            views[name] = {"psnr": float(-10 * np.log10(mse)), "step": step, "img": img}
        if len(views) == len(held_out):
            break
    return views


def training_log(run):
    ea = EventAccumulator(str(next(run.glob("events.out.tfevents.*"))), size_guidance={"scalars": 0, "images": 0})
    ea.Reload()
    tag = "Eval Images Metrics Dict (all images)/"
    curve = [(e.step, e.value) for e in ea.Scalars(tag + "psnr")]
    best = max(v for _, v in curve)
    return {
        "source": f"training log, step {curve[-1][0]}",
        "psnr": curve[-1][1],
        "ssim": ea.Scalars(tag + "ssim")[-1].value,
        "lpips": ea.Scalars(tag + "lpips")[-1].value,
        "train_psnr": float(np.mean([e.value for e in ea.Scalars("Train Metrics Dict/psnr")[-100:]])),
        "plateau_step": next(s for s, v in curve if v >= best - 0.25),
        "pairs": (
            (e.step, cv2.imdecode(np.frombuffer(e.encoded_image_string, np.uint8), cv2.IMREAD_COLOR))
            for e in reversed(ea.Images("Eval Images/img"))
        ),
    }


def find_floor(pts, cams, rng):
    """Largest near-horizontal plane below the cameras, by RANSAC."""
    sample = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    tri = sample[rng.integers(len(sample), size=(1000, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    tri, n = tri[length > 1e-12], n[length > 1e-12] / length[length > 1e-12, None]
    n[n[:, 2] < 0] *= -1
    d = np.einsum("ij,ij->i", n, tri[:, 0])
    cx, cy, cz = np.median(cams, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = (d - n[:, 0] * cx - n[:, 1] * cy) / n[:, 2]
    ok = (n[:, 2] > np.cos(np.radians(30))) & (height < cz)
    if not ok.any():
        return None
    n, d = n[ok], d[ok]
    best = np.argmax((np.abs(sample @ n.T - d) < FLOOR_THICK).sum(axis=0))
    inliers = np.abs(pts @ n[best] - d[best]) < FLOOR_THICK
    centre = pts[inliers].mean(axis=0)
    normal = np.linalg.svd(pts[inliers] - centre, full_matrices=False)[2][2]
    normal *= np.sign(normal[2])
    inliers = np.abs((pts - centre) @ normal) < FLOOR_THICK
    return {"normal": normal, "centre": centre, "inliers": inliers}


def floor_cover(floor_xy, cams):
    hull = Polygon(cams[ConvexHull(cams[:, :2]).vertices, :2])
    lo = np.floor(cams[:, :2].min(axis=0) / FLOOR_CELL).astype(int)
    hi = np.ceil(cams[:, :2].max(axis=0) / FLOOR_CELL).astype(int)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cells = cells[hull.contains_points((cells + 0.5) * FLOOR_CELL)]
    filled = {tuple(c) for c in np.unique(np.floor(floor_xy / FLOOR_CELL).astype(int), axis=0)}
    return sum(tuple(c) in filled for c in cells) / len(cells)


def paint(u, v, depth, rgb, lim, px=900):
    """Orthographic point render; the splat with the largest depth wins each pixel."""
    (u0, u1), (v0, v1) = lim
    k = px / max(u1 - u0, v1 - v0)
    ok = (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
    order = np.argsort(depth[ok])
    img = np.ones((int((v1 - v0) * k) + 1, int((u1 - u0) * k) + 1, 3))
    img[((v1 - v[ok]) * k).astype(int)[order], ((u[ok] - u0) * k).astype(int)[order]] = rgb[ok][order]
    return img


def draw_views(path, s, cams, floater, box_lo, box_hi, worst):
    xyz = s["xyz"]
    core = (s["opacity"] > 0.5) & ~floater
    wide_lo, wide_hi = np.percentile(xyz, [0.1, 99.9], axis=0)
    marked = np.where(floater[:, None], [0.85, 0.1, 0.1], [0.6, 0.6, 0.6])
    # (title, horizontal axis, vertical axis, depth axis, depth sign)
    views = [("top", 0, 1, 2, 1), ("front", 0, 2, 1, -1), ("side", 1, 2, 0, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for col, (title, a, b, c, sign) in enumerate(views):
        lim = ((box_lo[a], box_hi[a]), (box_lo[b], box_hi[b]))
        ax = axes[0, col]
        ax.imshow(
            paint(xyz[core, a], xyz[core, b], sign * xyz[core, c], s["rgb"][core], lim),
            extent=[*lim[0], *lim[1]],
        )
        ax.plot(cams[:, a], cams[:, b], color="cyan", linewidth=0.8)
        ax.set_title(title)

        lim = ((wide_lo[a], wide_hi[a]), (wide_lo[b], wide_hi[b]))
        ax = axes[1, col]
        # The scene is painted over the floaters. At this zoom one pixel holds
        # many splats, and painting floaters last turns the whole scene red.
        ax.imshow(paint(xyz[:, a], xyz[:, b], (~floater).astype(float), marked, lim), extent=[*lim[0], *lim[1]])
        ax.add_patch(
            plt.Rectangle(
                (box_lo[a], box_lo[b]), box_hi[a] - box_lo[a], box_hi[b] - box_lo[b], fill=False, edgecolor="blue"
            )
        )
        ax.set_title(title)
    for name, (x, y) in worst:
        axes[0, 0].plot(x, y, "o", color="red", markersize=5)
        axes[0, 0].annotate(name, (x, y), color="red", fontsize=8)
    for ax in axes.ravel():
        ax.set_aspect("equal")
    fig.suptitle(
        "Top row: opaque splats in the mapped region, camera path in cyan, worst held-out views in red.\n"
        "Bottom row: all splats, scene in grey over floaters in red, mapped region in blue."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_distributions(path, s, knn, isolated_at):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(s["opacity"], bins=50)
    for x in (FAINT, SOLID):
        axes[0].axvline(x, color="red")
    axes[0].set_title("opacity (red: faint / solid bands)")
    axes[1].hist(np.log10(s["scale"][:, 2]), bins=100)
    axes[1].axvline(np.log10(OVERSIZED), color="red")
    axes[1].set_title("log10 longest axis (red: oversized)")
    axes[2].hist(np.log10(knn), bins=100)
    axes[2].axvline(np.log10(isolated_at), color="red")
    axes[2].set_title(f"log10 mean distance to {ISOLATED_K} nearest splats (red: isolated)")
    for ax in axes:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_heldout(path, views):
    ranked = sorted(views.items(), key=lambda kv: kv[1]["psnr"], reverse=True)
    shown = ranked[:3] + ranked[-3:] if len(ranked) > 6 else ranked
    fig, axes = plt.subplots(len(shown), 1, figsize=(14, 4.2 * len(shown)), squeeze=False)
    for ax, (name, view) in zip(axes[:, 0], shown):
        ax.imshow(cv2.cvtColor(view["img"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}: {view['psnr']:.1f} dB  (real photo left, render right)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def verdict(intact, psnr, ssim, floater_weight, floor_ok):
    if not intact or psnr < PSNR_NOGO or floater_weight > FLOATER_NOGO:
        return "NO-GO"
    if psnr >= PSNR_GO and ssim >= SSIM_GO and floater_weight <= FLOATER_GO and floor_ok:
        return "GO"
    return "MARGINAL"


def main(ply, run, dataset, eval_dir=None):
    ply, run, dataset = Path(ply), Path(run), Path(dataset)
    out = ply.parent / "report"
    out.mkdir(exist_ok=True)

    s = load_splat(ply)
    xyz, opacity, scale = s["xyz"], s["opacity"], s["scale"]
    intact = s["size_ok"] and s["finite"] and s["quat_ok"]
    print(f"splat:      {ply}")
    print(f"gaussians:  {s['n']}")
    print(f"integrity:  size {'ok' if s['size_ok'] else 'MISMATCH'}, "
          f"values {'finite' if s['finite'] else 'NaN/Inf PRESENT'}, "
          f"rotations {'ok' if s['quat_ok'] else 'DEGENERATE'}")

    names, cams, sparse = model_frame(run, dataset)
    # auto_scale_poses normalises by the largest single coordinate, not the norm.
    cam_radius = float(np.abs(cams).max())
    cam_box = cams.max(axis=0) - cams.min(axis=0)
    print(f"\ncameras:    {len(cams)} posed, largest coordinate {cam_radius:.3f} (expect 1.000)")
    print(f"  path box: {cam_box[0]:.2f} x {cam_box[1]:.2f} x {cam_box[2]:.2f}")

    full = xyz.max(axis=0) - xyz.min(axis=0)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    robust = hi - lo
    aspect = float(min(robust[:2]) / max(robust[:2]))
    print("\nextent (model units, no metric meaning yet):")
    print(f"  full box:    {full[0]:.2f} x {full[1]:.2f} x {full[2]:.2f}")
    print(f"  1-99% box:   {robust[0]:.2f} x {robust[1]:.2f} x {robust[2]:.2f}  (footprint aspect {aspect:.2f})")
    print(f"  vs path box: {robust[0] / cam_box[0]:.2f}x by {robust[1] / cam_box[1]:.2f}x")

    sp_lo, sp_hi = np.percentile(sparse, [1, 99], axis=0)
    mid, half = (sp_lo + sp_hi) / 2, (sp_hi - sp_lo) / 2 * REGION_MARGIN
    box_lo, box_hi = mid - half, mid + half
    outside = (np.abs(xyz - mid) > half).any(axis=1)
    cam_dist = cKDTree(cams).query(xyz, workers=-1)[0]
    free = (cam_dist < FREE_SPACE_R) & (opacity > 0.5)
    knn = cKDTree(xyz).query(xyz, k=ISOLATED_K + 1, workers=-1)[0][:, 1:].mean(axis=1)
    isolated_at = ISOLATED_FACTOR * float(np.median(knn))
    isolated = knn > isolated_at
    floater = outside | free | isolated
    # Visual weight: opacity x the solid angle the footprint subtends from the
    # nearest training camera. World-space area alone lets a handful of huge
    # distant splats outweigh the whole scene.
    angle = np.minimum(np.pi * scale[:, 1] * scale[:, 2] / np.maximum(cam_dist, FREE_SPACE_R) ** 2, 4 * np.pi)
    weight = opacity * angle

    def share(mask):
        return float(mask.mean()), float(weight[mask].sum() / weight.sum())

    floaters = {k: share(m) for k, m in
                (("outside", outside), ("free_space", free), ("isolated", isolated), ("any", floater))}
    print("\nfloaters:            count    visual weight")
    for label, key in (("outside region", "outside"), ("in walked air", "free_space"),
                       ("isolated", "isolated"), ("any of the above", "any")):
        print(f"  {label:<17} {floaters[key][0] * 100:5.1f}%   {floaters[key][1] * 100:5.1f}%")

    mix = {
        "faint": float((opacity < FAINT).mean()),
        "solid": float((opacity > SOLID).mean()),
        "oversized": float((scale[:, 2] > OVERSIZED).mean()),
        "needle": float((scale[:, 2] / scale[:, 0] > NEEDLE).mean()),
    }
    print(f"\nquality mix: {mix['solid'] * 100:.1f}% solid, {mix['faint'] * 100:.1f}% faint, "
          f"{mix['oversized'] * 100:.2f}% oversized, {mix['needle'] * 100:.1f}% needle-shaped")

    core = (opacity > 0.5) & ~floater
    floor = find_floor(xyz[core], cams, np.random.default_rng(0))
    tilt = cover = cam_height = None
    if floor is not None:
        tilt = float(np.degrees(np.arccos(min(floor["normal"][2], 1.0))))
        cover = floor_cover(xyz[core][floor["inliers"], :2], cams)
        cam_height = float(np.median((cams - floor["centre"]) @ floor["normal"]))
    floor_ok = floor is not None and tilt <= FLOOR_TILT_MAX_DEG and cover >= FLOOR_COVER_MIN
    if floor is None:
        print("\nfloor:      no near-horizontal plane found below the cameras")
    else:
        print(f"\nfloor:      {'present' if floor_ok else 'WEAK'}")
        print(f"  splats on the plane: {floor['inliers'].mean() * 100:.1f}% of the opaque core")
        print(f"  tilt from level:     {tilt:.1f} deg")
        print(f"  covers:              {cover * 100:.1f}% of the camera-path footprint")
        print(f"  camera height above: {cam_height:.3f} median")

    held_out = eval_names(run, names)
    log = training_log(run)
    if eval_dir is None:
        scores = {k: log[k] for k in ("source", "psnr", "ssim", "lpips")}
        pairs = log["pairs"]
    else:
        results = json.loads((Path(eval_dir) / "metrics.json").read_text())["results"]
        scores = {"source": "ns-eval on the final checkpoint", **{k: results[k] for k in ("psnr", "ssim", "lpips")}}
        pairs = ((None, cv2.imread(str(p))) for p in sorted((Path(eval_dir) / "renders").glob("eval_img_*.png")))
    views = score_pairs(pairs, dataset, held_out)
    ranked = sorted(views.items(), key=lambda kv: kv[1]["psnr"])
    per_view = np.array([v["psnr"] for _, v in ranked])
    print(f"\nheld-out views ({scores['source']}):")
    print(f"  PSNR {scores['psnr']:.2f} dB, SSIM {scores['ssim']:.3f}, LPIPS {scores['lpips']:.3f}")
    print(f"  training-view PSNR {log['train_psnr']:.1f} dB over the last 1,000 steps "
          f"(gap {log['train_psnr'] - scores['psnr']:.1f} dB)")
    print(f"  held-out PSNR within 0.25 dB of its best from step {log['plateau_step']}")
    print(f"  per view: {len(views)}/{len(held_out)} found, mean {per_view.mean():.2f} dB, "
          f"min {per_view.min():.2f}, max {per_view.max():.2f}")
    print("  worst five:")
    for name, view in ranked[:5]:
        print(f"    {name}  {view['psnr']:.1f} dB")

    worst = [(name.removesuffix(".jpg"), cams[names.index(name), :2]) for name, _ in ranked[:5]]
    draw_views(out / "views.png", s, cams, floater, box_lo, box_hi, worst)
    draw_distributions(out / "distributions.png", s, knn, isolated_at)
    draw_heldout(out / "heldout.png", views)

    result = verdict(intact, scores["psnr"], scores["ssim"], floaters["any"][1], floor_ok)
    print(f"\npictures:   {out}/views.png, distributions.png, heldout.png")
    print(f"\nVERDICT: {result}")

    (out / "report.json").write_text(
        json.dumps(
            {
                "verdict": result,
                "gaussians": s["n"],
                "intact": bool(intact),
                "camera_max_radius": round(cam_radius, 4),
                "full_box": [round(float(v), 3) for v in full],
                "robust_box": [round(float(v), 3) for v in robust],
                "footprint_aspect": round(aspect, 3),
                "floaters": {k: {"count": round(c, 4), "weight": round(w, 4)} for k, (c, w) in floaters.items()},
                "quality_mix": {k: round(v, 4) for k, v in mix.items()},
                "floor": None if floor is None else {
                    "present": bool(floor_ok),
                    "tilt_deg": round(tilt, 2),
                    "cover": round(cover, 4),
                    "camera_height": round(cam_height, 4),
                },
                "heldout": {
                    "source": scores["source"],
                    "psnr": round(scores["psnr"], 3),
                    "ssim": round(scores["ssim"], 4),
                    "lpips": round(scores["lpips"], 4),
                    "train_psnr": round(log["train_psnr"], 3),
                    "plateau_step": log["plateau_step"],
                    "per_view_psnr": {name: round(view["psnr"], 2) for name, view in ranked},
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if result != "NO-GO" else 1


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("usage: python scripts/splat_report.py <splat.ply> <train-run-dir> <processed-run-dir> [<eval-dir>]")
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
