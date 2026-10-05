import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from splat_report import find_floaters, find_floor, load_splat, model_frame, paint

# Real SH basis for bands 1-3, in gsplat's order and normalisation (checked
# against gsplat.cuda._torch_impl._spherical_harmonics). Band 0 is the
# view-independent colour and does not change under rotation.
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
BANDS = (slice(0, 3), slice(3, 8), slice(8, 15))


def sh_basis(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3 * yy),
    ], axis=1)


def sh_rotation(rot):
    """Per-band M with basis(rot^T d) = basis(d) @ M, fitted on random directions.

    Each band maps onto itself under rotation, so the least-squares fit is
    exact. A splat's colour along d after rotating equals its old colour along
    rot^T d, so its coefficients become M @ c.
    """
    d = np.random.default_rng(0).normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    b, b_rot = sh_basis(d), sh_basis(d @ rot)
    return [np.linalg.lstsq(b[:, s], b_rot[:, s], rcond=None)[0] for s in BANDS]


def rotate_quats(q, rot):
    """Left-multiply wxyz quaternions by rot. Keeps their (unnormalised) length."""
    rw, rx, ry, rz = Rotation.from_matrix(rot).as_quat(scalar_first=True)
    w, x, y, z = q.T
    return np.stack([
        rw * w - rx * x - ry * y - rz * z,
        rw * x + rx * w + ry * z - rz * y,
        rw * y - rx * z + ry * w + rz * x,
        rw * z + rx * y - ry * x + rz * w,
    ], axis=1)


def columns(data, prefix, n):
    return np.stack([data[f"{prefix}{i}"] for i in range(n)], axis=1).astype(np.float64)


def draw(path, before, after, floor_z, lim_wide, lim_side):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for col, (title, s, xyz) in enumerate((("before", before, before["xyz"]), ("after", after, after["xyz"]))):
        ax = axes[0, col]
        ax.imshow(paint(xyz[:, 0], xyz[:, 1], xyz[:, 2], s["rgb"], lim_wide), extent=[*lim_wide[0], *lim_wide[1]])
        ax.set_title(f"{title}: top, all splats")
        opaque = s["opacity"] > 0.5
        ax = axes[1, col]
        ax.imshow(
            paint(xyz[opaque, 1], xyz[opaque, 2], xyz[opaque, 0], s["rgb"][opaque], lim_side),
            extent=[*lim_side[0], *lim_side[1]],
        )
        ax.axhline(floor_z[col], color="red", linewidth=0.8)
        ax.set_title(f"{title}: side, opaque splats, red line level at the floor")
    for ax in axes.ravel():
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(ply, run, dataset):
    ply, run, dataset = Path(ply), Path(run), Path(dataset)
    out = ply.with_name(f"{ply.stem}_clean.ply")
    s = load_splat(ply)
    names, cams, sparse = model_frame(run, dataset)

    f = find_floaters(s["xyz"], s["opacity"], cams, sparse)
    floater = f["outside"] | f["free_space"] | f["isolated"]
    keep = ~floater
    print(f"splat:      {ply}")
    print(f"gaussians:  {s['n']}")
    for label, key in (("outside region", "outside"), ("in walked air", "free_space"), ("isolated", "isolated")):
        print(f"  {label:<15} {f[key].sum():>8}")
    print(f"removed:    {floater.sum()} ({floater.mean() * 100:.1f}%), kept {keep.sum()}")

    floor = find_floor(s["xyz"][(s["opacity"] > 0.5) & keep], cams, np.random.default_rng(0))
    normal = floor["normal"]
    tilt = float(np.degrees(np.arccos(normal[2])))
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    rot = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(tilt)).as_matrix()
    print(f"levelling:  floor tilted {tilt:.2f} deg; rotating about the origin")

    ply_in = PlyData.read(str(ply))
    data = ply_in["vertex"].data[keep]
    xyz = s["xyz"][keep] @ rot.T
    for i, axis_name in enumerate("xyz"):
        data[axis_name] = xyz[:, i]
    quats = rotate_quats(columns(data, "rot_", 4), rot)
    for i in range(4):
        data[f"rot_{i}"] = quats[:, i]
    # f_rest is channel-major: index = channel * 15 + coefficient.
    rest = columns(data, "f_rest_", 45).reshape(-1, 3, 15)
    rotated = rest.copy()
    for band, m in zip(BANDS, sh_rotation(rot)):
        rotated[:, :, band] = rest[:, :, band] @ m.T
    rotated = rotated.reshape(-1, 45)
    for i in range(45):
        data[f"f_rest_{i}"] = rotated[:, i]
    PlyData(
        [PlyElement.describe(data, "vertex")],
        byte_order="<",
        comments=ply_in.comments + [f"splat_clean.py: removed {floater.sum()} floaters, levelled by {tilt:.2f} deg"],
    ).write(str(out))

    # Checks, on the file as written.
    c = load_splat(out)
    written = PlyData.read(str(out))["vertex"].data
    rng = np.random.default_rng(1)
    pick = rng.choice(c["n"], 2000, replace=False)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    colour_before = np.einsum("nk,nck->nc", sh_basis(d), rest[pick])
    colour_after = np.einsum("nk,nck->nc", sh_basis(d @ rot.T), columns(written[pick], "f_rest_", 45).reshape(-1, 3, 15))
    pos_err = np.abs(s["xyz"][keep] @ rot.T - c["xyz"]).max()
    scale = np.exp(columns(written[pick], "scale_", 3))
    cov = []
    for q in (columns(ply_in["vertex"].data[keep][pick], "rot_", 4), columns(written[pick], "rot_", 4)):
        r = Rotation.from_quat(q, scalar_first=True).as_matrix()
        cov.append(r @ (np.eye(3) * scale[:, None, :] ** 2) @ r.transpose(0, 2, 1))
    cov_err = np.abs(rot @ cov[0] @ rot.T - cov[1]).max() / np.abs(cov[0]).max()
    cams_level = cams @ rot.T
    refit = find_floor(c["xyz"][c["opacity"] > 0.5], cams_level, np.random.default_rng(0))
    tilt_after = float(np.degrees(np.arccos(min(refit["normal"][2], 1.0))))
    print("\nchecks:")
    print(f"  output:          {c['n']} gaussians, size {'ok' if c['size_ok'] else 'MISMATCH'}, "
          f"{'finite' if c['finite'] else 'NaN/Inf'}, rotations {'ok' if c['quat_ok'] else 'DEGENERATE'}")
    print(f"  floor tilt:      {tilt_after:.2f} deg after (was {tilt:.2f})")
    print(f"  positions:       max error {pos_err:.2e} (float32 rounding)")
    print(f"  colour by view:  max change {np.abs(colour_before - colour_after).max():.2e} (SH units)")
    print(f"  shape frame:     max relative covariance error {cov_err:.2e}")

    lo, hi = np.percentile(s["xyz"], [0.1, 99.9], axis=0)
    box_lo, box_hi = f["box"]
    draw(
        ply.parent / "report" / "clean.png",
        s,
        c,
        (floor["centre"][2], (rot @ floor["centre"])[2]),
        ((lo[0], hi[0]), (lo[1], hi[1])),
        ((box_lo[1], box_hi[1]), (box_lo[2], box_hi[2])),
    )
    print(f"\nwrote:      {out}")
    print(f"picture:    {ply.parent / 'report' / 'clean.png'}")
    print("note:       the cleaned file is levelled, so splat_report.py's camera mapping no longer applies to it")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python scripts/splat_clean.py <splat.ply> <train-run-dir> <processed-run-dir>")
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
