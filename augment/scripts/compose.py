import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vision"))
from detect.blobs import arena_mask  # noqa: E402

from classes import NAMES, NON_NEMESIS  # noqa: E402

AUGMENT = ROOT / "augment"
REAL = AUGMENT / "data" / "real"
PLATES = AUGMENT / "data" / "plates"
SPRITES = AUGMENT / "data" / "sprites"
# These images train the detector's "every other robot" class, so Nemesis is
# not in them: an unlabelled Nemesis would teach the detector to ignore it,
# and one labelled from an approximate mesh would teach it the wrong robot.
# Gigabyte is another robot, so its mesh is used.
MESHES = {"gigabyte": AUGMENT / "data" / "meshes" / "Meshy_AI_Gigabyte_0918035233_texture_obj"}
# The clips whose plates are composed on. fight1 has three robots and no
# labelled frames, so it supplies only hand-picked cut-outs (robots.py).
CLIPS = ("fight2", "fight3", "fight4", "fight5")
CUTOUT_CLIPS = ("fight1",) + CLIPS

# The camera looks down more steeply at the near floor than at the far wall.
# These bracket what the floor grid's foreshortening shows across the frame.
ELEVATIONS = (45, 55, 65)
YAWS = 16
SPRITE = 384
SUPERSAMPLE = 3
# One texture sample per triangle shows the mesh's facets. Each triangle
# covers about 80 texels, so the texture is averaged over that first.
TEXTURE_BLUR = 4.0
# The meshes carry no real scale (both are normalised to 0.12 units tall), so
# a robot is sized from the real ones: a box as wide as theirs at the same
# floor row, for a footprint of this many mesh units.
TYPICAL_FOOTPRINT = 0.40
SIZE_JITTER = (0.85, 1.2)
MESH_SHARE = 0.15
# A detector also needs frames with one robot in view and frames with none.
ONE_ROBOT, EMPTY = 0.10, 0.05
# Half the layouts put the two robots within reach of each other: combat is
# what the detector has to cope with, and what the real labelled frames lack.
CONTACT_SHARE = 0.5
MIN_VISIBLE = 0.45
FLOOR_MARGIN = 70
# Robots are stood only within this distance of where a real robot has stood
# in the same clip. That keeps them off the hazards by the far wall, which
# the arena mask includes.
STOOD_NEAR = 90
# A cut-out's scale and perspective are right where it was cut. It may be
# slid this far, further along a row than across rows, and is resized by the
# real robots' size at the new row.
MOVE = (220, 90)
# Measured on fight4. A frame differs from the plate on untouched floor by
# 10.8 grey levels (std): the plate is a median and has no sensor noise, so a
# composite needs it put back. A render is far sharper than the footage
# (Laplacian std 45 against 23 inside real robots, noise included); blurring
# it by one pixel and then adding the noise brings the two together.
NOISE_GREY = 10.8
# No frame from the colour camera exists yet, so its noise is a guess: about
# half the monochrome camera's, with a little of it in colour.
NOISE_COLOUR = (6.0, 2.0)
NOISE_GRAIN = 0.8
RENDER_BLUR = 1.0
SHADOW = 0.6


def load_mesh(folder):
    v, vt, vn, f, ft, fn = [], [], [], [], [], []
    for line in open(next(folder.glob("*.obj"))):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            v.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == "vn":
            vn.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            idx = [q.split("/") for q in p[1:]]
            for k in range(1, len(idx) - 1):
                tri = (idx[0], idx[k], idx[k + 1])
                f.append([int(t[0]) - 1 for t in tri])
                ft.append([int(t[1]) - 1 for t in tri])
                fn.append([int(t[2]) - 1 for t in tri])
    tex = cv2.GaussianBlur(cv2.imread(str(next(folder.glob("*.png")))), (0, 0), TEXTURE_BLUR)
    return np.array(v), np.array(vt), np.array(vn), np.array(f), np.array(ft), np.array(fn), tex


def render(mesh, yaw, elev):
    """Painter's algorithm, one colour per triangle. With 30,000 triangles in
    a few hundred pixels each is smaller than a pixel after downsampling, so
    this matches a textured z-buffer at a fraction of the cost. Returns BGRA
    and the sprite's pixels per mesh unit."""
    v, vt, vn, f, ft, fn, tex = mesh
    p = v - (v.max(0) + v.min(0)) / 2
    y, e = np.radians(yaw), np.radians(elev)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    q = p @ ry.T @ rx.T
    size = SPRITE * SUPERSAMPLE
    # one scale for every view, so a robot does not change size as it turns
    scale = size * 0.48 / np.linalg.norm(p, axis=1).max()
    pts = np.stack([q[:, 0] * scale + size / 2, -q[:, 1] * scale + size / 2], 1)

    # the mesh's own vertex normals, averaged per triangle: lit by face
    # normals alone, a curved surface shows every facet
    normal = vn[fn].mean(1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    light = np.array([0.25, 1.0, 0.2]) / np.linalg.norm([0.25, 1.0, 0.2])
    shade = 0.55 + 0.45 * np.clip(normal @ light, 0, 1)

    h, w = tex.shape[:2]
    uv = vt[ft].mean(1)
    colour = tex[np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1),
                 np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)] * shade[:, None]

    img = np.zeros((size, size, 4), np.uint8)
    for i in np.argsort(q[f][:, :, 2].mean(1)):
        b, g, r = colour[i]
        cv2.fillConvexPoly(img, pts[f[i]].astype(np.int32), (int(b), int(g), int(r), 255))
    return cv2.resize(img, (SPRITE, SPRITE), interpolation=cv2.INTER_AREA), scale / SUPERSAMPLE


def sprite_bank(name):
    out = SPRITES / name
    made = {"elevations": ELEVATIONS, "yaws": YAWS, "sprite": SPRITE, "texture_blur": TEXTURE_BLUR}
    meta = json.loads((out / "bank.json").read_text()) if (out / "bank.json").exists() else {}
    if {k: meta.get(k) for k in made} != json.loads(json.dumps(made)):
        out.mkdir(parents=True, exist_ok=True)
        mesh = load_mesh(MESHES[name])
        for elev in ELEVATIONS:
            for k in range(YAWS):
                img, ppu = render(mesh, 360 * k / YAWS, elev)
                cv2.imwrite(str(out / f"e{elev}_y{k:02d}.png"), img)
        meta = {**made, "pixels_per_unit": ppu}
        (out / "bank.json").write_text(json.dumps(meta))
    return out, meta["pixels_per_unit"]


def real_robots(clip):
    return [r for f in json.loads((REAL / clip / "manifest.json").read_text())["items"]
            for r in f["robots"] if r["on_change"] >= 0.75 and r["moving"] >= 0.15]


def size_model():
    """Width of a real robot's box against the row its box ends on, over every
    clip: one fight alone has too few robots to fit (fight2: one huge, one small)."""
    y, w = np.array([(r["bbox"][3], r["bbox"][2] - r["bbox"][0])
                     for c in CLIPS for r in real_robots(c)], float).T
    keep = np.ones(len(y), bool)
    for _ in range(2):
        b, a = np.polyfit(y[keep], w[keep], 1)
        rest = w - (a + b * y)
        keep = np.abs(rest) < 2 * rest[keep].std()
    return lambda row: float(np.clip(a + b * row, 40, 420))


def mesh_robot(name, rng, width, x, y, plate_h):
    bank, ppu = sprite_bank(name)
    elev = ELEVATIONS[min(int(3 * y / plate_h), 2)]
    img = cv2.imread(str(bank / f"e{elev}_y{rng.integers(YAWS):02d}.png"), cv2.IMREAD_UNCHANGED)
    s = width(y) / TYPICAL_FOOTPRINT * rng.uniform(*SIZE_JITTER) / ppu
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (0, 0), RENDER_BLUR)
    ys, xs = np.nonzero(img[:, :, 3] > 127)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # (x, y) is where the robot stands: the middle of its box's bottom edge
    return {"source": f"mesh:{name}", "bgr": img[:, :, :3].astype(np.float32),
            "alpha": img[:, :, 3] / 255.0, "x0": int(x - img.shape[1] / 2), "y0": int(y - img.shape[0])}


def cutout_pool():
    """Real robots cut from the clips and coloured by colorize.py."""
    pool = []
    for clip in CUTOUT_CLIPS:
        index = SPRITES / "cutouts" / clip / "index.json"
        pool += [{**c, "path": index.parent / c["file"]}
                 for c in (json.loads(index.read_text()) if index.exists() else [])]
    return pool


def cutout_robot(c, gain):
    img = cv2.imread(str(c["path"]), cv2.IMREAD_UNCHANGED)
    # per-fight gain differs, so a robot is brought to the plate's
    return {"source": f"cutout:{c['robot']}:{c['paint']}:{c['clip']}:{c['frame']}", "clip": c["clip"],
            "bgr": img[:, :, :3].astype(np.float32) * gain,
            "alpha": cv2.GaussianBlur(img[:, :, 3] / 255.0, (0, 0), 1.0), "x0": c["x0"], "y0": c["y0"]}


def touches(a, b, reach=0):
    return (a[0] - reach <= b[2] and b[0] - reach <= a[2]
            and a[1] - reach <= b[3] and b[1] - reach <= a[3])


def stand(robot):
    return robot["x0"] + robot["alpha"].shape[1] / 2, robot["y0"] + robot["alpha"].shape[0]


def slide(robot, x, y, width):
    """Move a cut-out so it stands at (x, y), resized for the new row."""
    hx, hy = stand(robot)
    s = width(y) / width(hy)
    robot["bgr"] = cv2.resize(robot["bgr"], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    robot["alpha"] = cv2.resize(robot["alpha"], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    robot["x0"], robot["y0"] = int(x - robot["alpha"].shape[1] / 2), int(y - robot["alpha"].shape[0])


def paste(canvas, robot):
    h, w = robot["alpha"].shape
    x0, y0 = robot["x0"], robot["y0"]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, canvas.shape[1]), min(y0 + h, canvas.shape[0])
    alpha = np.zeros(canvas.shape[:2], np.float32)
    alpha[ya:yb, xa:xb] = robot["alpha"][ya - y0:yb - y0, xa - x0:xb - x0]
    shadow = cv2.GaussianBlur(np.roll(alpha, (int(0.10 * h), int(0.06 * w)), (0, 1)), (0, 0), 0.06 * w)
    canvas *= (1 - SHADOW * shadow)[:, :, None]
    layer = np.zeros_like(canvas)
    layer[ya:yb, xa:xb] = robot["bgr"][ya - y0:yb - y0, xa - x0:xb - x0]
    canvas[:] = canvas * (1 - alpha[:, :, None]) + layer * alpha[:, :, None]
    return alpha > 0.5


def floor_level(clip):
    grey = cv2.imread(str(REAL / clip / "plate.png"), cv2.IMREAD_GRAYSCALE)
    return float(np.median(grey[arena_mask(grey) > 0]))


def stage(clip):
    """What composing on a clip's plates needs: the plates, where a robot may
    stand, and the floor brightness that cut-outs from other fights are brought to."""
    grey = cv2.imread(str(REAL / clip / "plate.png"), cv2.IMREAD_GRAYSCALE)
    arena = arena_mask(grey)
    stood = np.zeros(grey.shape, np.uint8)
    for r in real_robots(clip):
        cv2.circle(stood, ((r["bbox"][0] + r["bbox"][2]) // 2, r["bbox"][3]), STOOD_NEAR, 255, -1)
    floor = cv2.bitwise_and(stood, cv2.erode(arena, np.ones((FLOOR_MARGIN, FLOOR_MARGIN), np.uint8)))
    return {"clip": clip, "plates": sorted((PLATES / clip).glob("plate_color_*.png")), "floor": floor,
            "points": np.argwhere(floor), "level": floor_level(clip)}


def data_yaml(folder):
    # no `path`: Ultralytics then reads `train` relative to this file. This is
    # a pool to merge into a training set, so `val` only satisfies the format.
    (folder / "data.yaml").write_text(
        "train: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in NAMES.items()))


def main(run, n, seed, start):
    rng = np.random.default_rng(seed + start)
    out = AUGMENT / "out" / run
    for kind in ("colour", "grey"):
        for d in ("images", "labels"):
            (out / kind / d).mkdir(parents=True, exist_ok=True)
        data_yaml(out / kind)
    stages = [st for st in map(stage, CLIPS) if st["plates"]]
    level = {clip: floor_level(clip) for clip in CUTOUT_CLIPS}
    width = size_model()
    pool = cutout_pool()
    if not stages or not pool:
        raise SystemExit("no coloured plates or cut-outs: run colorize.py first")

    by_robot = {}
    for c in pool:
        by_robot.setdefault(c["robot"], []).append(c)

    def pick(other_than=None):
        """A robot first, then one of its cut-outs: fight5's two robots have
        seven times the cut-outs of fight2's and would otherwise be in half the images."""
        names = [n for n, cs in by_robot.items() if cs[0]["clip"] != other_than]
        cs = by_robot[names[rng.integers(len(names))]]
        return cs[rng.integers(len(cs))]

    def new_robot(st, h):
        if rng.random() < MESH_SHARE:
            y, x = st["points"][rng.integers(len(st["points"]))]
            return mesh_robot("gigabyte", rng, width, int(x), int(y), h), None
        c = pick()
        return cutout_robot(c, st["level"] / level[c["clip"]]), c["clip"]

    def nudge(robot, st, x, y):
        """Stand a cut-out at (x, y) if that is floor and near where it was cut."""
        hx, hy = stand(robot)
        if abs(x - hx) > MOVE[0] or abs(y - hy) > MOVE[1]:
            return False
        if not (0 <= x < st["floor"].shape[1] and 0 <= y < st["floor"].shape[0] and st["floor"][int(y), int(x)]):
            return False
        slide(robot, x, y, width)
        return True

    kinds = rng.choice(["empty", "one", "two"], size=n, p=[EMPTY, ONE_ROBOT, 1 - EMPTY - ONE_ROBOT])
    items, k = [], start
    while k < start + n:
        st = stages[rng.integers(len(stages))]
        plate_path = st["plates"][rng.integers(len(st["plates"]))]
        plate = cv2.imread(str(plate_path))
        h, w = plate.shape[:2]
        # drawn once per image and kept through retries: a two-robot layout
        # fails more often than a one-robot one, and drawing again each time
        # left only half the images with two robots
        kind = kinds[k - start]
        robots = []
        if kind != "empty":
            first, first_clip = new_robot(st, h)
            if first_clip and not nudge(first, st, stand(first)[0] + rng.uniform(-MOVE[0], MOVE[0]),
                                        stand(first)[1] + rng.uniform(-MOVE[1], MOVE[1])):
                continue
            robots.append(first)
        if kind == "two":
            # the second robot always comes from another fight, so it is another robot
            c = pick(other_than=first_clip)
            second = cutout_robot(c, st["level"] / level[c["clip"]])
            x, y = stand(first)
            reach = width(y)
            d = reach * (rng.uniform(0.7, 1.3) if rng.random() < CONTACT_SHARE else rng.uniform(1.5, 5))
            t = rng.uniform(0, 2 * np.pi)
            if not nudge(second, st, x + d * np.cos(t), y + 0.6 * d * np.sin(t)):
                continue
            robots.append(second)
        # the robot nearer the camera stands lower in the frame and is drawn last
        robots.sort(key=lambda r: stand(r)[1])

        canvas = plate.astype(np.float32)
        seen = [paste(canvas, r) for r in robots]
        if len(seen) == 2:
            seen[0] &= ~seen[1]
        whole = [float((r["alpha"] > 0.5).sum()) for r in robots]
        if any(s.sum() < MIN_VISIBLE * t for s, t in zip(seen, whole)):
            continue
        grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), NOISE_GRAIN)
        grain /= grain.std()
        tint = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), NOISE_GRAIN)
        image = np.clip(canvas + (grain * NOISE_COLOUR[0])[:, :, None]
                        + tint / tint.std() * NOISE_COLOUR[1], 0, 255).astype(np.uint8)
        # the same scene as the monochrome fight camera would see it
        image_grey = np.clip(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) + grain * NOISE_GREY,
                             0, 255).astype(np.uint8)
        rows, boxes = [], []
        for s in seen:
            ys, xs = np.nonzero(s)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            boxes.append([x0, y0, x1, y1])
            rows.append(f"{NON_NEMESIS} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                        f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
        for folder, picture in (("colour", image), ("grey", image_grey)):
            cv2.imwrite(str(out / folder / "images" / f"{k:05d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / folder / "labels" / f"{k:05d}.txt").write_text("".join(rows))
        items.append({"image": f"{k:05d}.jpg", "kind": str(kind), "clip": st["clip"], "plate": plate_path.name,
                      "robots": [{"source": r["source"], "class": NAMES[NON_NEMESIS], "bbox": bx,
                                  "visible": round(float(s.sum() / t), 2)}
                                 for r, bx, s, t in zip(robots, boxes, seen, whole)]})
        k += 1

    # one part per process, so several can fill a run side by side
    (out / f"manifest_{start:05d}.json").write_text(json.dumps(
        {"run": run, "seed": seed, "start": start,
         "classes": {str(i): name for i, name in NAMES.items()}, "items": items}))
    print(f"{n} images from {start} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="number of the first image")
    args = ap.parse_args()
    main(args.run, args.n, args.seed, args.start)
