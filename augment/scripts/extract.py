import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vision"))
from detect.blobs import arena_mask  # noqa: E402

from classes import NON_NEMESIS  # noqa: E402

AUGMENT = ROOT / "augment"
REAL = AUGMENT / "data" / "real"
SAM_WEIGHTS = AUGMENT / "data" / "weights" / "sam2.1_s.pt"
FIGHTS = ROOT / "vision" / "data" / "Real Fights"
CLIPS = {
    # fight1 was never copied into this repo; it is read in place.
    "fight1": Path.home() / "Projects" / "nemesis-tracking" / "data" / "fight1.avi",
    **{f"fight{i}": FIGHTS / f"fight{i}.avi" for i in (2, 3, 4, 5)},
}

# The arena lights are off for roughly the first half of every clip (mean
# brightness 2-20 dark, 85-130 lit), and they do not simply switch on: a
# light show strobes between the two for several hundred frames first
# (fight3: 126 at frame 13500, 0.5 at 13580, 110 at 13750, 12 at 14000). So
# "lit" means the longest run of probes near the lit level, and every frame is
# still checked against the plate's brightness before it is used.
LIT_MEAN = 50
LIT_PROBE = 250
LIT_BAND = 0.3
FRAME_BAND = (0.7, 1.4)
PLATE_FRAMES = 120
# The plate is the per-pixel median, so anything that sits still for half of
# the frames it is built from gets baked in. A robot can lie disabled for most
# of a fight (fight3), but both are mobile early on, so only the first part
# of the lit segment feeds the plate.
PLATE_SHARE = 0.4

# ~3 labelled frames per second of fight; neighbours are near-duplicates.
KEEP_EVERY = 30

# Difference against the empty-arena plate, in grey levels after a 5x5 blur.
# Sensor noise at gain 9-12 stays under this.
DIFF = 28
# Brightness alone also fires on shadows and on the start squares, which are
# lit differently before the match starts. Those keep the floor's texture; a
# robot replaces it. So a pixel counts as changed only if its neighbourhood
# has also stopped correlating with the plate.
TEXTURE_WINDOW = 21
TEXTURE_MAX = 0.6

# A robot by the far wall is about 900-1500 px, so this has to sit under that
# even though it lets more debris through to be flagged.
MIN_REGION = 800
MAX_REGIONS = 4
BOX_PAD = 0.10
MIN_AREA = 600
MAX_AREA = 70000
# Share of an outline lying on changed pixels. Measured on fight3: correct
# outlines score 0.58-0.98, and the ones under 0.75 had dragged in a patch of
# floor or shadow. An outline that has drifted onto bare floor scores near 0.
MIN_ON_CHANGE = 0.55
CUTOUT_ON_CHANGE = 0.75
# Changed pixels left outside every outline, as a share of all changed pixels
# in robot-sized regions. Past this, something in view is not labelled: the
# second robot of a pair in contact, a torn-off weapon, smoke.
MAX_UNLABELLED = 0.35
# Boxes are grown by this before testing whether one outline has swallowed
# the two robots of the previous frame. A robot covers up to ~170 px between
# kept frames.
CONTACT_REACH = 60
# Share of an outline that differs from the previous kept frame. Under this,
# the thing has not moved in a third of a second: a disabled robot, debris,
# or a hazard that changed state (the saw slots in fight2). Those cannot be
# told apart automatically, so the frame goes to review instead of guessing.
MIN_MOVING = 0.15
# fight2's lifter is all arms and forks, and a fork or its shadow sometimes
# becomes a region of its own, which then passed as the second robot. A real
# second robot next to it is a tenth of its area or more; these were ~0.05.
FRAGMENT = 0.07

SHEET_COLS, SHEET_ROWS, TILE_W = 3, 4, 480

# "Two outlines" only means every robot is boxed when the fight has two robots.
# fight1 has three: two large ones and a minibot. Its accepted frames either
# left one unboxed or put one box round the two large ones in contact.
UNTRUSTED = {
    "fight1": "three robots in this fight, so two outlines do not cover them",
}
# Passed every check and still wrong when looked at at full resolution.
# frame -> what is wrong with it.
REJECTED = {
    "fight2": {23630: "the second outline is a fork or its shadow, not the other robot"},
    "fight5": {12480: "the second outline is something at the frame's bottom edge, not a robot"},
}


def lit_range(cap, total):
    probes = list(range(0, total, LIT_PROBE))
    means = []
    for i in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        means.append(float(frame.mean()) if ok else 0.0)
    bright = [m for m in means if m > LIT_MEAN]
    if not bright:
        raise SystemExit("no lit frames found")
    level = float(np.median(bright))
    best, run = (0, 0), None
    for n, m in enumerate(means + [0.0]):
        if abs(m - level) < LIT_BAND * level:
            run = n if run is None else run
        elif run is not None:
            best, run = max(best, (n - run, run)), None
    length, first = best
    return probes[first], probes[first + length - 1], level


def floor_gain(gray, plate, mask):
    return float(np.median(plate[mask > 0])) / max(float(np.median(gray[mask > 0])), 1.0)


def build_plate(cap, start, end, level):
    """Robots keep moving, so the per-pixel median over the lit segment is the
    empty arena."""
    stack = []
    for i in np.linspace(start, start + PLATE_SHARE * (end - start), PLATE_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok and abs(frame.mean() - level) < LIT_BAND * level:
            stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def save_frames(cap, start, end, plate, mask, images, limit):
    kept = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(start, end):
        if not cap.grab() or (limit and len(kept) >= limit):
            break
        if (i - start) % KEEP_EVERY:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if FRAME_BAND[0] <= 1 / floor_gain(gray, plate, mask) <= FRAME_BAND[1]:
            cv2.imwrite(str(images / f"{i:06d}.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept.append(i)
    return kept


def texture_match(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    k = (TEXTURE_WINDOW, TEXTURE_WINDOW)
    ma, mb = cv2.boxFilter(a, -1, k), cv2.boxFilter(b, -1, k)
    va = cv2.boxFilter(a * a, -1, k) - ma * ma
    vb = cv2.boxFilter(b * b, -1, k) - mb * mb
    return (cv2.boxFilter(a * b, -1, k) - ma * mb) / np.sqrt(np.maximum(va * vb, 1.0))


def changed(gray, plate, mask):
    # per-fight gain was retuned by hand, once mid-fight (fight2), so match
    # the frame's floor brightness to the plate's before differencing
    g = cv2.GaussianBlur(cv2.convertScaleAbs(gray, alpha=floor_gain(gray, plate, mask)), (5, 5), 0)
    p = cv2.GaussianBlur(plate, (5, 5), 0)
    fg = (((cv2.absdiff(g, p) > DIFF) & (texture_match(g, p) < TEXTURE_MAX)) * 255).astype(np.uint8)
    fg = cv2.bitwise_and(fg, mask)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def largest_part(m):
    # SAM sometimes adds a detached island of floor marking to a robot
    n, ids, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    return ids == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def outline_regions(gray, fg, moved, outside, sam, device):
    """One SAM outline per robot-sized changed region. A robot parked on a
    start square is less salient than the square, and a box alone made SAM
    outline the square, so each prompt adds a point deep inside the region and
    two background points on the box's far corners."""
    n, ids, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    found = sorted(((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                    if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION), reverse=True)
    extra = len(found) > MAX_REGIONS
    found = found[:MAX_REGIONS]
    if not found:
        return [], 0, 0.0, extra
    boxes, points = [], []
    for _, i in found:
        x, y, w, h = (int(v) for v in stats[i, :4])
        px, py = w * BOX_PAD, h * BOX_PAD
        box = [max(0, x - px), max(0, y - py),
               min(gray.shape[1] - 1, x + w + px), min(gray.shape[0] - 1, y + h + py)]
        depth = cv2.distanceTransform((ids == i).astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(depth.argmax()), depth.shape)
        boxes.append(box)
        points.append([[int(cx), int(cy)], [box[2] - 3, box[1] + 3], [box[0] + 3, box[3] - 3]])
    r = sam(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), device=device, verbose=False,
            bboxes=boxes, points=points, labels=[[1, 0, 0]] * len(boxes))[0]
    outlines = [] if r.masks is None else [largest_part(m) for m in r.masks.data.cpu().numpy() > 0]
    in_regions = np.isin(ids, [i for _, i in found])
    robots, covered = [], np.zeros(gray.shape, bool)
    for m in outlines:
        area = int(m.sum())
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        on_change = round(float((m & (fg > 0)).sum() / area), 2)
        if on_change < MIN_ON_CHANGE or (m & covered).sum() > 0.5 * area:
            continue
        covered |= m
        ys, xs = np.nonzero(m)
        robots.append({"mask": m, "area": area, "on_change": on_change,
                       "moving": round(float((m & (moved > 0)).sum() / area), 2),
                       # change is only looked for inside the arena, so a
                       # robot leaning out of it is outlined in part
                       "at_edge": bool((m & outside).any()),
                       "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       "mean": round(float(gray[m].mean()), 1)})
    unlabelled = round(float((in_regions & ~covered).sum() / int(in_regions.sum())), 2)
    return robots, len(found), unlabelled, extra


def touches(a, b, reach):
    return (a[0] - reach <= b[2] and b[0] - reach <= a[2]
            and a[1] - reach <= b[3] and b[1] - reach <= a[3])


def judge(clip, f, last_pair):
    """Why a frame is not trusted. An empty list means its labels are kept."""
    robots, reasons = f["robots"], []
    if not robots:
        reasons.append("no robot outlined")
    # Every fight has two robots. A frame showing one was accepted at first,
    # and on review many hid the other: small and still by the far wall, or
    # tangled with the first inside a single outline.
    if len(robots) == 1:
        reasons.append("only one robot found")
    if len(robots) > 2 or f["extra"]:
        reasons.append("more than two robot-sized things")
    if f["missed"]:
        reasons.append("a region got no clean outline")
    if f["unlabelled"] > MAX_UNLABELLED:
        reasons.append("unlabelled change")
    if any(r["moving"] < MIN_MOVING for r in robots):
        reasons.append("static object")
    if any(r["at_edge"] for r in robots):
        reasons.append("robot at the arena edge")
    if len(robots) == 2:
        small, big = sorted(robots, key=lambda r: r["area"])
        if small["area"] < FRAGMENT * big["area"] and touches(small["bbox"], big["bbox"], 15):
            reasons.append("second outline may be a piece of the first")
    # one box round two robots is a wrong label for a one-class detector
    if len(robots) == 1 and last_pair and all(
            touches(robots[0]["bbox"], b, CONTACT_REACH) for b in last_pair):
        reasons.append("robots in contact")
    if clip in UNTRUSTED:
        reasons.append("clip not trusted: " + UNTRUSTED[clip])
    if f["frame"] in REJECTED.get(clip, {}):
        reasons.append("rejected on review: " + REJECTED[clip][f["frame"]])
    return reasons


def draw(gray, robots, text, accepted):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in robots:
        c, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, c, -1, (0, 255, 255), 2)
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if accepted else (0, 0, 255), 3)
    vis = cv2.resize(vis, (TILE_W, TILE_W * gray.shape[0] // gray.shape[1]))
    cv2.putText(vis, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(vis, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def sheets(tiles, out, name):
    per = SHEET_COLS * SHEET_ROWS
    for n, i in enumerate(range(0, len(tiles), per)):
        page = tiles[i:i + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[i:i + per]))
        rows = [np.hstack(page[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        cv2.imwrite(str(out / f"{name}_{n:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def cutout(gray, r, cell=120):
    x0, y0, x1, y1 = r["bbox"]
    crop = np.where(r["mask"][y0:y1 + 1, x0:x1 + 1], gray[y0:y1 + 1, x0:x1 + 1], 96).astype(np.uint8)
    s = (cell - 8) / max(crop.shape)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    tile = np.full((cell, cell), 96, np.uint8)
    oy, ox = (cell - crop.shape[0]) // 2, (cell - crop.shape[1]) // 2
    tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return tile


def spread(tiles, n):
    return [tiles[i] for i in np.linspace(0, len(tiles) - 1, min(n, len(tiles))).astype(int)]


def outline(clip, device, limit):
    """The slow half: frames, plate and one checked SAM outline per region."""
    clock = time.perf_counter()
    out = REAL / clip
    if out.exists():
        shutil.rmtree(out)
    images, masks_dir = out / "images", out / "masks"
    images.mkdir(parents=True)
    masks_dir.mkdir()

    cap = cv2.VideoCapture(str(CLIPS[clip]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end, level = lit_range(cap, total)
    plate = build_plate(cap, start, end, level)
    cv2.imwrite(str(out / "plate.png"), plate)
    arena = arena_mask(plate)
    kept = save_frames(cap, start, end, plate, arena, images, limit)
    cap.release()
    print(f"{clip}: lit {start}-{end} of {total}, {len(kept)} frames kept", flush=True)

    outside = cv2.dilate(cv2.bitwise_not(arena), np.ones((7, 7), np.uint8)) > 0
    sam = SAM(str(SAM_WEIGHTS))
    items, before = [], None
    for n, i in enumerate(kept):
        gray = cv2.imread(str(images / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if before is None:
            before = cv2.imread(str(images / f"{kept[min(n + 1, len(kept) - 1)]:06d}.jpg"),
                                cv2.IMREAD_GRAYSCALE)
        robots, n_regions, unlabelled, extra = outline_regions(
            gray, changed(gray, plate, arena), changed(gray, before, arena), outside, sam, device)
        before = gray
        ids = np.zeros(gray.shape, np.uint8)
        for k, rb in enumerate(robots):
            ids[rb.pop("mask")] = k + 1
        cv2.imwrite(str(masks_dir / f"{i:06d}.png"), ids)
        items.append({"frame": i, "robots": robots, "missed": len(robots) < n_regions,
                      "extra": extra, "unlabelled": unlabelled})
        if len(items) % 100 == 0:
            print(f"  {len(items)}/{len(kept)}", flush=True)
    (out / "outlines.json").write_text(json.dumps({
        "clip": clip, "source": str(CLIPS[clip]), "lit": [start, end], "keep_every": KEEP_EVERY,
        "seconds": round(time.perf_counter() - clock), "items": items}))


def review(clip, max_sheets):
    """The fast half: judge every frame from outlines.json and write labels,
    proposals and sheets. Run alone after changing a rule or REJECTED."""
    out = REAL / clip
    found = json.loads((out / "outlines.json").read_text())
    labels, proposals, sheet_dir = (out / d for d in ("labels", "proposals", "sheets"))
    for d in (labels, proposals, sheet_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    frames, good, bad, cuts = [], [], [], []
    last_pair = None
    for f in found["items"]:
        i, robots = f["frame"], f["robots"]
        reasons = judge(clip, f, last_pair)
        if len(robots) == 2:
            last_pair = [r["bbox"] for r in robots]
        elif "robots in contact" in reasons:
            # stay in contact until two outlines come back
            last_pair = [robots[0]["bbox"]] * 2
        else:
            last_pair = None
        accepted = not reasons
        gray = cv2.imread(str(out / "images" / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        ids = cv2.imread(str(out / "masks" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        shown = [{**rb, "mask": ids == k + 1} for k, rb in enumerate(robots)]
        h, w = gray.shape
        # Nemesis is in none of the fight clips, so every robot here is the other class
        rows = [f"{NON_NEMESIS} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"
                for x0, y0, x1, y1 in (rb["bbox"] for rb in robots)]
        # a starting point for whoever reviews a flagged frame, not a label
        (proposals / f"{i:06d}.txt").write_text("".join(rows))
        if accepted:
            (labels / f"{i:06d}.txt").write_text("".join(rows))
            cuts += [cutout(gray, rb) for rb in shown if rb["on_change"] >= CUTOUT_ON_CHANGE]
        tile = draw(gray, shown, f"{i}" + ("" if accepted else "  " + "; ".join(reasons)), accepted)
        (good if accepted else bad).append(tile)
        frames.append({**f, "accepted": accepted, "reasons": reasons})

    # every accepted frame is on a sheet; flagged ones are sampled
    if good:
        sheets(good, sheet_dir, "accepted")
    if bad:
        sheets(spread(bad, SHEET_COLS * SHEET_ROWS * max_sheets), sheet_dir, "flagged")
    if cuts:
        cuts = spread(cuts, 96)
        cuts += [np.full_like(cuts[0], 96)] * (-len(cuts) % 12)
        cv2.imwrite(str(sheet_dir / "cutouts.jpg"),
                    np.vstack([np.hstack(cuts[r:r + 12]) for r in range(0, len(cuts), 12)]))

    counts = {}
    for f in frames:
        for reason in f["reasons"]:
            reason = reason.split(":")[0]
            counts[reason] = counts.get(reason, 0) + 1
    summary = {**{k: v for k, v in found.items() if k != "items"},
               "frames": len(frames), "accepted": len(good), "flagged": len(bad),
               "flag_reasons": counts,
               "thresholds": {"DIFF": DIFF, "TEXTURE_MAX": TEXTURE_MAX, "PLATE_SHARE": PLATE_SHARE,
                              "MIN_REGION": MIN_REGION, "MIN_AREA": MIN_AREA, "MAX_AREA": MAX_AREA,
                              "MIN_ON_CHANGE": MIN_ON_CHANGE, "CUTOUT_ON_CHANGE": CUTOUT_ON_CHANGE,
                              "MAX_UNLABELLED": MAX_UNLABELLED, "CONTACT_REACH": CONTACT_REACH,
                              "MIN_MOVING": MIN_MOVING, "FRAGMENT": FRAGMENT}}
    (out / "manifest.json").write_text(json.dumps({**summary, "items": frames}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", choices=sorted(CLIPS))
    ap.add_argument("--max-sheets", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many kept frames")
    ap.add_argument("--review", action="store_true", help="re-judge saved outlines; skip SAM")
    args = ap.parse_args()
    if not args.review:
        outline(args.clip, args.device, args.frames)
    review(args.clip, args.max_sheets)
