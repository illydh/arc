import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vision"))
from detect.blobs import arena_mask  # noqa: E402

import colorize  # noqa: E402
import compose  # noqa: E402
import robots  # noqa: E402
from classes import NAMES, NON_NEMESIS  # noqa: E402

AUGMENT = ROOT / "augment"
DETECTOR = AUGMENT / "data" / "detector" / "finetune_color.pt"

# Thresholds. Stated up front so a run is judged against them rather than
# post-hoc; see augment/mds/context.md.
# The plate every other plate is compared with: looked at by eye and matched
# to the broadcast screenshot.
APPROVED_PLATE = compose.PLATES / "fight4" / "plate_color_3.png"
PAINTED = 14            # CIELAB chroma above which a floor pixel counts as painted
PLATE_REACH = 17        # px; the camera sits this far apart between fights
PLATE_PAINT_IOU = 0.5   # paint in the same places as the approved plate
PLATE_COLOUR_TOL = 12   # CIELAB (a, b) distance from the broadcast paint
# Share of a start square that is not its own colour. Empty squares measured
# 0.04 to 0.06; with a robot copied onto them from the reference, 0.13 to 0.18.
SQUARE_GAPS = 0.09
# A full run colours 12 plates. Those that fail the checks are set aside by
# `gate`; with fewer than this left, the set leans on too few backgrounds.
PLATES_MIN = 6
SET_ASIDE = "rejected_"
MIX_TOL = 0.04          # share of two-robot / one-robot / empty images
SIZE_RATIO = (0.6, 1.6)  # median box width against a real robot's at the same row
GRID = (6, 3)
COVERAGE_MIN = 0.6      # share of the floor grid with a robot in it; runs of 500+
DUPLICATE_MAX = 0.01
# The match-ups filmed. Images pair robots from different fights, so only
# Kraken against Tazbot (fight1; Tazbot's cut-outs are fight4's) recurs, by chance.
FILMED = {frozenset(v[1:]) for v in robots.CLIPS.values()} | {frozenset(("kraken", "tazbot"))}
AS_FILMED_MAX = 0.05    # share of two-robot images whose two robots met in the footage
DETECTOR_SAMPLE = 300
SHEETS, PER_SHEET = 4, 12


def painted(plate, arena):
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:] - 128
    return ab, (np.hypot(ab[:, :, 0], ab[:, :, 1]) > PAINTED) & (arena > 0)


def square_gaps(of):
    """An empty start square is solid paint. The full model copied the robots
    waiting on the squares in the reference picture onto the plates, where
    each is a gap in its square."""
    gaps = {}
    for name, end in (("red", np.argmin), ("blue", np.argmax)):   # red square far left, blue far right
        n, ids, stats, centres = cv2.connectedComponentsWithStats(of[name].astype(np.uint8))
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 5000]
        if not big:
            gaps[name] = None
            continue
        square = (ids == big[end([centres[i][0] for i in big])]).astype(np.uint8)
        outline = cv2.findContours(square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        hull = cv2.fillConvexPoly(np.zeros_like(square), cv2.convexHull(outline), 1)
        gaps[name] = round(1 - float(square.sum() / hull.sum()), 3)
    return gaps


def check_plate(path, approved):
    plate = cv2.imread(str(path))
    arena = arena_mask(cv2.imread(str(compose.REAL / path.parent.name / "plate.png"), cv2.IMREAD_GRAYSCALE))
    ab, paint = painted(plate, arena)
    grow = np.ones((PLATE_REACH, PLATE_REACH), np.uint8)
    a, b = (cv2.dilate(m.astype(np.uint8), grow) > 0 for m in (paint, approved))
    hue = np.degrees(np.arctan2(ab[:, :, 1], ab[:, :, 0])) % 360
    of = {name: paint & (((hue - low) % 360) <= ((high - low) % 360))
          for name, ((low, high), _) in colorize.PAINT.items()}
    off = {name: round(float(np.hypot(*(np.median(ab[m], 0) - colorize.PAINT[name][1]))), 1) if m.sum() > 500 else None
           for name, m in of.items()}
    off["floor"] = round(float(np.hypot(*(np.median(ab[(arena > 0) & ~paint], 0) - colorize.FLOOR))), 1)
    iou = round(float((a & b).sum() / (a | b).sum()), 2)
    gaps = square_gaps(of)
    ok = (iou >= PLATE_PAINT_IOU and all(v is not None and v <= PLATE_COLOUR_TOL for v in off.values())
          and all(v is not None and v <= SQUARE_GAPS for v in gaps.values()))
    return {"plate": f"{path.parent.name}/{path.name}", "paint_iou": iou, "colour_off": off,
            "square_gaps": gaps, "ok": bool(ok)}


def approved_paint():
    arena = arena_mask(cv2.imread(str(compose.REAL / "fight4" / "plate.png"), cv2.IMREAD_GRAYSCALE))
    return arena, painted(cv2.imread(str(APPROVED_PLATE)), arena)[1]


def gate():
    """Set aside every coloured plate that fails the plate checks, before
    anything is composed on it. compose.py picks plates up by name, so a
    renamed one is simply not used. The full model's first plates had a blue
    floor and robots copied onto the start squares (context.md)."""
    approved = approved_paint()[1]
    for path in sorted(compose.PLATES.glob("*/plate_color_*.png")):
        result = check_plate(path, approved)
        if not result["ok"]:
            path.rename(path.with_name(SET_ASIDE + path.name))
        print("kept     " if result["ok"] else "set aside", json.dumps(result))


def label_errors(run, items):
    errors = []
    for folder in ("colour", "grey"):
        for it in items:
            image = run / folder / "images" / it["image"]
            label = run / folder / "labels" / it["image"].replace(".jpg", ".txt")
            if not image.exists() or not label.exists():
                errors.append(f"{folder}/{it['image']}: image or label file missing")
                continue
            rows = [r.split() for r in label.read_text().splitlines()]
            if len(rows) != len(it["robots"]):
                errors.append(f"{folder}/{it['image']}: {len(rows)} boxes, {len(it['robots'])} robots")
            for r in rows:
                c, (x, y, w, h) = int(r[0]), map(float, r[1:])
                if c != NON_NEMESIS or not (0 <= x - w / 2 and x + w / 2 <= 1 and 0 <= y - h / 2 and y + h / 2 <= 1):
                    errors.append(f"{folder}/{it['image']}: class {c} or box outside the frame")
    return errors


def loads_in_ultralytics(run):
    from ultralytics.data.utils import check_det_dataset
    out = {}
    for folder in ("colour", "grey"):
        try:
            data = check_det_dataset(str(run / folder / "data.yaml"))
            out[folder] = data["names"] == NAMES
        except Exception as e:  # the check is whether the user's trainer would accept it
            out[folder] = f"{type(e).__name__}: {e}"
    return out


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detector_recall(run, items):
    """How many generated robots the detector as provided already finds. Not a
    pass mark: it shows whether they read as robots, and which ones are new to it."""
    if not DETECTOR.exists():
        return None
    from ultralytics import YOLO
    model = YOLO(str(DETECTOR))
    picked = [items[i] for i in np.linspace(0, len(items) - 1, min(DETECTOR_SAMPLE, len(items))).astype(int)]
    found = total = extra = 0
    by_source, by_kind = {}, {}
    for it in picked:
        boxes = model.predict(str(run / "colour" / "images" / it["image"]), imgsz=640, conf=0.25,
                              verbose=False)[0].boxes.xyxy.tolist()
        used = set()
        for r in it["robots"]:
            best = max(((iou(r["bbox"], b), k) for k, b in enumerate(boxes) if k not in used), default=(0, -1))
            hit = best[0] >= 0.5
            if hit:
                used.add(best[1])
            kind, name = r["source"].split(":")[:2]
            for table, key in ((by_source, name), (by_kind, kind)):
                table.setdefault(key, [0, 0])
                table[key][0] += hit
                table[key][1] += 1
            found += hit
            total += 1
        extra += len(boxes) - len(used)
    return {"images": len(picked), "robots": total, "found": found, "extra_boxes": extra,
            "recall": round(found / max(total, 1), 2),
            "recall_by_robot": {k: round(a / b, 2) for k, (a, b) in sorted(by_source.items())},
            "recall_by_source": {k: round(a / b, 2) for k, (a, b) in sorted(by_kind.items())}}


def sheets(run, items, out, name="sheet"):
    picked = [items[i] for i in np.linspace(0, len(items) - 1, min(SHEETS * PER_SHEET, len(items))).astype(int)]
    tiles = []
    for it in picked:
        img = cv2.imread(str(run / "colour" / "images" / it["image"]))
        for r in it["robots"]:
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        tiles.append(cv2.resize(img, (480, 480 * img.shape[0] // img.shape[1])))
    for n in range(0, len(tiles), PER_SHEET):
        page = tiles[n:n + PER_SHEET] + [np.zeros_like(tiles[0])] * (-len(tiles[n:n + PER_SHEET]) % 3)
        cv2.imwrite(str(out / f"{name}_{n // PER_SHEET:02d}.jpg"),
                    np.vstack([np.hstack(page[r:r + 3]) for r in range(0, len(page), 3)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(run):
    out = run / "report"
    out.mkdir(exist_ok=True)
    items = sorted((it for part in run.glob("manifest_*.json") for it in json.loads(part.read_text())["items"]),
                   key=lambda it: it["image"])
    n = len(items)
    soft, hard = [], []

    errors = label_errors(run, items)
    loads = loads_in_ultralytics(run)
    if errors:
        hard.append(f"{len(errors)} label or file errors")
    if any(v is not True for v in loads.values()):
        hard.append("a folder does not load in Ultralytics")

    # a combat trial is judged on its own mix
    combat = all(it["kind"] in compose.COMBAT_KINDS for it in items)
    target = dict(compose.COMBAT_KINDS) if combat else {
        "two": 1 - compose.EMPTY - compose.ONE_ROBOT, "one": compose.ONE_ROBOT, "empty": compose.EMPTY}
    kinds = {k: sum(it["kind"] == k for it in items) / n for k in target}
    for k, share in kinds.items():
        # a small run strays from the target by chance alone
        if abs(share - target[k]) > max(MIX_TOL, 3 * np.sqrt(target[k] * (1 - target[k]) / n)):
            soft.append(f"share of '{k}' images is {share:.2f}, target {target[k]:.2f}")

    approved_arena, approved = approved_paint()
    used = sorted({(it["clip"], it["plate"]) for it in items})
    plates = [check_plate(compose.PLATES / clip / name, approved) for clip, name in used]
    for p in plates:
        if not p["ok"]:
            soft.append(f"plate {p['plate']}: paint overlap {p['paint_iou']}, colour off by {p['colour_off']}, "
                        f"gaps in the start squares {p['square_gaps']}")
    set_aside = [check_plate(path, approved) for path in sorted(compose.PLATES.glob(f"*/{SET_ASIDE}plate_color_*.png"))]
    if n >= 500 and len(used) < PLATES_MIN:
        soft.append(f"only {len(used)} plates passed the checks ({len(set_aside)} set aside)")
    cv2.imwrite(str(out / "plates.jpg"), np.vstack([
        cv2.resize(cv2.imread(str(compose.PLATES / clip / name)), (720, 381)) for clip, name in used]),
        [cv2.IMWRITE_JPEG_QUALITY, 88])

    width = compose.size_model()
    robots = [r for it in items for r in it["robots"] if r["visible"] > 0.9]
    ratio = float(np.median([(r["bbox"][2] - r["bbox"][0]) / width(r["bbox"][3]) for r in robots]))
    if not SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]:
        soft.append(f"robots are {ratio:.2f} times the width of real ones at the same row")
    ys, xs = np.nonzero(approved_arena)
    cells = {(int((0.5 * (r["bbox"][0] + r["bbox"][2]) - xs.min()) / (np.ptp(xs) + 1) * GRID[0]),
              int((r["bbox"][3] - ys.min()) / (np.ptp(ys) + 1) * GRID[1])) for r in robots}
    coverage = len({c for c in cells if 0 <= c[0] < GRID[0] and 0 <= c[1] < GRID[1]}) / (GRID[0] * GRID[1])
    if n >= 500 and coverage < COVERAGE_MIN:
        soft.append(f"robots reach only {coverage:.2f} of the floor grid")

    seen, repeats = set(), 0
    for it in items:
        key = (it["plate"], tuple((r["source"], tuple(v // 8 for v in r["bbox"])) for r in it["robots"]))
        repeats += key in seen and bool(it["robots"])
        seen.add(key)
    if repeats / n > DUPLICATE_MAX:
        soft.append(f"{repeats} near-duplicate images")

    sources = {}
    for r in (r for it in items for r in it["robots"]):
        parts = r["source"].split(":")
        name = (":".join(parts[1:3]) if parts[0] == "cutout" else r["source"] if parts[0] in ("mesh", "new")
                else ":".join(parts[:3]))
        sources[name] = sources.get(name, 0) + 1
    two = [it for it in items if len(it["robots"]) == 2]
    contact = sum(compose.touches(*[r["bbox"] for r in it["robots"]]) for it in two) / max(len(two), 1)
    match_ups = [frozenset(r["source"].split(":")[1] for r in it["robots"]) for it in two]
    as_filmed = sum(m in FILMED for m in match_ups) / max(len(two), 1)
    if as_filmed > AS_FILMED_MAX:
        soft.append(f"{as_filmed:.2f} of two-robot images are a match-up from the footage")

    sheets(run, items, out)
    if combat:
        for kind in compose.COMBAT_KINDS:
            sheets(run, [it for it in items if it["kind"] == kind], out, f"sheet_{kind}")
    verdict = "NO-GO" if hard else "MARGINAL" if soft else "GO"
    report = {"run": run.name, "verdict": verdict, "failed": hard, "warnings": soft, "images": n,
              "kinds": {k: round(v, 3) for k, v in kinds.items()},
              "two_robot_images_with_boxes_touching": round(float(contact), 2),
              "match_ups": len(set(match_ups)), "two_robot_images_as_filmed": round(as_filmed, 3),
              "label_errors": errors[:20], "loads_in_ultralytics": loads, "plates": plates,
              "plates_set_aside": set_aside,
              "width_against_real": round(ratio, 2), "floor_coverage": round(coverage, 2),
              "near_duplicates": repeats, "robots_by_source": dict(sorted(sources.items())),
              "detector_as_provided": detector_recall(run, items)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate()
    else:
        main(Path(sys.argv[1]).resolve())
