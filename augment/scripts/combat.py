import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

import compose
import robots
from extract import CLIPS, FRAME_BAND, MIN_MOVING, REAL, SAM_WEIGHTS, changed, floor_gain, largest_part

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision"))
from detect.blobs import arena_mask  # noqa: E402

# Robots in combat, from the frames extract.py set aside: two robots touching,
# and single robots that are stopped, flipped, wrecked or lifted.
#
# Two robots in contact are one patch of change, and extract.py asks SAM for
# one outline per patch, so it outlines both as one robot or only one of
# them. Here each pair is followed into contact from the last frame where the
# two were apart, prompting SAM with each robot's previous box. Kept frames
# are 30 apart, too far to follow anything; inside a contact stretch every
# EVERY-th frame is used.
CLIPS_USED = ("fight2", "fight3", "fight4", "fight5")
EVERY = 6
LEAD = 150      # how far back a start frame with the two apart may be (1.5 s)
TAIL = 98       # followed this far past the stretch's last kept frame (1 s)
REACH = 60      # boxes this close make a kept frame part of a contact stretch
TOUCH = 15      # outlines this close count as touching
CLEAN = 0.75    # share of an outline on changed pixels; extract.CUTOUT_ON_CHANGE
START_CLEAN = 0.55
# SAM's grown box takes in floor lettering and slots next to a robot, which
# never differ from the plate. An outline is trimmed to what lies on or near
# changed pixels; if that trims more than this share, it was not a robot's.
NEAR_CHANGE = 15
TRIMMED = 0.25
# Beyond these between two steps the track has jumped to something else.
LOST_AREA = (0.6, 1.6)
LOST_JUMP = 90
MERGED = 0.5    # two outlines sharing this much have become one
KEEP = 2        # one touching frame in every KEEP is kept, about 0.12 s apart
# A single robot this much wider than a real robot standing on the same row
# is off the floor, nearer the camera: lifted, or thrown.
LIFTED = 1.4
SAME_SPOT = 0.7

# Every pose was looked at on the review sheets; only these are used, with
# the name corrected where the automatic one was wrong. Singles: (frame, k).
# Left out, mostly: floor slots and lettering taken for a robot, debris from
# the smashed Slot Machine, and outlines that ran onto the floor.
SINGLES = {
    # Mammoth is wider than most robots on the floor, so "lifted" catches it
    # driving as well as with its arms up.
    "fight2": {(13790, 0): None, (13910, 0): None, (13970, 0): None, (14030, 0): None,
               (18620, 0): None, (28910, 0): None, (29090, 0): None, (29510, 0): "mammoth",
               (29540, 0): None, (29750, 0): None},
    "fight3": {(14500, 1): None, (16690, 1): None, (16840, 1): None, (17350, 0): "nightmare",
               (19900, 0): None, (19900, 1): None, (20110, 0): None},
    # the WHIPLASH lettering, or its striped box shape, from above
    "fight4": {(14100, 1): None, (21210, 0): "whiplash", (21690, 0): "whiplash", (22200, 0): "whiplash"},
    "fight5": {(11280, 1): "hypershock", (11430, 0): None, (12570, 0): None, (12630, 0): None,
               (13410, 1): None, (14550, 0): None, (14610, 1): None, (16740, 0): None, (19410, 0): None},
}
# Pairs, by frame: None keeps both names, else (robot of outline 0, of outline 1).
# Left out: an outline that took in the far-wall platform, a floor patch or
# shadow, a spark cloud, or most of the other robot.
PAIRS = {
    "fight2": {22220: None},
    "fight3": {**{f: None for f in (14818, 14830, 14842, 14854, 15088, 15502, 15514, 15526, 15550,
                                    16264, 16276, 16288, 16300, 16312, 16324)},
               # followed back from a frame named by features alone, which got
               # them the wrong way round: the cross-shaped one is Nightmare
               **{f: ("nightmare", "slot-machine")
                  for f in (16480, 16492, 16504, 16516, 16528, 16540, 16552, 16564)}},
    "fight4": {f: None for f in (13806, 13818, 13830, 13848, 14268, 14280, 14292, 14304, 14316, 14328,
                                 14340, 14670, 14682, 14796, 14808, 14820, 14832, 14844, 14856, 14868,
                                 14880, 14892, 14904, 14922, 14940, 15084, 15096, 15108, 15120, 15132,
                                 15144, 15156, 16056, 16068, 16080, 16092, 16104, 16116, 16128)},
    # 14862-14886: Witch Doctor thrown up onto its side by HyperShock
    "fight5": {f: None for f in (11724, 11736, 11748, 11760, 11772, 11784, 11796, 11808, 11820, 11832,
                                 11844, 12810, 12822, 12900, 12912, 12924, 12936, 12948, 12960, 13650,
                                 13662, 13674, 13686, 13698, 14220, 14280, 14286, 14292, 14298, 14304,
                                 14310, 14316, 14826, 14862, 14874, 14886, 14910, 14922, 14934, 14946,
                                 14958, 14970, 14982, 15324, 15336)},
}


def touches(a, b, reach):
    return a[0] - reach <= b[2] and b[0] - reach <= a[2] and a[1] - reach <= b[3] and b[1] - reach <= a[3]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def items(clip):
    return json.loads((REAL / clip / "manifest.json").read_text())["items"]


def stretches(clip):
    """Runs of kept frames in which the two robots touch or have merged."""
    contact = [f["frame"] for f in items(clip) if "robots in contact" in f["reasons"] or (
        len(f["robots"]) == 2 and touches(f["robots"][0]["bbox"], f["robots"][1]["bbox"], REACH))]
    runs = []
    for frame in contact:
        if runs and frame - runs[-1][-1] <= 60:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return [(r[0], r[-1]) for r in runs]


def outlines(sam, gray, boxes, centres, device):
    """Both robots outlined at once. Each prompt is its robot's last box,
    grown, with a point on it and a background point on the other robot."""
    h, w = gray.shape
    grown = [[max(0, b[0] - 0.15 * (b[2] - b[0])), max(0, b[1] - 0.15 * (b[3] - b[1])),
              min(w - 1, b[2] + 0.15 * (b[2] - b[0])), min(h - 1, b[3] + 0.15 * (b[3] - b[1]))] for b in boxes]
    points = [[list(centres[0]), list(centres[1])], [list(centres[1]), list(centres[0])]]
    r = sam(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), device=device, verbose=False,
            bboxes=grown, points=points, labels=[[1, 0], [1, 0]])[0]
    if r.masks is None or len(r.masks) != 2:
        return None
    masks = [largest_part(m) for m in r.masks.data.cpu().numpy() > 0]
    both = masks[0] & masks[1]
    if both.sum() > MERGED * min(masks[0].sum(), masks[1].sum()):
        return None
    if both.any():
        # a pixel claimed by both goes to the robot whose centre is nearer
        ys, xs = np.nonzero(both)
        nearer = np.hypot(xs - centres[0][0], ys - centres[0][1]) <= np.hypot(xs - centres[1][0], ys - centres[1][1])
        masks[0][ys[~nearer], xs[~nearer]] = False
        masks[1][ys[nearer], xs[nearer]] = False
    return masks


def tidy(m, fg):
    """An outline cut back to changed pixels and their surroundings, with the
    robot's own unchanged patches (dark on dark floor) filled back in."""
    m = largest_part(m & (cv2.dilate(fg, np.ones((NEAR_CHANGE, NEAR_CHANGE), np.uint8)) > 0))
    filled = m.astype(np.uint8)
    edges, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, edges, -1, 1, -1)
    return filled > 0


def describe(m, fg, outside):
    ys, xs = np.nonzero(m)
    return {"area": int(m.sum()), "on_change": round(float((m & (fg > 0)).sum() / max(m.sum(), 1)), 2),
            "at_edge": bool((m & outside).any()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "centre": [int(xs.mean()), int(ys.mean())]}


def follow(clip, s, step, end, cap, sam, device, plate, arena, outside, out):
    """From kept frame `s`, where the two are apart, step through the video
    towards `end`, prompting SAM with each robot's previous box."""
    d = REAL / clip
    ids = cv2.imread(str(d / "masks" / f"{s['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    boxes = [r["bbox"] for r in s["robots"]]
    centres = [[int(c) for c in np.argwhere(ids == k + 1).mean(0)[::-1]] for k in range(2)]
    areas = [r["area"] for r in s["robots"]]
    frames, touching, i = [], 0, s["frame"]
    for i in range(s["frame"] + step, end + step, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not FRAME_BAND[0] <= 1 / floor_gain(gray, plate, arena) <= FRAME_BAND[1]:
            continue
        raw = outlines(sam, gray, boxes, centres, device)
        if raw is None or min(m.sum() for m in raw) == 0:
            break
        fg = changed(gray, plate, arena)
        masks = [tidy(m, fg) for m in raw]
        if min(m.sum() for m in masks) == 0:
            break
        now = [describe(m, fg, outside) for m in masks]
        if any(not LOST_AREA[0] <= n["area"] / a <= LOST_AREA[1] for n, a in zip(now, areas)) \
                or any(np.hypot(n["centre"][0] - c[0], n["centre"][1] - c[1]) > LOST_JUMP
                       for n, c in zip(now, centres)) \
                or min(n["on_change"] for n in now) < START_CLEAN:
            break
        boxes, centres, areas = [n["bbox"] for n in now], [n["centre"] for n in now], [n["area"] for n in now]
        touch = bool((cv2.dilate(masks[0].astype(np.uint8), np.ones((2 * TOUCH + 1,) * 2, np.uint8))
                      & masks[1]).any())
        kept = [m.sum() / r.sum() for m, r in zip(masks, raw)]
        if touch and min(kept) >= 1 - TRIMMED and not any(n["at_edge"] for n in now):
            touching += 1
            if touching % KEEP == 1:
                cv2.imwrite(str(out / "images" / f"{i:06d}.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out / "masks" / f"{i:06d}.png"), (masks[0] * 1 + masks[1] * 2).astype(np.uint8))
                frames.append({"frame": i, "robots": now})
    return frames, touching, i


def track(clip, device):
    """Follow every pair into contact, forwards from the last kept frame where
    the two were apart and backwards from the first one after; keep the
    touching frames."""
    d, out = REAL / clip, REAL / clip / "combat"
    for sub in ("images", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    plate = cv2.imread(str(d / "plate.png"), cv2.IMREAD_GRAYSCALE)
    arena = arena_mask(plate)
    outside = cv2.dilate(cv2.bitwise_not(arena), np.ones((7, 7), np.uint8)) > 0
    named, groups = robots.identify(clip), robots.groups(clip)
    kept = items(clip)
    apart = [f for f in kept if len(f["robots"]) == 2 and min(r["on_change"] for r in f["robots"]) >= START_CLEAN
             and not touches(f["robots"][0]["bbox"], f["robots"][1]["bbox"], REACH)]
    sam = SAM(str(SAM_WEIGHTS))
    cap = cv2.VideoCapture(str(CLIPS[clip]))
    tracks = []
    for first, last in stretches(clip):
        before = [f for f in apart if first - LEAD <= f["frame"] <= first]
        after = [f for f in apart if last <= f["frame"] <= last + LEAD]
        starts = ([(max(before, key=lambda f: f["frame"]), EVERY, last + TAIL)] if before else []) \
            + ([(min(after, key=lambda f: f["frame"]), -EVERY, first - TAIL)] if after else [])
        for s, step, end in starts:
            gray = cv2.imread(str(d / "images" / f"{s['frame']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            ids = cv2.imread(str(d / "masks" / f"{s['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
            names = [named.get((s["frame"], k)) for k in range(2)]
            how = "accepted frame"
            if None in names:
                names = robots.name(groups, [robots.features(gray, ids == k + 1, r) for k, r in enumerate(s["robots"])])
                how = "features"
            done = {f["frame"] for t in tracks for f in t["frames"]}
            frames, touching, reached = follow(clip, s, step, end, cap, sam, device, plate, arena, outside, out)
            frames = [f for f in frames if f["frame"] not in done]
            tracks.append({"stretch": [first, last], "start": s["frame"], "direction": "forward" if step > 0 else "back",
                           "names": names, "named_from": how, "followed_to": reached,
                           "touching": touching, "frames": frames})
            print(f"{clip} {first}-{last} {'forward' if step > 0 else 'back'} from {s['frame']} "
                  f"({' / '.join(names)}, {how}): {touching} touching steps, {len(frames)} kept", flush=True)
    cap.release()
    (out / "tracks.json").write_text(json.dumps({"clip": clip, "every": EVERY, "keep": KEEP, "tracks": tracks}))


def singles(clip, frames=None):
    """Clean outlines of one robot from set-aside frames that show combat:
    still (stopped, flipped or wrecked) or wider than a robot on the floor at
    that row (lifted). Frames of a contact stretch are left to the pairs.
    `frames` limits the search, so that only those frames need to be on disk."""
    d = REAL / clip
    width, groups = compose.size_model(), robots.groups(clip)
    spans = stretches(clip)
    found = []
    for f in items(clip):
        if f["accepted"] or any(a - LEAD <= f["frame"] <= b + TAIL for a, b in spans) \
                or (frames is not None and f["frame"] not in frames):
            continue
        kinds = {}
        for k, r in enumerate(f["robots"]):
            if r["on_change"] < CLEAN or r["at_edge"]:
                continue
            if r["moving"] < MIN_MOVING:
                kinds[k] = "still"
            elif (r["bbox"][2] - r["bbox"][0]) / width(r["bbox"][3]) > LIFTED:
                kinds[k] = "lifted"
        if not kinds:
            continue
        gray = cv2.imread(str(d / "images" / f"{f['frame']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        ids = cv2.imread(str(d / "masks" / f"{f['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
        # two outlines are named together, so that they come out different robots
        both = f["robots"][:2] if len(f["robots"]) >= 2 else f["robots"]
        names = robots.name(groups, [robots.features(gray, ids == k + 1, r) for k, r in enumerate(both)])
        for k, kind in kinds.items():
            r = f["robots"][k]
            if any(p["kind"] == kind and iou(p["bbox"], r["bbox"]) > SAME_SPOT for p in found):
                continue
            found.append({"clip": clip, "frame": f["frame"], "k": k, "bbox": r["bbox"], "kind": kind,
                          "robot": names[k] if k < len(names) else names[0], "dir": str(d)})
    return found


def pairs(clip):
    path = REAL / clip / "combat" / "tracks.json"
    if not path.exists():
        return []
    found = []
    for t in json.loads(path.read_text())["tracks"]:
        for f in t["frames"]:
            for k, r in enumerate(f["robots"]):
                found.append({"clip": clip, "frame": f["frame"], "k": k, "bbox": r["bbox"], "kind": "pair",
                              "robot": t["names"][k], "dir": str(REAL / clip / "combat")})
    return found


def poses(clip, everything=False):
    """The combat poses of a clip that passed review, named. `everything`
    gives every candidate instead, for the review sheets."""
    if everything:
        return pairs(clip) + singles(clip)
    found = pairs(clip) + singles(clip, {frame for frame, _ in SINGLES.get(clip, {})})
    out = []
    for p in found:
        if p["kind"] == "pair":
            if p["frame"] not in PAIRS.get(clip, {}):
                continue
            names = PAIRS[clip][p["frame"]]
            out.append({**p, "robot": names[p["k"]] if names else p["robot"]})
        elif (p["frame"], p["k"]) in SINGLES.get(clip, {}):
            out.append({**p, "robot": SINGLES[clip][(p["frame"], p["k"])] or p["robot"]})
    return out


def review(clip):
    """Sheets of every pose, named, for a look before anything is coloured."""
    out = REAL / clip / "sheets"
    tiles = []
    approved = {(p["frame"], p["k"]) for p in poses(clip)}
    for p in poses(clip, everything=True):
        d = Path(p["dir"])
        gray = cv2.imread(str(d / "images" / f"{p['frame']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        ids = cv2.imread(str(d / "masks" / f"{p['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        c, _ = cv2.findContours((ids == p["k"] + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, c, -1, (0, 255, 255) if p["k"] == 0 else (255, 0, 255), 2)
        x0, y0, x1, y1 = p["bbox"]
        pad = max(x1 - x0, y1 - y0) // 2 + 20
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        crop = vis[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
        tile = cv2.resize(crop, (220, 220))
        label = f"{p['frame']} k{p['k']} {p['kind']} {p['robot'][:12]}"
        cv2.rectangle(tile, (0, 0), (220, 16), (0, 90, 0) if (p["frame"], p["k"]) in approved else (0, 0, 0), -1)
        cv2.putText(tile, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(tile)
    per = 6 * 5
    for n, i in enumerate(range(0, len(tiles), per)):
        page = tiles[i:i + per] + [np.zeros_like(tiles[0])] * (-len(tiles[i:i + per]) % 6)
        cv2.imwrite(str(out / f"combat_{n:02d}.jpg"),
                    np.vstack([np.hstack(page[r:r + 6]) for r in range(0, len(page), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    kinds = {}
    for t in poses(clip):
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"{clip}: {len(tiles)} candidates, approved {kinds} -> {out}/combat_*.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("track", "review"))
    ap.add_argument("clips", nargs="+", choices=CLIPS_USED)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    for clip in args.clips:
        if args.what == "track":
            track(clip, args.device)
        review(clip)
