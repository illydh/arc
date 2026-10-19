import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "augment" / "data" / "real"

# The fight clips were recorded at BattleBots: Destruct-A-Thon, whose regulars
# are a fixed roster of "ShowBots". Each robot below was matched to that roster
# from lettering in the footage and the roster pictures on the public
# BattleBots wiki (see mds/context.md for the evidence and confidence).
# `livery` is the wiki picture put into words; it is all the image model is
# given. No wiki picture is downloaded or used.
LIVERY = {
    # Told only "black steel tubes", the model painted the frame red.
    "mammoth": "an open frame of thin steel tubes painted matt black all over, with small "
               "black wheels, and no red, orange or any other colored parts",
    "kraken": "a body covered in green reptile scales, a red jaw lined with white teeth, "
              "black tyres on white wheel hubs and a white name plate",
    # fight2's small robot is Chopper or OverKill; not settled. Both are bare
    # metal on dark tyres, so one description serves.
    "chopper-or-overkill": "a bare silver metal body with scratches, black rubber tyres on "
                           "white wheel hubs",
    # Told only "black steel frame", the full model painted 7 of 36 maroon,
    # teal or brown in run1.
    "nightmare": "a matt black steel frame with a large bare silver metal disc and black wheels, "
                 "and no red, maroon, brown, teal or any other colored parts",
    "slot-machine": "a black casino slot machine cabinet with a small lit red and gold front "
                    "panel, standing on a pink base with black wheels",
    "whiplash": "a black body with neon yellow front wedges and neon yellow wheel hubs, black "
                "tyres and small white lettering",
    "tazbot": "a bright red body and red wheels, with white spikes around the base and a "
              "bare silver metal arm",
    "hypershock": "a bright yellow body covered in pink-red and black spots, black tyres on "
                  "yellow hubs and yellow front forks",
    # Medium confidence: Malice is not excluded.
    "witch-doctor": "lime green side armour with black and white bone markings, a dark purple "
                    "top, and black tyres on lime green hubs",
}

# Roster robots with no footage, drawn by the image model for the combat
# trial. From the public wiki's pages "<name> (ShowBot)", read and put into
# words; no picture was downloaded or given to the model. Chopper and OverKill
# are both here because fight2's robot is one of them and which is not known.
NEW_ROBOTS = {
    "diablo": "a low tracked robot with a red and black body, a stack of eight silver spinning "
              "disks across its front between two small red wedges, a curved red arm with two "
              "prongs rising over its back, and black rubber tracks on both sides",
    # told "saw blades used as wheels", the model drew ordinary tyres
    "ginsu": "a large box-shaped robot with thick pink armour and clear plastic top plates; "
             "instead of wheels it has four large toothed silver circular saw blades, one at each "
             "corner, and no tyres",
    "malice": "a low, wide robot with red and black top armour on a black chassis, a wide "
              "horizontal silver spinning drum across its front, and large black tyres on both sides",
    "chopper": "a two-wheeled robot with a steep black front wedge, a cage of silver metal tubes "
               "over its body, a large silver double-bladed axe on an arm over the top, and "
               "bright red tyres",
    "overkill": "a two-wheeled black wedge-shaped robot with a huge serrated silver knife blade "
                "on an arm over the top, and grey tyres on white hubs",
}

# How to tell a clip's two robots apart once their outlines are split into two
# groups (see `identify`): the feature that separates them and which robot has
# more of it. Chosen by looking at a sheet of each group.
CLIPS = {
    "fight2": ("area", "mammoth", "chopper-or-overkill"),
    "fight3": ("tall", "slot-machine", "nightmare"),
    "fight4": ("bright", "tazbot", "whiplash"),
    "fight5": ("mean", "hypershock", "witch-doctor"),
}
FEATURES = ("mean", "spread", "bright", "dark", "tall", "area")
# Every accepted frame was looked at with its two robots named. These came out
# the wrong way round: Whiplash seen from above does not look like Whiplash
# seen from the side. In the last five the other robot is a small V-shaped
# thing by the far wall, taken to be Tazbot because Whiplash is the big one.
SWAPPED = {"fight4": {15000, 15030, 18840, 19080, 19110, 19140}}
# Neither outline can be named: wreckage, or both robots too small to tell.
UNNAMED = {"fight3": {16600, 16630}, "fight4": {18660}}
# Kraken is only in fight1, which has three robots, so its frames are not
# labelled and nothing there is named automatically. A cut-out needs a clean
# outline, not a complete frame: these were picked by eye from the outlines
# that are full-size, textured and free of flame. (frame, outline number).
HAND_PICKED = {"fight1": {"kraken": [
    (23840, 0), (23900, 0), (23930, 0), (23960, 0), (23990, 0), (24020, 0), (24800, 0),
    (26060, 0), (26180, 0), (26210, 0), (30440, 0), (30470, 0), (31190, 0), (36710, 0)]}}
# A cut-out is only as good as its outline; see extract.py.
CUTOUT_ON_CHANGE = 0.75


def features(gray, mask, robot):
    x0, y0, x1, y1 = robot["bbox"]
    inside = gray[mask]
    # a robot's outline shrinks toward the far wall, so its area is taken
    # relative to the row it stands on
    return [inside.mean(), inside.std(), (inside > 180).mean(), (inside < 60).mean(),
            (y1 - y0) / max(x1 - x0, 1), np.log(robot["area"]) - 2 * np.log(max(y1, 60))]


def groups(clip):
    """The two groups a clip's accepted outlines split into.

    The two robots of a frame are always different robots, so each frame can
    only be read one of two ways. Two groups are grown under that rule: far
    steadier than clustering the outlines one by one."""
    d = REAL / clip
    frames = [f for f in json.loads((d / "manifest.json").read_text())["items"] if f["accepted"]]
    feats = []
    for f in frames:
        gray = cv2.imread(str(d / "images" / f"{f['frame']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        ids = cv2.imread(str(d / "masks" / f"{f['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
        feats.append([features(gray, ids == k + 1, r) for k, r in enumerate(f["robots"])])
    x = np.array(feats)
    mean, std = x.mean((0, 1)), x.std((0, 1)) + 1e-9
    x = (x - mean) / std

    # a robot by the far wall is a few hundred blurred pixels and its features
    # mean little, so it counts for less and its partner decides the frame
    weight = np.array([[min(1.0, r["area"] / 4000) for r in f["robots"]] for f in frames])

    key, more, less = CLIPS[clip]
    col = FEATURES.index(key)
    n = np.arange(len(frames))
    has_more = (x[:, 1, col] > x[:, 0, col]).astype(int)   # which of the frame's two has more of it
    for _ in range(20):
        ca = np.average(x[n, has_more], axis=0, weights=weight[n, has_more])
        cb = np.average(x[n, 1 - has_more], axis=0, weights=weight[n, 1 - has_more])
        cost = [weight[:, k] * ((x[:, k] - ca) ** 2).sum(1)
                + weight[:, 1 - k] * ((x[:, 1 - k] - cb) ** 2).sum(1) for k in (0, 1)]
        new = (cost[1] < cost[0]).astype(int)
        if (new == has_more).all():
            break
        has_more = new
    if ca[col] < cb[col]:
        raise SystemExit(f"{clip}: the groups no longer differ in '{key}' the way CLIPS says")
    return {"frames": frames, "has_more": has_more, "mean": mean, "std": std,
            "centres": {more: ca, less: cb}}


def identify(clip):
    """{(frame, k): robot} for the robots of a clip's accepted frames."""
    g = groups(clip)
    more, less = g["centres"]
    flip = np.array([f["frame"] in SWAPPED.get(clip, ()) for f in g["frames"]])
    has_more = np.where(flip, 1 - g["has_more"], g["has_more"])
    return {(f["frame"], k): (more if k == has_more[i] else less)
            for i, f in enumerate(g["frames"]) for k in range(2)
            if f["frame"] not in UNNAMED.get(clip, ())}


def name(g, feats):
    """Names for the outlines of one frame of other footage, from a clip's
    groups: the nearer group, and for two outlines the two different robots."""
    z = [(np.array(f) - g["mean"]) / g["std"] for f in feats]
    (a, ca), (b, cb) = g["centres"].items()
    far = lambda v, c: float(((v - c) ** 2).sum())
    if len(z) == 1:
        return [a if far(z[0], ca) <= far(z[0], cb) else b]
    return [a, b] if far(z[0], ca) + far(z[1], cb) <= far(z[0], cb) + far(z[1], ca) else [b, a]


def cutouts(clip):
    """The named robots of a clip whose outlines are clean enough to cut out."""
    frames = json.loads((REAL / clip / "manifest.json").read_text())["items"]
    if clip in HAND_PICKED:
        by_frame = {f["frame"]: f for f in frames}
        return [{"clip": clip, "frame": frame, "k": k, "bbox": by_frame[frame]["robots"][k]["bbox"],
                 "robot": robot}
                for robot, picked in HAND_PICKED[clip].items() for frame, k in picked]
    named = identify(clip)
    return [{"clip": clip, "frame": f["frame"], "k": k, "bbox": r["bbox"],
             "robot": named[(f["frame"], k)]}
            for f in frames if f["accepted"] for k, r in enumerate(f["robots"])
            if r["on_change"] >= CUTOUT_ON_CHANGE and (f["frame"], k) in named]
