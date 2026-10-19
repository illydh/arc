import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vision"))
from detect.blobs import arena_mask  # noqa: E402

import robots  # noqa: E402

AUGMENT = ROOT / "augment"
REAL = AUGMENT / "data" / "real"
PLATES = AUGMENT / "data" / "plates"
CUTOUTS = AUGMENT / "data" / "sprites" / "cutouts"
POSES = AUGMENT / "data" / "sprites" / "combat"

# FLUX.2 klein 4B. On a CUDA machine (CARC) it runs at full precision through
# diffusers and stays loaded between calls. Its two large parts (7.8 + 8.0 GB)
# do not fit this 16 GB Mac together, so here a 4-bit build (4.6 GB) runs
# through mflux, which lives in its own venv because it pins its own stack.
# The prompts were tuned on the 4-bit build: look at what the full one paints.
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
MFLUX = AUGMENT / "mlx" / ".venv" / "bin" / "mflux-generate-flux2-edit"
MFLUX_T2I = AUGMENT / "mlx" / ".venv" / "bin" / "mflux-generate-flux2"
MFLUX_MODEL = "RunPod/FLUX.2-klein-4B-mflux-4bit"
MFLUX_BASE = "flux2-klein-4b"
_klein = None

# The arena from above, in colour: a crop of a screenshot of public broadcast
# footage that the user supplied (2026-10-04), in the same orientation as the
# fight camera. It replaced a ground-level phone still, which could not say
# which start square is red or that the centre logo is painted at all.
REFERENCE = AUGMENT / "data" / "refs" / "arena_broadcast_1.jpg"
REFERENCE_SIZE = (768, 272)
# A robot waits on each start square in the screenshot. The 4-bit build left
# them there; the full model copied both onto the plates. Each is covered
# with its square's colour before the model sees the picture.
# (x0, y0, x1, y1) in the screenshot, each well inside its square.
REFERENCE_ROBOTS = ((330, 232, 495, 322), (1538, 245, 1722, 352))
# klein wants multiples of 16. The plate is 1440x762; colour is low-frequency,
# so it is generated small and scaled up, and detail comes from the real plate.
# One reference at this size takes about 3 minutes here and peaks at 12.3 GB
# of this machine's 16; two references at 1024 px took 8 minutes.
MODEL_SIZE = (768, 416)
# Told only which colours the arena has, the model painted a large red field
# across bare floor and made both start squares blue. It has to be told which
# thing gets which colour and that the rest of the floor gets none.
PLATE_PROMPT = (
    "Colorize image 1, a black-and-white photo of a combat robot arena seen from a high "
    "camera. Image 2 is a color photo of the same arena from a similar viewpoint: copy its "
    "colors onto image 1. The floor is dark charcoal grey. The large square at the left edge "
    "is red and the large square at the right edge is blue. In the center, the left letter B "
    "panel is red and the right letter B panel is blue, and the BATTLEBOTS lettering below "
    "them is white. The thin outlines around the small floor slots are yellow. The bumpers "
    "along the walls are yellow and black. Do not paint any other part of the floor. Keep "
    "every line, marking and texture of image 1 exactly where it is."
)
# Detail finer than this many pixels comes from the real plate. Coarser
# brightness is half the model's: the camera is monochrome and red-sensitive,
# so the plate shows both start squares equally bright, and keeping its
# brightness outright turns the red one pink.
DETAIL = 15.0
# The arena's paint as the broadcast shows it under fight lighting, CIELAB
# (a, b), beside the range of hues in degrees that counts as that paint in the
# model's picture. Whatever the model lays down in a range is turned to the
# measured hue and scaled to the measured strength, so a run that comes out
# too strong or off-hue (one made the red twice too strong and the blue cyan)
# lands on the same colours. The phone stills, shot under work lights, read
# much duller: red (27, 28), blue (0, -10). Fights happen under these lights.
PAINT = {"red": ((345, 70), (51, 39)), "yellow": ((75, 125), (-2, 47)), "blue": ((180, 300), (-7, -29))}
FLOOR = (3, -5)
# Paint is strong colour that is also far from the floor's own colour. The
# full model paints the whole floor navy; judged on strength alone, that floor
# counted as blue paint, outweighed the two blue squares, and all three were
# scaled up together.
PAINT_CHROMA = 12
# The model also tints what lies outside the floor: one run turned the whole
# left wall bright red. Out there colour is kept faint.
OUTSIDE = 0.25

# Cut-outs are coloured nine to a sheet, one model run for all nine. A sheet
# holds one robot and the model is told that robot's real livery in words.
CELL, GRID = 256, 3
LIVERY_PROMPT = (
    "Colorize this black-and-white contact sheet. It shows the same combat robot "
    "photographed nine times from above, on a plain grey background. The robot has {livery}. "
    "Paint every one of the nine the same way. Keep each one's shape, position and details "
    "exactly as they are, and keep the background plain grey."
)
# Guests and try-out robots also fight at the venue, so every fifth sheet
# gets paint of the model's own choosing instead.
INVENTED_EVERY = 5
INVENTED_PROMPT = (
    "Colorize this black-and-white contact sheet. It shows nine combat robots photographed "
    "from above, each on a plain grey background. Give each robot a different realistic look: "
    "bare aluminium and steel with scratches, and painted panels in one color such as red, "
    "orange, yellow, green, blue, white or black. Tyres are black rubber. Keep each robot's "
    "shape, position and details exactly as they are, and keep the background plain grey."
)
# Robots with no footage are drawn whole by the model, nine views to a sheet,
# and cut out of it. The camera's height and angle are asked for in words.
NEWBOTS = AUGMENT / "data" / "sprites" / "newbots"
NEWBOT_PROMPT = (
    "Photograph of nine copies of the same heavy BattleBots fighting robot, a real metal machine "
    "built for an arena, standing directly on a plain, flat, light grey floor in three rows of "
    "three, well apart from each other. Every copy is seen from a high camera looking down at it "
    "at about 50 degrees, each turned to face a different direction. The robot is {description}. "
    "It is a low, heavy vehicle that drives on the floor, with no legs, not a humanoid and not a "
    "toy. Realistic photograph, sharp focus, even overhead lighting. No mats, platforms or "
    "squares under the robots, no text, no people, nothing else on the floor."
)
# The model does not keep to a grid, so robots are found wherever they stand:
# anything that differs from the smooth grey floor around it by this much,
# in a patch of at least this share of the sheet. A robot touching the sheet's
# edge is cut off and dropped, and so is an outline that fills its own box
# (a mat or a square, not a robot).
NEWBOT_DIFF = 28
NEWBOT_SHARE = (0.006, 0.12)
NEWBOT_EDGE = 6
NEWBOT_BOXY = 0.9

# A robot keeps the detail the camera recorded and takes its colour from the
# model, toned down to sit with the calibrated floor. Its overall brightness
# moves halfway to the model's: with the camera's brightness kept outright, a
# white robot painted red came out pink.
ROBOT_CHROMA = 0.8


def generate(images, prompt, size, seed, out):
    """The one place a model is called. Swap this to change model. With no
    pictures it draws from the prompt alone. A picture already generated is
    not generated again: on the Mac each takes minutes."""
    if out.exists():
        return
    import torch
    if not torch.cuda.is_available():
        given = ["--image-paths", *map(str, images)] if images else []
        subprocess.run([str(MFLUX if images else MFLUX_T2I), "--model", MFLUX_MODEL, "--base-model", MFLUX_BASE,
                        *given, "--prompt", prompt,
                        "--width", str(size[0]), "--height", str(size[1]),
                        "--steps", "4", "--guidance", "1.0", "--seed", str(seed),
                        "--output", str(out)], check=True)
        return
    global _klein
    if _klein is None:
        from diffusers import Flux2KleinPipeline
        _klein = Flux2KleinPipeline.from_pretrained(KLEIN, torch_dtype=torch.bfloat16).to("cuda")
    from PIL import Image
    given = {"image": [Image.open(i).convert("RGB") for i in images]} if images else {}
    _klein(**given, prompt=prompt,
           width=size[0], height=size[1], num_inference_steps=4, guidance_scale=1.0,
           generator=torch.Generator("cuda").manual_seed(seed)).images[0].save(out)


def repaint(ab, arena):
    cast = np.median(ab[arena], 0)   # the floor is most of the arena
    off = np.hypot(ab[:, :, 0] - cast[0], ab[:, :, 1] - cast[1])
    strong = np.minimum(off, np.hypot(ab[:, :, 0], ab[:, :, 1]))
    hue = np.degrees(np.arctan2(ab[:, :, 1], ab[:, :, 0])) % 360
    out = ab + (np.float32(FLOOR) - cast) * np.clip(2 - 2 * off / PAINT_CHROMA, 0, 1)[:, :, None]
    for (low, high), target in PAINT.values():
        inside = ((hue - low) % 360) <= ((high - low) % 360)
        # measured on the floor only: the full model paints the bumpers a
        # stronger yellow than the slot outlines, and the target is the floor's
        sure = inside & (strong > PAINT_CHROMA) & arena
        if sure.sum() < 500:
            continue
        now = np.median(ab[sure], 0)
        turn = np.arctan2(target[1], target[0]) - np.arctan2(now[1], now[0])
        grow = np.hypot(*target) / np.hypot(*now)
        spin = np.float32([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]) * grow
        weight = cv2.GaussianBlur((inside & (strong > PAINT_CHROMA * 2 / 3)).astype(np.float32), (0, 0), 1.5)[:, :, None]
        out = out * (1 - weight) + (ab @ spin.T) * weight
    return out


def relight(gray, generated):
    """Colour from the model, fine detail from the real plate, so every
    marking and scratch stays where the camera saw it."""
    h, w = gray.shape
    lab = cv2.cvtColor(cv2.resize(generated, (w, h), interpolation=cv2.INTER_CUBIC),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    real = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)[:, :, 0]
    real = real.astype(np.float32)
    lab[:, :, 0] = np.clip(real + 0.5 * (cv2.GaussianBlur(lab[:, :, 0], (0, 0), DETAIL)
                                         - cv2.GaussianBlur(real, (0, 0), DETAIL)), 0, 255)
    arena = arena_mask(gray)
    inside = cv2.GaussianBlur(arena.astype(np.float32) / 255, (0, 0), 12.0)
    ab = repaint(cv2.GaussianBlur(lab[:, :, 1:], (0, 0), 2.0) - 128, arena > 0)
    lab[:, :, 1:] = 128 + ab * (OUTSIDE + (1 - OUTSIDE) * inside)[:, :, None]
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def empty_reference():
    reference = cv2.imread(str(REFERENCE))
    for x0, y0, x1, y1 in REFERENCE_ROBOTS:
        # A square shades smoothly from side to side and toward the camera, so
        # a flat cover shows as a rectangle. The cover is a plane fitted to
        # the square just above and below the robot.
        ys, xs = np.mgrid[y0 - 12:y1 + 12, x0:x1]
        near = reference[y0 - 12:y1 + 12, x0:x1].astype(np.float32)
        ring = (ys < y0) | (ys >= y1)
        ring &= np.abs(near - np.median(near[ring], 0)).sum(2) < 80   # not a bumper or a scuff
        plane = np.linalg.lstsq(np.c_[xs[ring], ys[ring], np.ones(ring.sum())], near[ring], rcond=None)[0]
        reference[y0:y1, x0:x1] = np.clip(np.dstack([xs, ys, np.ones_like(xs)]) @ plane, 0, 255)[12:-12]
    return reference


def plate(clip, seed):
    out = PLATES / clip
    out.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(REAL / clip / "plate.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(out / "input.png"),
                cv2.cvtColor(cv2.resize(gray, MODEL_SIZE, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(out / "reference.jpg"),
                cv2.resize(empty_reference(), REFERENCE_SIZE, interpolation=cv2.INTER_AREA))
    generate([out / "input.png", out / "reference.jpg"], PLATE_PROMPT, MODEL_SIZE, seed,
             out / f"generated_{seed}.png")
    cv2.imwrite(str(out / f"plate_color_{seed}.png"),
                relight(gray, cv2.imread(str(out / f"generated_{seed}.png"))))
    print(out / f"plate_color_{seed}.png")


def cutouts(clip, sheets, seed, found=None, folder=CUTOUTS):
    """Colour a clip's cut-outs, run1's calm ones by default, or the combat
    poses with `found` and `folder`."""
    out = folder / clip
    out.mkdir(parents=True, exist_ok=True)
    per = GRID * GRID
    by_robot = {}
    for c in robots.cutouts(clip) if found is None else found:
        by_robot.setdefault(c["robot"], []).append(c)
    index = []
    for robot, found in by_robot.items():
        if sheets:
            found = [found[i] for i in np.linspace(0, len(found) - 1, min(sheets * per, len(found))).astype(int)]
        for s in range(0, len(found), per):
            n = s // per
            invented = n % INVENTED_EVERY == INVENTED_EVERY - 1
            sheet = np.full((CELL * GRID, CELL * GRID), 128, np.uint8)
            placed = []
            for i, c in enumerate(found[s:s + per]):
                d = Path(c.get("dir", REAL / clip))
                gray = cv2.imread(str(d / "images" / f"{c['frame']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
                ids = cv2.imread(str(d / "masks" / f"{c['frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
                x0, y0, x1, y1 = c["bbox"]
                crop, mask = gray[y0:y1 + 1, x0:x1 + 1], ids[y0:y1 + 1, x0:x1 + 1] == c["k"] + 1
                scale = (CELL - 32) / max(crop.shape)
                w, h = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
                small = cv2.resize(np.where(mask, crop, 128).astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_CUBIC)
                cx, cy = (i % GRID) * CELL + (CELL - w) // 2, (i // GRID) * CELL + (CELL - h) // 2
                sheet[cy:cy + h, cx:cx + w] = small
                placed.append((c, crop, mask, (cx, cy, w, h)))
            name = f"{robot}_{seed}_{n}"
            sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
            # A coloured sheet is only good for the grey sheet it was painted
            # from. --sheets 1 spreads its nine over all of a robot's cut-outs,
            # so its sheet 0 is not a full run's sheet 0; reusing it would put
            # one cut-out's colours on another.
            if (out / f"{name}.png").exists() and not np.array_equal(cv2.imread(str(out / f"{name}.png")), sheet):
                (out / f"{name}_color.png").unlink(missing_ok=True)
            cv2.imwrite(str(out / f"{name}.png"), sheet)
            prompt = INVENTED_PROMPT if invented else LIVERY_PROMPT.format(livery=robots.LIVERY[robot])
            generate([out / f"{name}.png"], prompt, (CELL * GRID, CELL * GRID), seed + n,
                     out / f"{name}_color.png")
            coloured = cv2.resize(cv2.imread(str(out / f"{name}_color.png")), (CELL * GRID, CELL * GRID))
            for c, crop, mask, (cx, cy, w, h) in placed:
                lab = cv2.cvtColor(cv2.resize(coloured[cy:cy + h, cx:cx + w], crop.shape[::-1],
                                              interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2LAB).astype(np.float32)
                real = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)[:, :, 0]
                ratio = np.clip(lab[:, :, 0][mask].mean() / max(real[mask].mean(), 1.0), 0.5, 1.2)
                lab[:, :, 0] = np.clip(real * (0.5 + 0.5 * ratio), 0, 255)
                lab[:, :, 1:] = 128 + (cv2.GaussianBlur(lab[:, :, 1:], (0, 0), 1.5) - 128) * ROBOT_CHROMA
                sprite = np.dstack([cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR),
                                    (mask * 255).astype(np.uint8)])
                file = f"{c['frame']:06d}_{c['k']}.png"
                cv2.imwrite(str(out / file), sprite)
                index.append({"file": file, "clip": clip, "frame": c["frame"], "robot": robot,
                              "paint": "invented" if invented else "livery",
                              "x0": c["bbox"][0], "y0": c["bbox"][1], **({"kind": c["kind"]} if "kind" in c else {})})
    (out / "index.json").write_text(json.dumps(index))
    print(f"{clip}: {len(index)} coloured cut-outs -> {out}")


def newbots(names, seeds):
    """Draw each roster robot that has no footage, nine views to a sheet, and
    cut every clean view out with SAM 2.1 into a sprite."""
    from ultralytics import SAM
    sam = SAM(str(AUGMENT / "data" / "weights" / "sam2.1_s.pt"))
    for name in names:
        out = NEWBOTS / name
        out.mkdir(parents=True, exist_ok=True)
        index = []
        for seed in seeds:
            sheet_path = out / f"sheet_{seed}.png"
            generate([], NEWBOT_PROMPT.format(description=robots.NEW_ROBOTS[name]), (CELL * GRID, CELL * GRID),
                     seed, sheet_path)
            sheet = cv2.imread(str(sheet_path))
            h, w = sheet.shape[:2]
            gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY).astype(np.int16)
            floor = cv2.medianBlur(gray.astype(np.uint8), 61).astype(np.int16)
            sat = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)[:, :, 1]
            fg = ((np.abs(gray - floor) > NEWBOT_DIFF) | (sat > 60)).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            n, ids, stats, cents = cv2.connectedComponentsWithStats(fg, 8)
            blobs = [i for i in range(1, n) if NEWBOT_SHARE[0] <= stats[i, cv2.CC_STAT_AREA] / (h * w) <= NEWBOT_SHARE[1]]
            if not blobs:
                continue
            boxes = [[max(0, stats[i, 0] - 8), max(0, stats[i, 1] - 8), min(w - 1, stats[i, 0] + stats[i, 2] + 8),
                      min(h - 1, stats[i, 1] + stats[i, 3] + 8)] for i in blobs]
            r = sam(sheet, verbose=False, bboxes=boxes,
                    points=[[[int(cents[i][0]), int(cents[i][1])]] for i in blobs], labels=[[1]] * len(blobs))[0]
            for j, mask in enumerate(r.masks.data.cpu().numpy() > 0 if r.masks is not None else []):
                ys, xs = np.nonzero(mask)
                if not len(ys) or ys.min() < NEWBOT_EDGE or xs.min() < NEWBOT_EDGE \
                        or ys.max() >= h - NEWBOT_EDGE or xs.max() >= w - NEWBOT_EDGE:
                    continue
                box_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
                if mask.sum() / box_area > NEWBOT_BOXY or not NEWBOT_SHARE[0] <= mask.sum() / (h * w) <= NEWBOT_SHARE[1]:
                    continue
                crop = sheet[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                alpha = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                # toned down as the coloured cut-outs are, to sit with the floor
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:, :, 1:] = 128 + (lab[:, :, 1:] - 128) * ROBOT_CHROMA
                sprite = np.dstack([cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), (alpha * 255).astype(np.uint8)])
                cv2.imwrite(str(out / f"{seed}_{j}.png"), sprite)
                index.append({"file": f"{seed}_{j}.png", "robot": name, "seed": seed})
        (out / "index.json").write_text(json.dumps(index))
        print(f"{name}: {len(index)} views cut out -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plate", "cutouts", "poses", "newbots"))
    ap.add_argument("clips", nargs="+", help="clips, or for newbots robot names")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--sheets", type=int, default=0,
                    help="cut-outs: at most this many sheets of nine per robot (default: all)")
    args = ap.parse_args()
    # one process for everything asked, so the CUDA model is loaded once
    if args.what == "newbots":
        newbots(args.clips, args.seeds)
    for clip in args.clips if args.what != "newbots" else []:
        for seed in args.seeds:
            if args.what == "plate":
                plate(clip, seed)
        if args.what == "cutouts":
            cutouts(clip, args.sheets, args.seeds[0])
        if args.what == "poses":
            import combat
            cutouts(clip, 0, args.seeds[0], combat.poses(clip), POSES)
