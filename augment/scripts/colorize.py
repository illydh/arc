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

# FLUX.2 klein 4B. On a CUDA machine (CARC) it runs at full precision through
# diffusers and stays loaded between calls. Its two large parts (7.8 + 8.0 GB)
# do not fit this 16 GB Mac together, so here a 4-bit build (4.6 GB) runs
# through mflux, which lives in its own venv because it pins its own stack.
# The prompts were tuned on the 4-bit build: look at what the full one paints.
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
MFLUX = AUGMENT / "mlx" / ".venv" / "bin" / "mflux-generate-flux2-edit"
MFLUX_MODEL = "RunPod/FLUX.2-klein-4B-mflux-4bit"
MFLUX_BASE = "flux2-klein-4b"
_klein = None

# The arena from above, in colour: a crop of a screenshot of public broadcast
# footage that the user supplied (2026-10-04), in the same orientation as the
# fight camera. It replaced a ground-level phone still, which could not say
# which start square is red or that the centre logo is painted at all.
REFERENCE = AUGMENT / "data" / "refs" / "arena_broadcast_1.jpg"
REFERENCE_SIZE = (768, 272)
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
# A robot keeps the detail the camera recorded and takes its colour from the
# model, toned down to sit with the calibrated floor. Its overall brightness
# moves halfway to the model's: with the camera's brightness kept outright, a
# white robot painted red came out pink.
ROBOT_CHROMA = 0.8


def generate(images, prompt, size, seed, out):
    """The one place a model is called. Swap this to change model. A picture
    already generated is not generated again: on the Mac each takes minutes."""
    if out.exists():
        return
    import torch
    if not torch.cuda.is_available():
        subprocess.run([str(MFLUX), "--model", MFLUX_MODEL, "--base-model", MFLUX_BASE,
                        "--image-paths", *map(str, images), "--prompt", prompt,
                        "--width", str(size[0]), "--height", str(size[1]),
                        "--steps", "4", "--guidance", "1.0", "--seed", str(seed),
                        "--output", str(out)], check=True)
        return
    global _klein
    if _klein is None:
        from diffusers import Flux2KleinPipeline
        _klein = Flux2KleinPipeline.from_pretrained(KLEIN, torch_dtype=torch.bfloat16).to("cuda")
    from PIL import Image
    _klein(image=[Image.open(i).convert("RGB") for i in images], prompt=prompt,
           width=size[0], height=size[1], num_inference_steps=4, guidance_scale=1.0,
           generator=torch.Generator("cuda").manual_seed(seed)).images[0].save(out)


def repaint(ab):
    chroma = np.hypot(ab[:, :, 0], ab[:, :, 1])
    hue = np.degrees(np.arctan2(ab[:, :, 1], ab[:, :, 0])) % 360
    bare = chroma < 8
    out = ab + (np.float32(FLOOR) - np.median(ab[bare], 0)) * np.clip(1 - chroma / 12, 0, 1)[:, :, None]
    for (low, high), target in PAINT.values():
        inside = ((hue - low) % 360) <= ((high - low) % 360)
        sure = inside & (chroma > 12)
        if sure.sum() < 500:
            continue
        now = np.median(ab[sure], 0)
        turn = np.arctan2(target[1], target[0]) - np.arctan2(now[1], now[0])
        grow = np.hypot(*target) / np.hypot(*now)
        spin = np.float32([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]) * grow
        weight = cv2.GaussianBlur((inside & (chroma > 8)).astype(np.float32), (0, 0), 1.5)[:, :, None]
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
    inside = cv2.GaussianBlur(arena_mask(gray).astype(np.float32) / 255, (0, 0), 12.0)
    ab = repaint(cv2.GaussianBlur(lab[:, :, 1:], (0, 0), 2.0) - 128)
    lab[:, :, 1:] = 128 + ab * (OUTSIDE + (1 - OUTSIDE) * inside)[:, :, None]
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def plate(clip, seed):
    out = PLATES / clip
    out.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(REAL / clip / "plate.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(out / "input.png"),
                cv2.cvtColor(cv2.resize(gray, MODEL_SIZE, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(out / "reference.jpg"),
                cv2.resize(cv2.imread(str(REFERENCE)), REFERENCE_SIZE, interpolation=cv2.INTER_AREA))
    generate([out / "input.png", out / "reference.jpg"], PLATE_PROMPT, MODEL_SIZE, seed,
             out / f"generated_{seed}.png")
    cv2.imwrite(str(out / f"plate_color_{seed}.png"),
                relight(gray, cv2.imread(str(out / f"generated_{seed}.png"))))
    print(out / f"plate_color_{seed}.png")


def cutouts(clip, sheets, seed):
    out = CUTOUTS / clip
    out.mkdir(parents=True, exist_ok=True)
    per = GRID * GRID
    by_robot = {}
    for c in robots.cutouts(clip):
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
                d = REAL / clip
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
            cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR))
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
                              "x0": c["bbox"][0], "y0": c["bbox"][1]})
    (out / "index.json").write_text(json.dumps(index))
    print(f"{clip}: {len(index)} coloured cut-outs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plate", "cutouts"))
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--sheets", type=int, default=0,
                    help="cut-outs: at most this many sheets of nine per robot (default: all)")
    args = ap.parse_args()
    # one process for everything asked, so the CUDA model is loaded once
    for clip in args.clips:
        for seed in args.seeds:
            if args.what == "plate":
                plate(clip, seed)
        if args.what == "cutouts":
            cutouts(clip, args.sheets, args.seeds[0])
