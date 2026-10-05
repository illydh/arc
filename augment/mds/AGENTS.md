# Agent instructions

See `context.md` for project background, measured facts about the inputs, and
open decisions.

## Stack
Python 3.12 locally (3.11.9 on CARC), PyTorch (MPS locally, CUDA on CARC),
diffusers, Ultralytics 8.4.11, OpenCV, NumPy.

These conventions are not `vision/`'s or `splatting/`'s. Each of the three has
its own requirements manifest and none should leak into another. This one also
has its own venv: the root `.venv` is pinned to nerfstudio and has two OpenCV
builds clobbering each other.

## Layout
```
mds/                 context.md, carc.md + this file (tracked, mirroring vision/mds/ and splatting/mds/)
scripts/extract.py   plates, labelled real frames and robot cut-outs from the fight clips
scripts/robots.py    which robot is which in each clip, and each robot's livery in words
scripts/colorize.py  colours a clip's plate, or its robot cut-outs, with the image model
scripts/compose.py   robots onto the coloured plates -> labelled images, colour and grey
scripts/augment_report.py  grades a composed run, GO / MARGINAL / NO-GO
scripts/classes.py   the detector's class ids, written into every label file
scripts/carc/        setup_env.sh, generate.job for CARC; stage.sh, fetch.sh run on the Mac
mlx/                 requirements.txt + .venv for the Mac-only image model runtime (mflux)
out/<run>/           compose.py and augment_report.py output -- git-ignored:
  colour/ grey/        each images/, labels/, data.yaml: ready to train on
  manifest_*.json      what is in every image, one part per compose process
  report/              report.json, sheet_*.jpg, plates.jpg
data/               git-ignored:
  detector/           finetune_color.pt, the detector to improve
  meshes/             Meshy AI meshes of Nemesis and Gigabyte
  weights/            sam2.1_s.pt, a symlink to the SAM 2.1 checkpoint in the Hugging Face cache
  refs/               broadcast screenshots of the arena from above, the colour reference
  plates/<clip>/      colorize.py output: generated_<seed>.png, plate_color_<seed>.png
  sprites/<robot>/    compose.py's bank of mesh renders, rebuilt when its settings change
  sprites/cutouts/<clip>/  colorize.py output: <robot>_<seed>_<n>.png sheets, their _color.png,
                      one sprite per cut-out, index.json
  real/<clip>/        extract.py output: plate.png, images/, masks/, outlines.json (outline pass);
                      labels/, proposals/, sheets/, manifest.json (review pass)
.venv/              local environment -- git-ignored
```

No packages: a handful of scripts and one pipeline. Share code by importing
from the script that owns it, as `splat_clean.py` does from `splat_report.py`.
Import `vision/detect/blobs.py` and `vision/capture/source.py` rather than
copying them.

## Conventions
- Minimal, direct implementations. No speculative abstraction, no config
  options for cases that don't exist yet.
- No docstrings/comments unless they explain a non-obvious constraint.
- Every generated image carries its labels by construction: boxes come
  from the layout, not from a detector run afterwards.
- Nemesis is not put in these images. They train the other class.
- The image model sits behind one generate function, so swapping models does
  not touch the pipeline. The model is a measured choice, see context.md.
- Judge the result on real fight frames, not on generated ones. A generated
  set that scores well only on itself has proven nothing.
- Report thresholds are fixed at the top of the report script before a run,
  so runs are compared on the same bars.

## Running
```
~/.pyenv/versions/3.12.11/bin/python -m venv augment/.venv
augment/.venv/bin/python -m pip install -r augment/requirements.txt
```
Append new dependencies to `requirements.txt` as they are introduced.

```
augment/.venv/bin/python augment/scripts/extract.py <fight1..fight5> [--device mps|cpu] [--frames N]
augment/.venv/bin/python augment/scripts/extract.py <fight1..fight5> --review
```
The first form writes `data/real/<clip>/`, replacing what was there, and takes
3 to 9 minutes per clip on this machine. `--frames N` stops after N kept
frames, for a quick trial. `--review` skips SAM and re-judges the saved
outlines in a few seconds: use it after changing a check, `UNTRUSTED` or
`REJECTED`.

Look at `sheets/` before trusting a run: the script flags what its own checks
catch, not what they miss. Every accepted frame is on an `accepted_*.jpg`
sheet. Judge small or doubtful boxes on a full-resolution crop; at sheet scale
real far-wall robots look like debris. When a frame that passed is wrong, add
it to `REJECTED` with the reason and re-run with `--review`.

`labels/` holds a file only for frames that passed every check. A flagged
frame keeps its image and outlines but gets no label file; `manifest.json`
says why. `proposals/` holds the boxes found on every frame, flagged or not,
as a starting point for review. Do not train or score on proposals or on
flagged frames without reviewing them.

The outline model is SAM 2.1 through Ultralytics, so there is no separate
`sam2` package. `data/weights/sam2.1_s.pt` must be a symlink (or copy) of
`sam2.1_hiera_small.pt`; Ultralytics picks the architecture from that file name.

```
augment/.venv/bin/python augment/scripts/colorize.py plate <clip>... [--seeds N...]
augment/.venv/bin/python augment/scripts/colorize.py cutouts <clip>... [--sheets N]
augment/.venv/bin/python augment/scripts/compose.py <run> [-n 12] [--seed 0] [--start 0]
augment/.venv/bin/python augment/scripts/augment_report.py augment/out/<run>
```
`colorize.py` calls the image model: mflux from `augment/mlx/.venv` on this
Mac (build it from `augment/mlx/requirements.txt`), diffusers on a CUDA
machine. On the Mac each model run takes about 3 minutes and 12 GB of memory,
so run one at a time; `--sheets 1` colours one sheet of nine per robot. A
picture already generated is not generated again: delete it to have it redone.
Look at a result before using it. The model invents colour unless the prompt
says what stays grey, and it drifts from a livery it is told (it painted
Mammoth red until told "no red").

`compose.py` draws on every `plate_color_*.png` and every coloured cut-out
there is, so what `colorize.py` has made decides what a run looks like.
`--start` numbers the first image, so several processes can fill one run.
`augment_report.py` reads the whole run; its thresholds are fixed at the top.

Robot identities are in `scripts/robots.py`: the rule that splits each clip's
two robots, the frames where that rule was wrong (`SWAPPED`, `UNNAMED`), the
hand-picked Kraken cut-outs, and the liveries. After re-running `extract.py`
on a clip, check its robots again on a sheet before trusting the names.

Every robot these scripts label is `non-nemesis`. The class ids are in
`scripts/classes.py` and match the detector's `data.yaml`; after changing
them, re-run `extract.py <clip> --review` and `compose.py`.

The CARC half is in `mds/carc.md`.

Loading the detector: `YOLO("augment/data/detector/finetune_color.pt")`. It was
trained at `imgsz=640`; keep that for inference, native-size inference was
worse on the frames tried (context.md).

Do not patch `site-packages` to work around a library bug, and do not run
SAM 2 video tracking on this machine: it took two hours without finishing one
clip (context.md).

## Keeping context current
`context.md` and this file are working documents, not one-time setup.
Update them whenever:
- scope changes (a step is added, dropped, or reordered)
- an open decision gets resolved
- a pipeline stage is completed or its design changes from what's documented
- a new constraint or measurement is found that a future session would
  otherwise have to re-derive

Stale context is worse than missing context -- don't leave a resolved open
decision or an outdated description in place.
