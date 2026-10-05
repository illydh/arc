# Runbook -- bulk generation on USC CARC

Scripts live in `scripts/carc/`. Same practice as `splatting/`: a persistent
checkout at `/home1/illyhoan/arc` on CARC, jobs submitted from a CARC
terminal, paths hardcoded in the scripts, and only `RUN` (plus optional `N`,
`SHEETS`, `SEEDS`) given on the `sbatch` line.

**Not yet run on CARC.** Everything below was written and checked on the
Mac: the scripts pass a syntax check, the upload was rehearsed into a local
folder (392 files, 80 MB), and a 50-image run of the same composer and report
grades GO. The model call for CUDA (`colorize.generate`) cannot run on the
Mac, so the smoke test is its first real test.

The user commits, pushes and submits (decided 2026-10-04).

## What runs where

| Step | Where | Needs |
|---|---|---|
| 1. Commit and push | Mac | GitHub |
| 2. `git pull` | CARC terminal | |
| 3. Upload inputs (`stage.sh`) | Mac | one CARC login |
| 4. Environment and model download (`setup_env.sh`) | CARC login node, once | internet |
| 5. Smoke test, then the full run (`generate.job`) | CARC, one A100 | nothing online |
| 6. Download results (`fetch.sh`) | Mac | one CARC login |

## 1. Mac: commit and push

From the repo root. `augment/data/`, `augment/out/` and the venvs are
git-ignored, so this adds only code and docs (15 files).

```sh
git add augment CLAUDE.md
git commit -m "augment: generated non-nemesis training images, CARC job"
git push
```

## 2. CARC: pull

```sh
cd /home1/illyhoan/arc && git pull
```

This has to come before the upload: it creates `augment/`, which the upload
copies into.

## 3. Mac: upload the inputs

```sh
./augment/scripts/carc/stage.sh
```

One rsync, so one CARC login. It copies into
`/home1/illyhoan/arc/augment/data/`:

| What | Size | Used for |
|---|---|---|
| `real/<clip>/plate.png`, `manifest.json` for fights 1 to 5 | 8 MB | plates to colour; where real robots stood and how big they were |
| `real/<clip>/images/`, `masks/` of the labelled frames, plus fight1's hand-picked Kraken frames | 50 MB | the robot cut-outs |
| `refs/` | 1 MB | the broadcast screenshot the plates are coloured from |
| `meshes/` | 17 MB | Gigabyte |
| `detector/` | 5 MB | the report's detector check |
| `plates/` | 2 MB | fight4's plate with seed 3, approved by eye on the Mac |

Coloured cut-outs are not copied. CARC colours its own with the full model.

To copy by other means, `./augment/scripts/carc/stage.sh bundle` writes the
same files to `augment/out/carc-inputs.tar.gz` (72 MB). On CARC:
`tar -xzf carc-inputs.tar.gz -C /home1/illyhoan/arc/augment`.

## 4. CARC, once: environment

```sh
cd /home1/illyhoan/arc/augment/scripts/carc
bash setup_env.sh
```

On a login node, about 15 minutes. It builds
`/home1/illyhoan/arc/augment/.venv` and downloads FLUX.2 klein 4B (about
16 GB) to `/scratch1/illyhoan/hf`. It ends by importing everything the job
needs and printing the versions; if that fails, the job would too.

Differences from the Mac's `requirements.txt`, on purpose:
- `torch==2.6.0` and `torchvision==0.21.0` from the `cu124` wheel index.
  `diffusers` 0.40.0 needs torch 2.6 or newer, and `cu124` pairs with the
  `cuda/12.4.1` module.
- `numpy` is left for pip to choose. The Mac's pin (2.5.3) needs Python 3.12
  and the CARC module is 3.11.9.

## 5. CARC: smoke test, then the full run

```sh
sbatch --export=ALL,RUN=smoke,N=50,SHEETS=1,SEEDS=3 generate.job
squeue -u $USER
tail -f /home1/illyhoan/arc/logs/augment-<jobid>.out
```

The smoke test colours one plate per fight and one sheet of nine cut-outs per
robot, composes about 50 images and grades them. A few minutes once the job
starts. Its log ends with the report and a `done;` line.

**Look before the full run.** Bring the smoke run down (step 6) and look at
`report/plates.jpg` and `report/cutouts/`. The prompts were tuned on the Mac's
4-bit build of the model; the full-precision one may paint differently. On
the Mac the model, left to itself, painted a red field across bare floor and
painted Mammoth red instead of black. What right looks like:
- every plate: charcoal floor, red square left, blue square right, a red and
  a blue B in the centre, yellow slot outlines, and `"ok": true` in
  `report.json`;
- every cut-out sheet: the livery in `scripts/robots.py`, the same on all nine.

```sh
sbatch --export=ALL,RUN=run1 generate.job
```

| Variable | Default | Meaning |
|---|---|---|
| `RUN` | required | output folder, `augment/out/<RUN>` |
| `N` | 3000 | number of images |
| `SHEETS` | 0 | cut-out sheets per robot; 0 means all (about 40 in total) |
| `SEEDS` | `1 2 3` | colour variants of each fight's plate |

A picture that already exists is not generated again. A re-run after a failure
picks up where it stopped, and the full run reuses the smoke test's plates and
sheets. To have something redone, delete it on CARC first: a plate's
`data/plates/<clip>/generated_<seed>.png` and `plate_color_<seed>.png`, or a
sheet's `data/sprites/cutouts/<clip>/<robot>_7_<n>_color.png`.

## 6. Mac: bring results down

```sh
./augment/scripts/carc/fetch.sh smoke
```

Or `run1`; the full run is about 2 GB. One rsync, one CARC login. It writes
`augment/out/<RUN>/` and prints the verdict.

## What comes back

```
out/<RUN>/
  colour/images/ labels/ data.yaml    ready to train on
  grey/images/ labels/ data.yaml      the same scenes as the monochrome camera would see them
  manifest_*.json                     what is in every image, one part per compose process
  report/report.json                  verdict GO / MARGINAL / NO-GO and the numbers behind it
  report/sheet_*.jpg                  contact sheets with boxes
  report/plates.jpg                   every plate the run used
  report/cutouts/                     the coloured cut-out sheets, as the model painted them
```

Every box is class `1`, `non-nemesis`. `data.yaml` has no `path`, so
Ultralytics reads `images/` beside it; its `val` points at the same images
only because the format requires the key. This is a pool to merge into a
training set, not a split.

## Likely failure modes

| Symptom | Cause |
|---|---|
| `setup_env.sh`: pip cannot resolve a package | a pin in `requirements.txt` has no build for Python 3.11; leave it out in the `grep -vE` line there, as numpy is |
| `setup_env.sh`: `ImportError: libGL.so.1` on `import cv2` | Ultralytics pulled `opencv-python`; swap it for `opencv-python-headless` (the script says how), never both |
| `no GPU visible` | `--gpus-per-task` wrong, or the partition has no free A100 |
| Job stalls or fails at the first model call with a hub error | weights not in `/scratch1/illyhoan/hf` (scratch purged, or `setup_env.sh` not run); jobs run offline, so run `setup_env.sh` again |
| CUDA out of memory | not expected: the model needs about 13 GB and an A100 has 40 |
| `no coloured plates or cut-outs` from `compose.py` | the colouring step failed; read the log above that line |
| `stage.sh`: rsync says the destination does not exist | `git pull` on CARC has not created `augment/` yet |
| `FileNotFoundError` under `augment/data/real` on CARC | `stage.sh` was not run, or `scripts/robots.py` changed since and names frames that were not uploaded; run `stage.sh` again |
| Report says a plate is not `ok` | the model painted where it should not, or its colours could not be calibrated; look at `plates.jpg`, delete that plate's `generated_<seed>.png` and `plate_color_<seed>.png`, and re-run with another seed |
