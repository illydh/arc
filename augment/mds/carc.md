# Runbook -- bulk generation on USC CARC

Scripts live in `scripts/carc/`. Same practice as `splatting/`: a persistent
checkout at `/home1/illyhoan/arc` on CARC, jobs submitted from a CARC
terminal, paths hardcoded in the scripts, and only `RUN` (plus optional `N`,
`SHEETS`, `SEEDS`) given on the `sbatch` line.

**Not yet run on CARC.** Everything below was written and syntax-checked on
the Mac. The model call for CUDA (`colorize.generate`) cannot run there, so
the smoke test is its first real test.

## What runs where

| Step | Where | Needs |
|---|---|---|
| Upload inputs (`stage.sh`) | Mac | CARC login |
| Environment and model download (`setup_env.sh`) | CARC login node, once | internet |
| Colour plates and cut-outs, compose, grade (`generate.job`) | CARC, one A100 | nothing online |
| Download results (`fetch.sh`) | Mac | CARC login |

## 1. From the Mac: get code and inputs to CARC

The code travels by git, so `augment/` has to be committed and pushed first.
The inputs do not: `augment/data/` is git-ignored.

```sh
./augment/scripts/carc/stage.sh     # about 90 MB, a minute or two
```

It copies, into `/home1/illyhoan/arc/augment/data/`:

| What | Size | Used for |
|---|---|---|
| `real/<clip>/plate.png`, `manifest.json` for fights 1 to 5 | 8 MB | plates to colour; where real robots stood and how big they were |
| `real/<clip>/images/`, `masks/` of the labelled frames, plus fight1's hand-picked Kraken frames | 50 MB | the robot cut-outs |
| `refs/` | 1 MB | the broadcast screenshot the plates are coloured from |
| `meshes/` | 17 MB | Gigabyte |
| `detector/` | 5 MB | the report's detector check |
| `plates/` | 7 MB | fight4's plate with seed 3, approved by eye on the Mac |

Coloured cut-outs are not copied. CARC colours its own with the full model.

## 2. On CARC, once: environment

```sh
cd /home1/illyhoan/arc && git pull
cd augment/scripts/carc
bash setup_env.sh      # login node, about 15 minutes
```

It builds `/home1/illyhoan/arc/augment/.venv` and downloads FLUX.2 klein 4B
(about 16 GB) to `/scratch1/illyhoan/hf`. It ends by importing everything the
job needs and printing the versions; if that fails, the job would too.

Differences from the Mac's `requirements.txt`, on purpose:
- `torch==2.6.0` and `torchvision==0.21.0` from the `cu124` wheel index.
  `diffusers` 0.40.0 needs torch 2.6 or newer, and `cu124` pairs with the
  `cuda/12.4.1` module.
- `numpy` is left for pip to choose. The Mac's pin (2.5.3) needs Python 3.12
  and the CARC module is 3.11.9.

## 3. On CARC: smoke test, then the full run

```sh
cd /home1/illyhoan/arc/augment/scripts/carc
sbatch --export=ALL,RUN=smoke,N=50,SHEETS=1,SEEDS=3 generate.job
squeue -u $USER
tail -f /home1/illyhoan/arc/logs/augment-<jobid>.out
```

The smoke test colours one plate per fight and one sheet of nine cut-outs per
robot, composes about 50 images and grades them. A few minutes once the job
starts.

**Look before the full run.** Bring the smoke run down (step 4) and look at
`out/smoke/report/plates.jpg` and the coloured cut-out sheets
(`data/sprites/cutouts/<clip>/*_color.png`). The prompts were tuned on the
Mac's 4-bit build of the model; the full-precision one may paint differently.
On the Mac the model, left to itself, painted a red field across bare floor
and painted Mammoth red instead of black.

```sh
sbatch --export=ALL,RUN=run1 generate.job     # about 3,000 images
```

| Variable | Default | Meaning |
|---|---|---|
| `RUN` | required | output folder, `augment/out/<RUN>` |
| `N` | 3000 | number of images |
| `SHEETS` | 0 | cut-out sheets per robot; 0 means all (about 40 in total) |
| `SEEDS` | `1 2 3` | colour variants of each fight's plate |

A picture that already exists is not generated again. A re-run after a failure
picks up where it stopped, and the full run reuses the smoke test's plates and
sheets. To have something redone, delete its `generated_<seed>.png` or
`<robot>_7_<n>_color.png` first.

## 4. From the Mac: bring results down

```sh
./augment/scripts/carc/fetch.sh smoke     # or run1; the full run is about 2 GB
```

It writes `augment/out/<RUN>/` and overwrites the Mac's coloured plates and
cut-outs with CARC's.

## What comes back

```
out/<RUN>/
  colour/images/ labels/ data.yaml    ready to train on
  grey/images/ labels/ data.yaml      the same scenes as the monochrome camera would see them
  manifest_*.json                     what is in every image, one part per compose process
  report/report.json                  verdict GO / MARGINAL / NO-GO and the numbers behind it
  report/sheet_*.jpg plates.jpg       contact sheets with boxes; every plate used
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
| `FileNotFoundError` under `augment/data/real` | `stage.sh` was not run, or was run before `git pull` brought a newer `robots.py` |
| Report says a plate is not `ok` | the model painted where it should not, or its colours could not be calibrated; look at `plates.jpg`, delete that plate's `generated_<seed>.png` and `plate_color_<seed>.png`, and re-run with another seed |
