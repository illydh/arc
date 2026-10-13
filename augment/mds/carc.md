# Runbook -- bulk generation on USC CARC

Scripts live in `scripts/carc/`. Same practice as `splatting/`: a persistent
checkout at `/home1/illyhoan/arc` on CARC, jobs submitted from a CARC
terminal, paths hardcoded in the scripts, and only `RUN` (plus optional `N`,
`SHEETS`, `SEEDS`) given on the `sbatch` line.

**Smoke run done 2026-10-05, verdict MARGINAL. Next: the full run (step 5),
with no second smoke test (decided 2026-10-12).** The job works end to end on
CARC and the cut-out sheets are right, but the three plates the full model
coloured were wrong: a blue floor, and two blobs copied from the robots in
the broadcast screenshot. Both are fixed in the code and checked on the Mac
against the failed pictures; the full model has not run the fix yet. So the
job now checks every plate itself and sets a failing one aside before
anything is composed on it (`augment_report.py gate`). Details in
`context.md`, CARC smoke run. `setup_env.sh` first stopped on the login
node's process limit (see Likely failure modes) and now caps its threads.

The user commits, pushes and submits (decided 2026-10-04).

**For the full run only steps 1, 2, 5 and 6 are left.** The inputs (step 3)
and the environment with the model (step 4) are on CARC from the smoke run
and need nothing more.

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
git-ignored, so this adds only code and docs. For the full run that is six
changed files: the plate fix, the plate gate and the docs.

```sh
git add augment
git commit -m "augment: plate fix and plate gate for the full run"
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

Running it again is safe and takes a minute or two: the packages and the
model are already there and are not fetched twice.

Differences from the Mac's `requirements.txt`, on purpose:
- `torch==2.6.0` and `torchvision==0.21.0` from the `cu124` wheel index.
  `diffusers` 0.40.0 needs torch 2.6 or newer, and `cu124` pairs with the
  `cuda/12.4.1` module.
- `numpy` is left for pip to choose. The Mac's pin (2.5.3) needs Python 3.12
  and the CARC module is 3.11.9.

## 5. CARC: the full run

Pull, delete the three plates the smoke run got wrong, and submit:

```sh
cd /home1/illyhoan/arc && git pull
rm -f augment/data/plates/fight{2,3,5}/generated_3.png augment/data/plates/fight{2,3,5}/plate_color_3.png
cd augment/scripts/carc
sbatch --export=ALL,RUN=run1 generate.job
squeue -u $USER
tail -f /home1/illyhoan/arc/logs/augment-<jobid>.out
```

Without the `rm` the job reuses the old plate pictures, robots and all.
Nothing needs uploading: the fix is in the code, and the inputs on CARC are
unchanged.

What the job does, in order:
1. Colours 11 more plates (fights 2 to 5, seeds 1 to 3; fight4 seed 3 is the
   Mac's).
2. **Gate.** Checks every plate against the approved one, the broadcast
   colours and for empty start squares. A plate that fails is renamed
   `rejected_plate_color_<seed>.png`, which `compose.py` does not pick up.
   The log says `kept` or `set aside` for each, with the numbers.
3. Colours all 41 cut-out sheets. The nine from the smoke run are painted
   again by themselves: a smoke run spreads its one sheet per robot over all
   of that robot's cut-outs, so those sheets hold other cut-outs than a full
   run's first sheets, and `colorize.py` repaints a sheet whose grey sheet
   has changed.
4. Composes 3,000 images on the plates that passed, grades them, and copies
   the painted pictures beside the report.

Its log ends with the report and a `done;` line.

**No second smoke test.** It would only have looked at the plates, and the
gate now does that on every run. What is still unseen until the results are
down: the cut-out sheets beyond the first of each robot, and the
invented-paint sheets (tried once on the Mac's build, never on the full
model). They come back in `report/cutouts/`; a bad sheet costs a recompose,
not a new run.

If fewer than six plates pass, the report says so and grades the run
MARGINAL: the images are valid but share too few backgrounds. The raw
pictures in `report/plates/` show why each failed.

For a quick trial instead, the smoke settings still work:
`sbatch --export=ALL,RUN=smoke2,N=50,SHEETS=1,SEEDS=3 generate.job`.

| Variable | Default | Meaning |
|---|---|---|
| `RUN` | required | output folder, `augment/out/<RUN>` |
| `N` | 3000 | number of images |
| `SHEETS` | 0 | cut-out sheets per robot; 0 means all (about 40 in total) |
| `SEEDS` | `1 2 3` | colour variants of each fight's plate |

A picture that already exists is not generated again, except a cut-out sheet
whose grey sheet has changed. A re-run after a failure picks up where it
stopped, and the full run reuses the smoke test's plates. A plate set aside by the gate is rebuilt from its raw picture and set
aside again on every run until that raw picture is deleted. To have something redone, delete it on CARC first: a plate's
`data/plates/<clip>/generated_<seed>.png` and `plate_color_<seed>.png`, or a
sheet's `data/sprites/cutouts/<clip>/<robot>_7_<n>_color.png`.

## 6. Mac: bring results down

```sh
./augment/scripts/carc/fetch.sh smoke
```

Or `run1`; the full run is about 2 GB. One rsync, one CARC login. It writes
`augment/out/<RUN>/` and prints the verdict.

To copy by other means, pack the run into one file on CARC and unpack it on
the Mac from the repo root:

```sh
# CARC
tar -cf /home1/illyhoan/arc/augment-run1.tar -C /home1/illyhoan/arc/augment/out run1
# Mac, after copying augment-run1.tar into the repo root
tar -xf augment-run1.tar -C augment/out && rm augment-run1.tar
```

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
  report/plates/                      the plates as the model painted them, before the colour correction
```

Every box is class `1`, `non-nemesis`. `data.yaml` has no `path`, so
Ultralytics reads `images/` beside it; its `val` points at the same images
only because the format requires the key. This is a pool to merge into a
training set, not a split.

## Likely failure modes

| Symptom | Cause |
|---|---|
| `setup_env.sh`: pip cannot resolve a package | a pin in `requirements.txt` has no build for Python 3.11; leave it out in the `grep -vE` line there, as numpy is |
| `OpenBLAS blas_thread_init: pthread_create failed ... Resource temporarily unavailable`, then `KeyboardInterrupt` in an import, on a login node | CARC allows a user 64 processes per login node, threads included. OpenBLAS (inside numpy) starts one per core, is refused, and interrupts its own process. Nothing is broken. `setup_env.sh` sets `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; put the same in front of any Python run by hand on a login node. If it still happens, something else of yours is holding the 64: `ps -u $USER -L --no-headers \| wc -l` |
| `setup_env.sh`: `ImportError: libGL.so.1` on `import cv2` | Ultralytics pulled `opencv-python`; swap it for `opencv-python-headless` (the script says how), never both |
| `no GPU visible` | `--gpus-per-task` wrong, or the partition has no free A100 |
| Job stalls or fails at the first model call with a hub error | weights not in `/scratch1/illyhoan/hf` (scratch purged, or `setup_env.sh` not run); jobs run offline, so run `setup_env.sh` again |
| CUDA out of memory | not expected: the model needs about 13 GB and an A100 has 40 |
| `no coloured plates or cut-outs` from `compose.py` | the colouring step failed; read the log above that line |
| `stage.sh`: rsync says the destination does not exist | `git pull` on CARC has not created `augment/` yet |
| `FileNotFoundError` under `augment/data/real` on CARC | `stage.sh` was not run, or `scripts/robots.py` changed since and names frames that were not uploaded; run `stage.sh` again |
| Log says a plate was `set aside`, or the report lists it under `plates_set_aside` | the gate working: that plate failed a check and nothing was composed on it. To get a replacement, delete its `generated_<seed>.png` on CARC and run again with another seed in `SEEDS` |
| Report says a plate is not `ok` | low `paint_iou`: the model painted where it should not. High `colour_off`: its colours could not be calibrated. `square_gaps` over 0.09: something sits on a start square. Look at `plates.jpg` and the raw picture in `report/plates/`, delete that plate's `generated_<seed>.png` and `plate_color_<seed>.png` on CARC, and re-run with another seed |
