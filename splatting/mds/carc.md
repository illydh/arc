# Phase 2 runbook -- training on USC CARC

Scripts live in `scripts/carc/`.

Phase 2 has run on CARC (jobs 12341373, 12341876, 12343109, 12371036 while
this was stabilizing). The actual practice has been a persistent checkout at
`/home1/illyhoan/arc` on CARC, submitting `sbatch` directly from a CARC
terminal -- not the laptop-upload path this doc originally assumed. `train.job`
and `setup_env.sh` are now self-contained accordingly: `CARC_ENV`, `CARC_DATA`,
`CARC_OUT` and the `--output` log path are hardcoded in them, all under
`/home1/illyhoan/arc/`, and nothing is read from the submitting environment
except `RUN` (the dataset), `LABEL` (the name of this training run) and
optionally `ITERS` and `EXTRA`.

## Submitting a job (CARC terminal)

This is the whole procedure. Both scripts assume the dataset already exists
at `/home1/illyhoan/arc/splatting/data/processed/interior400/`. That dataset was produced locally by Phase 1 and copied to CARC by hand -- see "Optional: syncing from a laptop" below only if a future run's data isn't already there.

```sh
cd /home1/illyhoan/arc/splatting/scripts/carc

bash setup_env.sh   # step 1, once only (~10 min): builds the venv, creates
                     # the log directory. Skip on every later submit.

# step 2, every submit. LABEL is required and must be new:
sbatch --export=ALL,RUN=interior400,LABEL=smoke,ITERS=500 train.job   # smoke test, ~2 min
squeue -u $USER
tail -f /home1/illyhoan/arc/logs/splatfacto-<jobid>.out

sbatch --export=ALL,RUN=interior400,LABEL=run3 train.job              # full run
```

That's two scripts total, one of them (`setup_env.sh`) run once ever, not
once per job. `stage.sh`/`fetch.sh` are not part of this and don't need to run.

One submission now does everything for a run: train, export `splat.ply`,
score the held-out views. It all lands in one folder,
`out/<RUN>/splatfacto/<LABEL>/`, so runs can't overwrite each other and can
run in parallel. `train.job` refuses a `LABEL` whose folder already exists.

Extra `ns-train` settings go in `EXTRA`, set in the submitting shell (inside
`--export` its commas and spaces break). Dataparser flags go last, after the
`nerfstudio-data` subcommand:

```sh
EXTRA="--pipeline.model.use-bilateral-grid True nerfstudio-data --downscale-factor 1" \
  sbatch --export=ALL,RUN=interior400,LABEL=run3 train.job
```

## Run 2 (two parallel jobs)

The whole iteration is two commands, one on each machine.

```sh
# 1. CARC terminal, once the scripts are pushed:
cd /home1/illyhoan/arc && git pull && bash splatting/scripts/carc/submit_run2.sh

# 2. laptop, from the repo root, once both jobs have finished:
rsync -avh --exclude nerfstudio_models --exclude '2026-*' \
  illyhoan@discovery.usc.edu:/home1/illyhoan/arc/splatting/out/interior400/splatfacto/ \
  splatting/out/splatfacto/
```

`submit_run2.sh` records the exact settings and submits both jobs; each job
trains, exports and scores on its own. Both turn on camera pose refinement,
the bilateral grid and scale regularization, all off in run 1. `run2a`
trains at half size like run 1, so the two compare on the same 28 held-out
views at the same resolution. `run2b` trains at full size (2560x1440); it is
slower and uses more GPU memory, both unmeasured. The script checks that
every full-size frame named in `transforms.json` is on CARC and skips
`run2b`, with a message, if one is missing. Logs are
`logs/run2a-<jobid>.out` and `logs/run2b-<jobid>.out`.

The `rsync` copies every labelled run in one go. It skips the checkpoints
(about 740 MB at half size, not needed locally) and run 1's timestamped
folder, which is already local. Then grade each run:

```sh
.venv/bin/python splatting/scripts/splat_report.py \
  splatting/out/splatfacto/run2a/export/splat.ply splatting/out/splatfacto/run2a \
  splatting/data/processed/interior400 splatting/out/splatfacto/run2a/eval
```

## After training: export-only and eval-only jobs

Only needed if a step of `train.job` failed. Both take the run's `LABEL` and
never retrain. Run 1 predates labels; its label is its timestamp.

```sh
sbatch --export=ALL,RUN=interior400,LABEL=run2a export.job   # rewrite splat.ply only
sbatch --export=ALL,RUN=interior400,LABEL=run2a eval.job     # re-score held-out views
sbatch --export=ALL,RUN=interior400,LABEL=2026-09-27_171857 eval.job   # run 1
```

`eval.job` runs `ns-eval` on the final checkpoint and writes
`out/<RUN>/splatfacto/<LABEL>/eval/metrics.json` (mean and spread of PSNR /
SSIM / LPIPS over the held-out views) and `.../eval/renders/eval_img_*.png`
(real photo left, render right). Pass that `eval/` folder as
`splat_report.py`'s fourth argument. It runs on CARC for the same reason the
export does: the model only loads onto a CUDA device. Run 1's earlier
outputs stay where the old scripts put them, `out/interior400/export/` and
`out/interior400/eval/`.

## Preflight

Resolved for this account: partition `gpu`, `--gpus-per-task=a100:1`, modules
`gcc/12.3.0 cuda/12.4.1 python/3.11.9`. If the cluster config changes, redo
this and update both scripts' `module load` lines -- they must match exactly,
or gsplat's CUDA kernel compile breaks mid-job:

| What | Command | Goes into |
|---|---|---|
| Slurm account | `myaccount`, or `sacctmgr show assoc user=$USER format=account,partition` | `--account=` |
| GPU partition | `sinfo -s` | `--partition=` |
| GPU types available | `sinfo -o "%P %G %D"` | `--gres=gpu:<type>:1` |
| Module names | `module avail cuda`, `module avail python`, `module avail gcc` | both scripts' `module load` lines |

## Optional: syncing from a laptop

Not part of submitting a job -- skip this section entirely if the CARC
checkout already has the dataset, which is how this has actually been run.
Use `stage.sh`/`fetch.sh` only if working from a laptop without a persistent
CARC checkout, to move a dataset up or a finished run down over `rsync`/`ssh`.
They still read their paths from the environment; set these to match the
hardcoded paths above, or staging and training will look in different places:

```sh
export CARC_USER=illyhoan
export CARC_HOST=discovery.usc.edu
export CARC_CODE=/home1/illyhoan/arc/splatting
export CARC_DATA=/home1/illyhoan/arc/splatting/data/processed
export CARC_OUT=/home1/illyhoan/arc/splatting/out

./splatting/scripts/carc/stage.sh interior400   # laptop -> CARC, before a run
./splatting/scripts/carc/fetch.sh interior400   # CARC -> laptop, after a run
```

## What actually gets uploaded

Optional path only (see above) -- `stage.sh` sends two things.

**1. The scripts** -> `$CARC_CODE/scripts/carc/` (~4 KB): `setup_env.sh`,
`train.job`, `stage.sh`, `fetch.sh`. Only the first two are used there.

**2. The dataset** -> `$CARC_DATA/interior400/` (198 MB of the 654 MB on disk):

| Path | Size | Needed because |
|---|---|---|
| `images/` | 121 MB | the dataparser opens a full-res file to measure dimensions before choosing a downscale |
| `images_2/` | 48 MB | what training actually reads (1280x720) |
| `images_4/`, `images_8/` | 28 MB | not used at this resolution; harmless, and free if you re-run at higher downscale |
| `sparse_pc.ply` | 1.2 MB | seeds the initial Gaussians (`--load-3D-points` defaults True) |
| `transforms.json` | 244 KB | camera poses + OPENCV intrinsics |
| `run.json`, `report.json` | 8 KB | provenance; not read by training |
| ~~`colmap/`~~ | 456 MB | **excluded** -- SIFT database and sparse model, never read by training |

**Leaner option (~50 MB):** `images/` is only opened to pick the downscale
factor. Pass `--downscale-factor 2` explicitly and you can skip both `images/`
and `images_4|8/`, uploading just `images_2/`, `transforms.json` and
`sparse_pc.ply`. Not the default here because it trades a robust path for
~150 MB on a one-time transfer, and it silently breaks if you later want to
train at full resolution.

## What to expect

- 280 images at 1280x720 (the dataparser downscales to `images_2/` on its own,
  since it targets a max dimension under 1600 px and these are 2560 wide).
- 30k iterations of splatfacto on one A100: roughly 20-40 minutes. The
  `--time=02:00:00` request is deliberately loose.
- Output: `$CARC_OUT/interior400/splatfacto/<LABEL>/` with `config.yml`,
  `nerfstudio_models/`, `dataparser_transforms.json`, the tfevents file,
  `export/splat.ply` and `eval/`.

## Viewing -- read before planning Phase 3

**`ns-viewer` will not run on the Apple Silicon machine.** It renders through
gsplat, which disables itself without CUDA. The same applies to `ns-export`,
which loads the model onto a CUDA device. This is why `train.job` runs the
export **on CARC**, immediately after training.

So the local artifact is `export/splat.ply`, viewed in any WebGL splat viewer
in the browser -- superspl.at/editor, antimatter15.com/splat, or PlayCanvas.
No CUDA, no install.

If you want the interactive nerfstudio viewer, it has to run on the CARC GPU
node with an SSH tunnel back (`ns-viewer --load-config ...` plus
`ssh -L 7007:<node>:7007`), which is more setup than the .ply is worth for a
first look.

## Likely failure modes

| Symptom | Cause |
|---|---|
| `no GPU visible` from the assert in `train.job` | `--gres` wrong or partition has no GPU |
| gsplat CUDA compile error mid-job | `module load` differs between `setup_env.sh` and `train.job` |
| `ModuleNotFoundError: pymeshlab` at the export step | `setup_env.sh` not re-run after it was added |
| Job pends a long time | a100 contention; try a different `--gres` type from the preflight table |
| `Could not find any image files` | dataset not at `/home1/illyhoan/arc/splatting/data/processed/<RUN>/`, or (optional path) `stage.sh` was run before the dataset finished processing |
