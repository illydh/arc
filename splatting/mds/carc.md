# Phase 2 runbook -- training on USC CARC

Scripts live in `scripts/carc/`. Everything is driven by environment
variables so no username or account name is committed.

Nothing in here has been executed -- there is no CARC access from the machine
this was written on. Items marked **CONFIRM** are cluster facts that have to
be checked on the login node before the first run; they are the likely
failure points, not afterthoughts.

## Config block

Export these in your shell (or drop them in `~/.splatting_carc` and source
it). `CARC_HOST` defaults to `discovery.usc.edu` if unset.

```sh
export CARC_USER=<your-carc-username>
export CARC_HOST=discovery.usc.edu
export CARC_DATA=/scratch1/$CARC_USER/splatting/data
export CARC_OUT=/scratch1/$CARC_USER/splatting/outputs
export CARC_ENV=/home1/$CARC_USER/envs/splatting
export CARC_CODE=/home1/$CARC_USER/splatting
```

`CARC_ENV` and `CARC_CODE` go in `/home1` (persistent); data and outputs go in
`/scratch1` (fast, purged periodically). **CONFIRM** both paths exist for your
account and that scratch has ~1 GB free -- `myquota`.

Set the same block again in your shell **on the login node** -- `train.sbatch`
reads `CARC_ENV`, `CARC_DATA` and `CARC_OUT` from the submitting environment
via `--export=ALL`, and will refuse to start without them.

## Preflight

Run on the login node and fill the results into `train.sbatch`:

| What | Command | Goes into |
|---|---|---|
| Slurm account | `myaccount`, or `sacctmgr show assoc user=$USER format=account,partition` | `--account=` |
| GPU partition | `sinfo -s` | `--partition=` |
| GPU types available | `sinfo -o "%P %G %D"` | `--gres=gpu:<type>:1` |
| Module names | `module avail cuda`, `module avail python`, `module avail gcc` | both scripts' `module load` lines |

The `module load` line appears in **both** `setup_env.sh` and `train.sbatch`
and the two must match exactly. gsplat compiles its CUDA kernels at first use
inside the job, against whatever toolkit is loaded then -- a version skew
between install time and run time surfaces as a compile error mid-job, not at
install.

## Sequence

Note which machine each step runs on. `sbatch` exists only on the cluster.

```sh
### ON THE LAPTOP -- uploads the dataset (~198 MB) and the scripts
./splatting/scripts/carc/stage.sh interior400

### ON THE LOGIN NODE
ssh $CARC_USER@$CARC_HOST
# re-export the config block here, then:
cd $CARC_CODE/scripts/carc
bash setup_env.sh                                    # once only, ~10 min

# smoke test: ~2 min, proves GPU + env + data path before a real run
sbatch --export=ALL,RUN=interior400,ITERS=500 train.sbatch
squeue -u $USER
tail -f splatfacto-<jobid>.out

# the real run, once the smoke test exits 0
sbatch --export=ALL,RUN=interior400 train.sbatch

### ON THE LAPTOP -- once the job finishes
./splatting/scripts/carc/fetch.sh interior400
```

## What actually gets uploaded

`stage.sh` sends two things.

**1. The scripts** -> `$CARC_CODE/scripts/carc/` (~4 KB): `setup_env.sh`,
`train.sbatch`, `stage.sh`, `fetch.sh`. Only the first two are used there.

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
- Output: `$CARC_OUT/interior400/<timestamp>/` with `config.yml`, `nerfstudio_models/`,
  and `export/splat.ply`.

## Viewing -- read before planning Phase 3

**`ns-viewer` will not run on the Apple Silicon machine.** It renders through
gsplat, which disables itself without CUDA. The same applies to `ns-export`,
which loads the model onto a CUDA device. This is why `train.sbatch` runs the
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
| `no GPU visible` from the assert in `train.sbatch` | `--gres` wrong or partition has no GPU |
| gsplat CUDA compile error mid-job | `module load` differs between `setup_env.sh` and `train.sbatch` |
| `ModuleNotFoundError: pymeshlab` at the export step | `setup_env.sh` not re-run after it was added |
| Job pends a long time | a100 contention; try a different `--gres` type from the preflight table |
| `Could not find any image files` | `stage.sh` was run before the dataset finished processing, or `CARC_DATA` mismatch |
