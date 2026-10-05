#!/usr/bin/env bash
# Step 1 of 2 on CARC (see generate.job for step 2). Run once, on a LOGIN
# node, from a checkout at /home1/illyhoan/arc. About 15 minutes, most of it
# the 16 GB model download.
set -euo pipefail

# This venv lives only on CARC and is separate from splatting's. Must match
# the path generate.job activates.
CARC_ENV="/home1/illyhoan/arc/augment/.venv"
# Model weights go to scratch: 16 GB is a sixth of the /home1 quota, and they
# can be downloaded again if scratch is purged. Must match generate.job.
export HF_HOME="/scratch1/illyhoan/hf"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /home1/illyhoan/arc/logs "$HF_HOME"

# A login node allows each user 64 processes, threads included. Left alone,
# numpy's OpenBLAS starts a thread per core (32), is refused part-way and
# interrupts its own process: the import check below dies with
# KeyboardInterrupt.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# The module line splatting uses, known to load together on this cluster.
module purge
module load gcc/12.3.0 cuda/12.4.1 python/3.11.9

python -m venv "$CARC_ENV"
# shellcheck disable=SC1091
source "$CARC_ENV/bin/activate"
pip install --upgrade pip

# diffusers 0.40.0 needs torch >= 2.6, and cu124 pairs with cuda/12.4.x.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# requirements.txt is pinned for the Mac (Python 3.12). Its torch lines are
# the Mac builds, and its numpy needs Python 3.12 while this module is 3.11,
# so those three are left out and pip picks a numpy that fits.
grep -vE '^(torch|torchvision|numpy)==' "$HERE/../../requirements.txt" | pip install -r /dev/stdin

# Fetched here because compute jobs run with the hub offline, so that no job
# stalls on a download. The repo also holds a single-file copy of the
# transformer (7.8 GB) that diffusers does not read.
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("black-forest-labs/FLUX.2-klein-4B",
                        ignore_patterns=["flux-2-klein-4b.safetensors", "*.jpg"]))
PY

# Ultralytics brings opencv-python, which wants libGL. If the import below
# fails on that, swap it for the headless build -- never install both:
#   pip uninstall -y opencv-python && pip install opencv-python-headless
python - <<'PY'
import cv2, diffusers, torch, ultralytics
from diffusers import Flux2KleinPipeline
# The report checks that the set loads in Ultralytics, and that check fetches
# a font the first time. Fetched here, not in the middle of a job.
from ultralytics.utils.checks import check_font
check_font("Arial.ttf")
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
print("diffusers", diffusers.__version__, "| ultralytics", ultralytics.__version__, "| cv2", cv2.__version__)
print("NOTE: torch.cuda.is_available() is False on a login node -- that is")
print("expected. generate.job checks it on the GPU node.")
PY
