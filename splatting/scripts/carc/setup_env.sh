#!/usr/bin/env bash
# One-time environment build on CARC. Run this on a LOGIN node.
#
# No GPU is needed to install. gsplat JIT-compiles its CUDA kernels on first
# use inside the job, which is why train.sbatch loads the same cuda module --
# if the module differs between here and there, the compile fails at runtime.
set -euo pipefail

: "${CARC_ENV:?set CARC_ENV, e.g. /home1/\$USER/envs/splatting}"

# CONFIRM these against `module avail` -- names and versions are cluster- and
# date-specific, and a mismatch here is the most likely thing to break.
module purge
module load gcc/13.3.0 cuda/12.4.1 python/3.11.9

python -m venv "$CARC_ENV"
# shellcheck disable=SC1091
source "$CARC_ENV/bin/activate"
pip install --upgrade pip

# The torch CUDA build must match the cuda module loaded above and in
# train.sbatch. cu124 pairs with cuda/12.4.x.
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install nerfstudio==1.1.5
# ns-export imports pymeshlab at module load, so the export step needs it even
# though a Gaussian-splat export touches no mesh code.
pip install pymeshlab

python - <<'EOF'
import torch, gsplat, nerfstudio
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
print("gsplat", gsplat.__version__)
print("nerfstudio ok")
print("NOTE: torch.cuda.is_available() is False on a login node -- that is")
print("expected. train.sbatch checks it on the GPU node.")
EOF
