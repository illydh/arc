#!/bin/bash

#SBATCH --job-name=splatfacto
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=illyhoan@usc.edu

# Submit with the run name and paths exported:
#   sbatch --export=ALL,RUN=interior400 train.sbatch
# See mds/carc.md for the full export block and the CONFIRM_ items above.

set -euo pipefail

: "${RUN:?export RUN=<run-name>}"
: "${CARC_ENV:?export CARC_ENV}"
: "${CARC_DATA:?export CARC_DATA}"
: "${CARC_OUT:?export CARC_OUT}"
ITERS="${ITERS:-30000}"

# Must match setup_env.sh, or gsplat's runtime CUDA compile fails.
module purge
module load gcc/13.3.0 cuda/12.4.1 python/3.11.9
# shellcheck disable=SC1091
source "$CARC_ENV/bin/activate"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('gpu:', torch.cuda.get_device_name(0))"

# --vis tensorboard, not viewer: there is no browser reachable from a compute
# node, and the default viewer blocks waiting for one.
# No --downscale-factor: the dataparser already picks images_2/ on its own,
# because it downscales until the max dimension is under 1600px and these
# frames are 2560 wide.
ns-train splatfacto \
  --data "$CARC_DATA/$RUN" \
  --output-dir "$CARC_OUT" \
  --experiment-name "$RUN" \
  --vis tensorboard \
  --max-num-iterations "$ITERS" \
  --viewer.quit-on-train-completion True

# Export the .ply here rather than locally: ns-viewer and ns-export both load
# the model onto a CUDA device, so neither runs on the Apple Silicon box. A
# .ply opens in any WebGL splat viewer with no CUDA at all.
CONFIG="$(ls -t "$CARC_OUT/$RUN"/*/config.yml | head -1)"
echo "exporting from $CONFIG"
ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$CARC_OUT/$RUN/export"

echo "done; checkpoints under $CARC_OUT/$RUN, splat.ply under $CARC_OUT/$RUN/export"
