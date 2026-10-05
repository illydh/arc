#!/usr/bin/env bash
# The one command for run 2 of interior400. From a CARC terminal, any cwd:
#   cd /home1/illyhoan/arc && git pull && bash splatting/scripts/carc/submit_run2.sh
# Submits two parallel jobs. Each trains, exports splat.ply and scores the
# held-out views, so nothing else has to be run on CARC.
# Both turn on three settings that were off in run 1 (see mds/context.md,
# Phase 3 findings):
#   camera-optimizer SO3xR3     lets training adjust each photo's pose
#   use-bilateral-grid          absorbs per-photo exposure and colour changes
#   use-scale-regularization    penalises needle-shaped splats
# Two of them act on training photos only, which can lower held-out scores
# without the result being worse: the bilateral grid leaves a held-out render
# free to differ from its photo in overall colour, and pose refinement leaves
# it slightly off its photo's pose. color-corrected-metrics adds
# cc_psnr/cc_ssim/cc_lpips to the eval output, which discount the first.
#   run2a  half size (1280x720) like run 1, so the two compare directly
#   run2b  full size (2560x1440); skipped if any full-size frame is missing
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA="/home1/illyhoan/arc/splatting/data/processed/interior400"

# run2b reads the full-size frames, which run 1 never needed. Check every
# frame transforms.json names before queueing a job that would die on one.
total=0
missing=0
while read -r frame; do
  total=$((total + 1))
  [ -f "$DATA/$frame" ] || missing=$((missing + 1))
done < <(grep -o '"images/[^"]*"' "$DATA/transforms.json" | tr -d '"')
if [ "$total" -eq 0 ]; then
  echo "no frames listed in $DATA/transforms.json" >&2
  exit 1
fi

SETTINGS="--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.use-bilateral-grid True \
--pipeline.model.use-scale-regularization True \
--pipeline.model.color-corrected-metrics True"

EXTRA="$SETTINGS" \
  sbatch --job-name=run2a --export=ALL,RUN=interior400,LABEL=run2a train.job
if [ "$missing" -eq 0 ]; then
  EXTRA="$SETTINGS nerfstudio-data --downscale-factor 1" \
    sbatch --job-name=run2b --export=ALL,RUN=interior400,LABEL=run2b train.job
else
  echo "run2b not submitted: $missing of $total full-size frames missing under $DATA/images" >&2
fi
