#!/usr/bin/env bash
# Push a processed run AND the carc scripts to CARC.
#
# The scripts have to go up too: setup_env.sh and train.sbatch run on the
# cluster, and sbatch does not exist locally. They live under CARC_CODE in
# /home1 rather than with the data in /scratch1, which is purged periodically.
#
# colmap/ is excluded from the dataset: it is ~70% of the run's bytes (the
# SIFT database) and training never reads it -- transforms.json and
# sparse_pc.ply already carry the poses and the seed point cloud.
#
# images/ IS uploaded despite training reading images_2/: the dataparser opens
# the full-resolution file to measure dimensions before picking a downscale.
# Skipping it requires passing --downscale-factor 2 explicitly (see carc.md).
set -euo pipefail

: "${CARC_USER:?set CARC_USER (see mds/carc.md)}"
: "${CARC_HOST:=discovery.usc.edu}"
: "${CARC_DATA:?set CARC_DATA, e.g. /scratch1/\$CARC_USER/splatting/data}"
: "${CARC_CODE:?set CARC_CODE, e.g. /home1/\$CARC_USER/splatting}"

RUN="${1:?usage: stage.sh <run-name>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$(cd "$HERE/../../data/processed/$RUN" && pwd)"

echo "==> scripts -> $CARC_HOST:$CARC_CODE/scripts/carc"
ssh "$CARC_USER@$CARC_HOST" "mkdir -p '$CARC_CODE/scripts/carc' '$CARC_DATA/$RUN'"
rsync -avh "$HERE/" "$CARC_USER@$CARC_HOST:$CARC_CODE/scripts/carc/"

echo "==> dataset $RUN -> $CARC_HOST:$CARC_DATA/$RUN"
rsync -avh --progress --exclude 'colmap/' \
  "$LOCAL/" "$CARC_USER@$CARC_HOST:$CARC_DATA/$RUN/"

cat <<EOF

staged. next, on the cluster:
  ssh $CARC_USER@$CARC_HOST
  cd $CARC_CODE/scripts/carc
  bash setup_env.sh                                     # once only
  sbatch --export=ALL,RUN=$RUN,ITERS=500 train.sbatch   # smoke test
EOF
