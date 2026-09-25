#!/usr/bin/env bash
# Pull a finished training run back for local viewing.
set -euo pipefail

: "${CARC_USER:?set CARC_USER (see mds/carc.md)}"
: "${CARC_HOST:=discovery.usc.edu}"
: "${CARC_OUT:?set CARC_OUT, e.g. /scratch1/\$CARC_USER/splatting/outputs}"

RUN="${1:?usage: fetch.sh <run-name>}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/outputs/$RUN"

mkdir -p "$DEST"
echo "fetching $CARC_HOST:$CARC_OUT/$RUN -> $DEST"
rsync -avh --progress "$CARC_USER@$CARC_HOST:$CARC_OUT/$RUN/" "$DEST/"

echo
echo "view the exported splat at $DEST/export/*.ply"
echo "Drop it into a WebGL viewer (e.g. superspl.at/editor or antimatter15.com/splat)."
echo "ns-viewer will NOT work on this machine -- it renders through gsplat, which needs CUDA."
