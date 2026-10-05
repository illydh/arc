#!/usr/bin/env bash
# CARC -> Mac, after a job. Brings down a run's images, labels and report,
# and the coloured plates and cut-out sheets it was made from, so they can be
# looked at here.
set -euo pipefail

CARC="illyhoan@discovery.usc.edu"
SRC="/home1/illyhoan/arc/augment"
RUN="${1:?usage: fetch.sh <run-name>}"
AUGMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$AUGMENT/out/$RUN"
echo "==> $CARC:$SRC/out/$RUN -> $AUGMENT/out/$RUN"
rsync -avh "$CARC:$SRC/out/$RUN/" "$AUGMENT/out/$RUN/"
echo "==> coloured plates and cut-outs -> $AUGMENT/data"
rsync -avh "$CARC:$SRC/data/plates" "$AUGMENT/data/"
rsync -avh "$CARC:$SRC/data/sprites/cutouts" "$AUGMENT/data/sprites/"

echo
echo "report:  $AUGMENT/out/$RUN/report/report.json"
echo "sheets:  $AUGMENT/out/$RUN/report/sheet_*.jpg and plates.jpg"
