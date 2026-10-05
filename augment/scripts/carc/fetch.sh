#!/usr/bin/env bash
# CARC -> Mac, after a job. Brings down a run: images, labels and report.
# The report folder also holds every plate the run used (plates.jpg) and the
# coloured cut-out sheets (cutouts/), so one rsync, and one CARC login, is
# enough to look at everything here.
set -euo pipefail

CARC="illyhoan@discovery.usc.edu"
SRC="/home1/illyhoan/arc/augment/out"
RUN="${1:?usage: fetch.sh <run-name>}"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/out/$RUN"

mkdir -p "$OUT"
echo "==> $CARC:$SRC/$RUN -> $OUT"
rsync -ah "$CARC:$SRC/$RUN/" "$OUT/"

echo
echo "verdict:  $(grep -m1 '"verdict"' "$OUT/report/report.json" 2>/dev/null || echo 'no report.json -- read the job log')"
echo "report:   $OUT/report/report.json"
echo "look at:  $OUT/report/plates.jpg, sheet_*.jpg and cutouts/"
