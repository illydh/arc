#!/usr/bin/env bash
# Mac -> CARC, before the first job. Copies the inputs that git does not
# carry (augment/data/ is git-ignored): about 75 MB. The code itself reaches
# CARC through `git pull` in the checkout at /home1/illyhoan/arc.
#
# Of the fight frames only the labelled ones go up, with their outlines,
# plus fight1's hand-picked Kraken frames: they are what the robot cut-outs
# are made from. Coloured cut-outs are not
# copied: CARC colours its own with the full model.
set -euo pipefail

CARC="illyhoan@discovery.usc.edu"
DEST="/home1/illyhoan/arc/augment/data"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$(cd "$HERE/../../data" && pwd)"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
"$HERE/../../.venv/bin/python" - "$DATA" "$HERE/.." > "$LIST" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
import robots
for clip in ("fight1", *robots.CLIPS):
    print(f"real/{clip}/plate.png")
    print(f"real/{clip}/manifest.json")
    frames = {c["frame"] for c in robots.cutouts(clip)}
    frames |= {f["frame"] for f in json.load(open(f"{sys.argv[1]}/real/{clip}/manifest.json"))["items"]
               if f["accepted"]}
    for frame in sorted(frames):
        print(f"real/{clip}/images/{frame:06d}.jpg")
        print(f"real/{clip}/masks/{frame:06d}.png")
PY

ssh "$CARC" "mkdir -p '$DEST'"
echo "==> labelled frames and plates -> $CARC:$DEST/real"
rsync -avh --files-from="$LIST" "$DATA/" "$CARC:$DEST/"
echo "==> references, mesh, detector, approved plate -> $CARC:$DEST"
rsync -avh "$DATA/refs" "$DATA/meshes" "$DATA/detector" "$DATA/plates" "$CARC:$DEST/"

cat <<MSG

staged. next, in a CARC terminal:
  cd /home1/illyhoan/arc && git pull
  cd augment/scripts/carc
  bash setup_env.sh                                                   # once only
  sbatch --export=ALL,RUN=smoke,N=50,SHEETS=1,SEEDS=3 generate.job    # smoke test
MSG
