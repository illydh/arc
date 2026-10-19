#!/usr/bin/env bash
# Mac -> CARC, before the first job, AFTER `git pull` on CARC has created
# /home1/illyhoan/arc/augment. Copies the inputs that git does not carry
# (augment/data/ is git-ignored): about 80 MB, in one rsync so that CARC asks
# for a login once.
#
# Of the fight frames only the labelled ones go up, with their outlines,
# plus fight1's hand-picked Kraken frames: they are what the robot cut-outs
# are made from. Coloured cut-outs are not copied: CARC colours its own with
# the full model. plates/ carries fight4's plate with seed 3, the one approved
# by eye on the Mac -- anything else left in it would be composed on too.
set -euo pipefail

CARC="illyhoan@discovery.usc.edu"
DEST="/home1/illyhoan/arc/augment/data"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$(cd "$HERE/../../data" && pwd)"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

# `stage.sh combat` writes one archive of what the combat set needs on top of
# what is already on CARC: the reviewed combat frames with their outlines and
# the pair tracks (combat.py), the clips' manifests the single poses are found
# from (so CARC searches the same frames), and the SAM 2.1 checkpoint that cuts
# the new robots out (a symlink here, so it goes up as the file it points to).
# Unpack it on CARC with:
#   tar -xzf carc-combat-inputs.tar.gz -C /home1/illyhoan/arc/augment
if [ "${1:-}" = "combat" ]; then
  "$HERE/../../.venv/bin/python" - "$HERE/.." > "$LIST" 2>/dev/null <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import combat
data = combat.REAL.parent
files = {"weights/sam2.1_s.pt"}
for clip in combat.CLIPS_USED:
    files |= {f"real/{clip}/combat/tracks.json", f"real/{clip}/manifest.json"}
    for c in combat.poses(clip):
        d = Path(c["dir"]).relative_to(data)
        files |= {f"{d}/images/{c['frame']:06d}.jpg", f"{d}/masks/{c['frame']:06d}.png"}
print("\n".join(sorted(files)))
PY
  mkdir -p "$DATA/../out"
  sed 's|^|data/|' "$LIST" | tar -czhf "$DATA/../out/carc-combat-inputs.tar.gz" -C "$DATA/.." -T -
  echo "wrote $(cd "$DATA/../out" && pwd)/carc-combat-inputs.tar.gz ($(wc -l < "$LIST" | tr -d ' ') files)"
  exit 0
fi
"$HERE/../../.venv/bin/python" - "$DATA" "$HERE/.." > "$LIST" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import robots
data = Path(sys.argv[1])
for clip in ("fight1", *robots.CLIPS):
    print(f"real/{clip}/plate.png")
    print(f"real/{clip}/manifest.json")
    frames = {c["frame"] for c in robots.cutouts(clip)}
    frames |= {f["frame"] for f in json.loads((data / "real" / clip / "manifest.json").read_text())["items"]
               if f["accepted"]}
    for frame in sorted(frames):
        print(f"real/{clip}/images/{frame:06d}.jpg")
        print(f"real/{clip}/masks/{frame:06d}.png")
for folder in ("refs", "meshes", "detector", "plates"):
    for f in sorted((data / folder).rglob("*")):
        if f.is_file() and f.name != ".DS_Store":
            print(f.relative_to(data))
PY

# `stage.sh bundle` writes the same files to one archive instead, for copying
# to CARC by other means. Unpack it there with:
#   tar -xzf carc-inputs.tar.gz -C /home1/illyhoan/arc/augment
if [ "${1:-}" = "bundle" ]; then
  mkdir -p "$DATA/../out"
  sed 's|^|data/|' "$LIST" | tar -czf "$DATA/../out/carc-inputs.tar.gz" -C "$DATA/.." -T -
  echo "wrote $(cd "$DATA/../out" && pwd)/carc-inputs.tar.gz"
  exit 0
fi

echo "==> $(wc -l < "$LIST" | tr -d ' ') files -> $CARC:$DEST"
rsync -ah --files-from="$LIST" "$DATA/" "$CARC:$DEST/"

cat <<MSG

staged. next, in a CARC terminal:
  cd /home1/illyhoan/arc/augment/scripts/carc
  bash setup_env.sh                                                   # once only
  sbatch --export=ALL,RUN=smoke,N=50,SHEETS=1,SEEDS=3 generate.job    # smoke test
MSG
