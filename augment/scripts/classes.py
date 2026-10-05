# The detector is being retrained with two classes, Nemesis and every other
# robot. These ids are written into every label file and match the order of
# `names` in the detector's data.yaml (confirmed by the user, 2026-10-04).
# If that order ever changes, change it here and re-run
# `extract.py <clip> --review` and `compose.py`.
NEMESIS, NON_NEMESIS = 0, 1
NAMES = {NEMESIS: "nemesis", NON_NEMESIS: "non-nemesis"}
