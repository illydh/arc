# Agent instructions

See `context.md` for project background, data caveats, and open decisions.

## Stack
Python 3.10, OpenCV (`cv2`), NumPy. No framework (no ROS) unless a future
requirement forces it. Keep dependencies minimal.

## Layout
```
capture/   frame source abstraction (video file + live camera)
calib/     pixel -> arena-floor homography, undistortion
detect/    background subtraction + blob filtering
track/     Kalman filter / detection association
identify/  Nemesis-vs-opponent identity (undecided, see context.md)
transport/ UDP/JSON output to the driver system
scripts/   dev harnesses for running the pipeline against data/ clips
```
Modules are plain Python packages (`__init__.py` + files), no plugin
framework or config-driven registries -- there are only two frame sources and
one pipeline, so a factory/registry layer is unneeded indirection.

## Conventions
- Minimal, direct implementations. No speculative abstraction, no config
  options for cases that don't exist yet (e.g. don't add a second detection
  backend until there's a second detection backend).
- No docstrings/comments unless they explain a non-obvious constraint (e.g.
  why a threshold value was chosen from footage, a workaround for a specific
  clip's noise characteristics).
- Validate against the real clips in `data/` (`fight1.avi`, `fight3.avi`,
  `fight5.mp4`), not synthetic data, since the noise/lighting characteristics
  are the actual hard part of this problem.

## Running
Dependencies are listed in `requirements.txt` (`pip install -r
requirements.txt`). Append new dependencies there as they're introduced --
don't let the manifest drift from what the code actually imports.

Dev harnesses live in `scripts/` and take a path into `data/` as an argument.

## Keeping context current
`context.md` and this file are working documents, not one-time setup.
Update them whenever:
- scope changes (a step in the plan is added, dropped, or reordered)
- an open decision in `context.md` gets resolved (e.g. the identity approach)
- a pipeline stage is completed or its design changes from what's documented
- a new assumption is made that a future session would otherwise have to
  re-derive
Stale context is worse than missing context -- don't leave a resolved "open
decision" or an outdated architecture description in place.
