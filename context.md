# Project context

## Goal
Extract pixel positions of combatants from BattleBots arena footage, track them
across frames, and emit arena-coordinate state (position/heading/velocity) to a
separate autonomous-driver system for a physically built combatant robot
("Nemesis"). Target is real-time operation off a live arena camera; current
work builds/validates the pipeline against recorded footage in `data/`.

## Camera / footage
- Single fixed, elevated, wide/fisheye-lens camera looking down into the
  arena. Visible barrel distortion on floor grid lines.
- 1440x762, ~98 fps (4899/50), h264/yuv420p.
- Grayscale-like, noisy, low-light. Gain (9-12) and exposure (7-10ms) are
  manually tuned per fight and are NOT consistent across clips (see
  `data/*_notes.txt`).
- `fight2.avi` and `fight4.avi` are referenced in the notes but not present in
  `data/`. Only `fight1.avi`, `fight3.avi`, `fight5.mp4` exist.
- Known per-clip issues from notes: fight3 has particle/spark clutter ("slot
  machine"), fight5 has smoke haze building on the box.
- No camera calibration data (no checkerboard shots) exists yet. Homography
  from arena floor markings (panel grid, center cross, BATTLEBOTS logo) is
  used as an interim substitute for full lens undistortion.

## Status
- Step 1 (frame source abstraction) done: `capture/source.py`
  (`VideoFileSource` working, `LiveCameraSource` stubbed pending SDK info),
  validated against all 3 clips via `scripts/replay.py`. Decode throughput
  (~590-745 fps) is well above the ~98 fps source rate, so raw I/O isn't the
  real-time bottleneck.
- Step 2 (calibration) not started.

## Architecture (see AGENTS.md for module layout/conventions)
1. Frame source abstraction (video file for dev, live camera later)
2. Calibration: pixel -> arena-floor homography
3. Preprocessing: CLAHE (handles per-fight gain/exposure variance), light
   denoise
4. Detection: background subtraction (fixed camera) + blob filtering
5. Tracking: Kalman + centroid/IoU association, survives blob-merge on
   collision
6. Identity: Nemesis vs. opponent -- **undecided, see Open decisions**
7. Output: UDP/JSON, ~30Hz, versioned schema (no existing driver-side
   interface to match yet)

## Assumptions (unconfirmed)
- Vision runs off-board on a ground-station machine, streaming state to
  Nemesis's onboard driver over a wireless link.
- Live camera is a GenICam/USB3-class machine-vision unit with
  software-controllable gain/exposure (consistent with per-fight notes).
  Exact SDK unknown until integration.

## Open decisions
- **Identity (step 6):** physical fiducial marker (ArUco/AprilTag) on Nemesis
  vs. track-continuity-only (seed Nemesis's track from known start position,
  carry ID via tracking). Marker recommended for real-time reliability;
  requires a hardware change. Not yet decided as of first implementation
  pass -- generic (unlabeled) multi-blob tracking is being built first.
- Full fisheye lens calibration deferred until checkerboard shots can be
  taken with the actual camera.
- Live camera SDK/protocol unknown.
- Driver-side input format is being defined by this project (see step 7);
  no existing spec to match.
