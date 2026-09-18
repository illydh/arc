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
- All five clips are present as `data/Real Fights/fight{1..5}.avi` (fight5 is
  `.avi`, not `.mp4` as previously recorded).
- The arena lights are OFF for roughly the first half of every clip -- mean
  frame brightness sits near 2-20 and jumps to ~85-115 when they come up.
  Detection only works on the lit segment, so every entry point has to seek
  into it first. Approximate first lit frame: fight1 ~23.9k, fight2 ~12.6k,
  fight3 ~13.5k, fight4 ~11.5k, fight5 ~11.1k.
- Known per-clip issues from notes: fight3 has particle/spark clutter ("slot
  machine"), fight5 has smoke haze building on the box. Both produce
  foreground blobs several times a robot's footprint, which is why detection
  association scores blob area, not just distance.
- No camera calibration data (no checkerboard shots) exists yet. Homography
  from arena floor markings (panel grid, center cross, BATTLEBOTS logo) is
  used as an interim substitute for full lens undistortion.

## Status
End-to-end two-robot tracking runs: `scripts/track_fight.py <clip> [out.csv]`.
Flow is seek to the lit segment -> warm up the background model -> click home
(Nemesis) then opponent -> track until `q`, writing a CSV and driving a
top-down arena window.

- Step 1 (frame source abstraction) done: `capture/source.py`
  (`VideoFileSource` working, `LiveCameraSource` stubbed pending SDK info).
  Decode throughput (~590-745 fps) is well above the ~98 fps source rate, so
  raw I/O isn't the real-time bottleneck.
- Step 2 (calibration) tooling built: `calib/homography.py`
  (compute/apply/save/load a pixel<->world homography) and
  `scripts/calibrate.py` (interactive point-picker). `track_fight.py` also
  quick-calibrates inline when `calib/homography.json` is absent: click four
  floor-grid intersections bounding a whole number of tiles, enter the tile
  span, and it saves the same file. Still not run against real footage with
  measured references -- no committed `calib/homography.json` exists, and one
  must not be generated from eyeballed points.
- Steps 3-5 (preprocessing/detection/tracking) done as one pass:
  `detect/blobs.py` (MOG2 + arena-ROI mask + morphology + area filter) and
  `track/tracker.py` (two fixed-identity Kalman tracks, greedy gated
  association scored on distance + blob intensity + blob area). CLAHE turned
  out to be unnecessary -- MOG2 absorbs the per-fight gain/exposure
  differences on its own, so it was never added.
- Step 7 (UDP/JSON transport) not started; `transport/` is still empty.

### Measured behaviour on the real clips
Validated on lit segments of fight1/fight3/fight5 (1500 frames each):
116-133 fps end to end at full 1440x762 including both display windows and
CSV writing, against a 98 fps source -- real-time holds with headroom. Tracks
hold a genuine detection 62-88% of frames, recover via template match for
most of the rest, and fully coast (no measurement at all) 0-4%.

## Architecture (see AGENTS.md for module layout/conventions)
1. Frame source abstraction (video file for dev, live camera later)
2. Calibration: pixel -> arena-floor homography
3. Preprocessing: none. MOG2 handles the per-fight gain/exposure variance
   directly; CLAHE was planned but proved unnecessary.
4. Detection: MOG2 background subtraction (fixed camera), masked to the arena
   floor, + blob area filtering
5. Tracking: two fixed-identity Kalman tracks, greedy gated association,
   template re-acquisition, clinch handling (see below)
6. Identity: assigned by the user at session start (click home, then
   opponent) and carried by track continuity -- see Open decisions for the
   unresolved reliability question
7. Output: CSV per session (both robots on one row per frame) + a top-down
   arena window. UDP/JSON to the driver system still to come.

## Assumptions (unconfirmed)
- Vision runs off-board on a ground-station machine, streaming state to
  Nemesis's onboard driver over a wireless link.
- Live camera is a GenICam/USB3-class machine-vision unit with
  software-controllable gain/exposure (consistent with per-fight notes).
  Exact SDK unknown until integration.

## Open decisions
- **Identity (step 6):** still open, but no longer blocking. Implemented
  today as user-assigned + track continuity: the operator clicks home
  (Nemesis) then opponent during setup and the two tracks keep those labels
  for the session. That is enough to run, and it removes the need for
  unlabeled multi-blob tracking. What remains undecided is whether to add a
  physical fiducial (ArUco/AprilTag) on Nemesis. The argument for it is
  concrete: identity here rests on appearance (blob mean intensity and
  footprint area) plus motion continuity, so a long clinch where both robots
  are inside one blob and then separate can hand back the labels swapped,
  with nothing in the pipeline able to notice or correct it. A marker fixes
  that outright but requires a hardware change.
- Full fisheye lens calibration deferred until checkerboard shots can be
  taken with the actual camera.
- Live camera SDK/protocol unknown.
- Driver-side input format is being defined by this project (see step 7);
  no existing spec to match.
- ~~Homography reference measurements needed~~ resolved: arena play surface
  is 48ft x 48ft, laid out as a 12x12 grid of 4ft x 4ft tiles (confirmed by
  user, not measured/assumed by this project). In meters: 14.6304m overall,
  1.2192m per tile. Grid line intersections are the natural
  `scripts/calibrate.py` reference points -- visible in every clip, and
  world coordinates are just `(i * 1.2192, j * 1.2192)` for grid intersection
  `(i, j)` relative to whichever corner/center is chosen as the origin.
  `scripts/calibrate.py` auto-sets the first click as origin (0,0), then
  prompts each subsequent point for its tile offset (di,dj) from the
  PREVIOUS point (not from origin) and accumulates -- easier to eyeball
  against adjacent grid lines than counting back to a fixed origin every
  time. No manual meter math required.
- Bottom-left and bottom-right arena corners are occluded in-frame (not just
  inconvenient to click) -- omitted from calibration, using other visible
  grid intersections instead. Consequence: position estimates for combatants
  near those corners are extrapolated beyond the calibration point cloud and
  are inherently less accurate until lens undistortion is added or the
  corners become visible from a different angle.

## Tracking design notes
Behaviour that was derived from the footage rather than chosen up front, and
that is expensive to re-derive:

- **A stationary robot is invisible to background subtraction.** MOG2 absorbs
  anything that stops moving, and robots do stop (pinned, immobilized, or
  staged pre-match). Motion alone held only ~1.4 of the 2 robots per frame.
  Each track therefore keeps a grayscale template from its last real
  detection and falls back to `matchTemplate` around the prediction.
- **The template needs a leash.** Searching around a free-running prediction
  lets a lost track random-walk across the floor and lock onto floor texture
  permanently. Matches are rejected beyond `ANCHOR_RADIUS` of the last real
  detection, and outside the arena mask.
- **A stale track must be able to reach across the arena.** There are only
  ever two robots, so an unclaimed robot-sized blob is almost certainly the
  missing one. The association gate widens without limit while a track has no
  real detection. This is what recovers a track that has stranded on scenery;
  capping the gate left it stuck permanently. The healthy track still claims
  its own blob first because assignment takes the globally cheapest pair.
- **Clinches are handled explicitly.** When one robot's blob swallows the
  other, the unassigned track follows the merged blob weakly instead of
  template-searching, which otherwise walks it off onto the floor.
- **Smoke and debris are robot-sized in area terms but not in shape.** Blobs
  run to 20k-44k px in fight3/fight5 versus 3k-9k for a robot, so
  association scores blob area against a per-track running average.

Known weak spots, in rough priority order:
1. Identity can swap after a long clinch (see Open decisions).
2. `arena_mask` is a convex hull of the largest bright region, so it includes
   some of the bottom wall/barrier; a track can strand there before the
   widening gate pulls it back.
3. Thresholds (`min_area`/`max_area`, `TEMPLATE_SCORE`, gate) were tuned on
   fight1/fight3/fight5 lit segments at 1440x762 and are resolution- and
   arena-specific.
