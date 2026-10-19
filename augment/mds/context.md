# Augment task context

Generated colour images of robots fighting in the BattleBots arena, labelled,
to add to the robot detector's training data. Separate concern from `vision/`
and `splatting/` -- see below.

## Goal
1. A labelled set of generated colour images of robots in combat, in the
   arena, from the fight camera's viewpoint. The detector is being retrained
   with two classes, Nemesis and every other robot, and these images are for
   fine-tuning the second one (user, 2026-10-04).
2. A measured answer on whether adding them to the detector's training data
   improves detection on real fight frames (the "proof test").

## Relationship to `vision/` and `splatting/`
- `vision/` finds robots by MOG2 background subtraction, which cannot be
  trained. The trainable detector this task serves lives outside this repo
  (see The detector). `vision/`'s rule to validate on real clips and not on
  synthetic data still holds here: generated images are training data only,
  and the proof test scores on real frames.
- `splatting/` supplies colour reference stills of the arena. The splat itself
  is not used: it was shot at head height, has no overhead coverage, and
  renders only on CUDA.

## Inputs (measured 2026-10-04)

### Fight clips
`vision/data/Real Fights/fight{2..5}.avi`, 1440x762, ~98 fps. `fight1.avi` is
read in place from `~/Projects/nemesis-tracking/data/` (same format, 36,754
frames); it was never copied into this repo.
- True monochrome: both chroma planes are constant (U=V=127, `signalstats`
  SATMAX=1). There is no colour to recover from them.
- Nemesis is not in any of these clips (user, 2026-10-04). Every robot in
  them is someone else's.
- fight1 has three robots (two large, one minibot). The others have two.
- The clips are from BattleBots' Las Vegas live show, Destruct-A-Thon (see
  The robots in the clips).
- Seeking by frame index is exact on these files (checked against a
  sequential read on fight3).
- **The lights do not simply switch on.** A light show strobes for several
  hundred frames first. fight3, mean brightness by frame: 13250 55, 13500 127,
  13580 0.5, 13750 110, 14000 12, 14090 34, then steadily lit from about
  14500. A brightness threshold alone picks strobe frames.
- **The start squares are lit differently before the match starts.** Part of
  the square under a waiting robot differs from the plate in brightness but
  not in texture. So do shadows.

### The robots in the clips (identified 2026-10-04)
The clips were recorded at **BattleBots: Destruct-A-Thon**, the live show at
Caesars Entertainment Studios in Las Vegas. Its regulars are a fixed roster of
twelve "ShowBots", built for the show and heavier than competition robots:
HyperShock, Witch Doctor, Kraken, Tazbot, Malice, Chopper, Mammoth, Whiplash,
Diablo, OverKill, Ginsu and Nightmare. A thirteenth, the Slot Machine, is a
retired slot machine on half of Ginsu's chassis and exists to be destroyed by
Nightmare. Guests and try-out robots also appear on some nights. Source: the
BattleBots wiki (battlebots.fandom.com), pages "BattleBots: Destruct-A-Thon"
and "<name> (ShowBot)".

Matched by reading lettering in the footage and comparing shapes with the
wiki's roster pictures. No footage was uploaded to any service.

| Clip | Robot | Evidence | Confidence | Livery on the wiki |
|---|---|---|---|---|
| fight1 | Kraken | "KRAKEN" lettering readable (frame 35390); scaled body; flamethrower | High | Green scales, red jaw, white teeth, white wheel hubs |
| fight1 | Tazbot and its detachable minibot | Lifting arm locked with Kraken; a third small robot | Medium-high | Red body and wheels, white spikes, silver arm |
| fight2 | Mammoth | Tube-frame lifter, unique shape | High | Black tube frame |
| fight2 | Chopper or OverKill | Two-wheeled, bright metal body, thin overhead weapon, white hubs | Not settled | Chopper: silver wedge and axe. OverKill: black wedge, silver blade, white hubs |
| fight3 | Nightmare | Huge vertical disc on a wide two-wheel frame | High | Black frame, silver disc |
| fight3 | Slot Machine | `fight3_notes.txt` says so; the wiki says it always fights Nightmare | High | Dark cabinet with a lit panel, on Ginsu's pink chassis |
| fight4 | Whiplash | "WHIPLASH" lettering readable in the cut-outs | High | Black, neon-yellow wedges and wheel hubs |
| fight4 | Tazbot | Turret arm, ring of white spikes | High | As above |
| fight5 | HyperShock | Spotted bright body, front forks | High | Yellow with pink-red and black spots, yellow hubs |
| fight5 | Witch Doctor | Dark top, light side pods, exposed wheels, forks; HyperShock's usual opponent | Medium; Malice not excluded | Lime green and purple, bone graphics, green hubs |

What follows from it:
- The `non-nemesis` class at this venue is mostly a known, small roster, so
  real liveries matter more than invented variety.
- Roster robots with no footage here: Diablo, Ginsu, Malice, and whichever of
  Chopper and OverKill is not in fight2.
- Kraken is only in fight1, whose frames are not labelled. Its cut-outs are
  still usable: a cut-out needs a clean outline, not a complete frame.
- Tazbot's rear two wheels are a minibot that can detach, which is fight1's
  third robot and means fight4 may at times hold a third robot too.
- Liveries are not fixed. The wiki records a gold repaint for the 500th show
  and a 4 July livery.
- Several ShowBots carry flamethrowers (Kraken, Malice, Nightmare), which is
  the bright bloom in fight1.

### Arena colour reference
- `augment/data/refs/arena_broadcast_1.jpg` and `_2.jpg` (git-ignored): crops
  of two screenshots of public broadcast footage, supplied by the user on
  2026-10-04. The first looks down the arena the same way round as the fight
  camera, under fight lighting, and is the reference `colorize.py` uses. The
  second is a corner view. Used as a colour reference only; they stay out of
  git and out of the generated set.
- **A robot waits on each start square in the first one**: a blue one on the
  red square, a yellow and black one on the blue. `colorize.empty_reference`
  covers both with their square's colour before the model sees the picture
  (see CARC smoke run). The file itself is left as supplied.
- What the first one shows: red start square on the left, blue on the right;
  the centre logo is a red B panel and a blue B panel over white lettering;
  dark charcoal floor; yellow slot outlines; yellow and black bumpers.
- Paint measured in it (CIELAB a, b): red (+51, +39), blue (-7, -29), yellow
  (-2, +47), floor (+3, -5).
- `splatting/data/processed/interior400/images/`: 428 phone stills at eye
  level, shot under work lights. The same paint reads much duller there: red
  (+27, +28), blue (0, -10). They were the only reference until the
  screenshots arrived and are no longer used.

### The target camera
Teledyne FLIR Forge 5GigE **FG-P5G-51S4C-C** (user, 2026-10-04; first given
as `51S4M-C`, the monochrome variant, and corrected the same day).

| Field | Value |
|---|---|
| Colour | Yes |
| Sensor | Sony IMX537, 1/1.8", 2.74 um pixels, global shutter |
| Resolution | 2448 x 2048 |
| Frame rate | 122 fps, 260 fps in burst |
| Filter | IR cut at 670 nm |
| Mount, link | C-mount, 5 Gb Ethernet |

- Its frame is about 6:5. The fight clips, and every image made here, are
  1440x762. The lens, the mounting position and any cropping are not known,
  so the framing of the real images is still an assumption.
- It has an IR-cut filter. The clips' monochrome camera sees both start
  squares equally bright, which a filtered colour sensor will not.
- Its noise at fight gain is not known. No frame from it exists yet.

### The detector
`augment/data/detector/finetune_color.pt` (git-ignored). The same file is at
`~/Projects/nemesis-tracking/ckpts/`, where the tracker loads it.

Read from the checkpoint without executing it (every class replaced by an
inert stub; it references only `torch`, `ultralytics`, `collections` and
`set`):

| Field | Value |
|---|---|
| Architecture | YOLO11n, `DetectionModel`, head `Detect`. 2.59M parameters |
| Task | `detect`: boxes, not outlines |
| Classes | one, `{0: 'robot'}` |
| Saved by | Ultralytics 8.4.11 on 2026-02-05 |
| Trained from | `color_best2.pt`, with `data.yaml`, run `finetune_runs/collision_detector6` |
| Settings | 100 epochs, `imgsz=640`, batch 16, `lr0=0.001`, mosaic 0.5, degrees 10, default HSV jitter |
| Its own validation | precision 0.998, recall 1.0, mAP50 0.995, mAP50-95 0.982 |

The training call is reproducible from the checkpoint. The dataset behind
`data.yaml` is not available and will not be (user, 2026-10-04).

Out-of-the-box check on six lit frames (`conf=0.25`, MPS):

| Frame | `imgsz=640` | `imgsz=1440` |
|---|---|---|
| fight2 @150 s (one large lifter robot in view) | 0 | 0 |
| fight2 @250 s | 0 | 1 (0.81, at the frame edge) |
| fight3 @165 s (two robots) | 2 (0.96, 0.95) | 1 (0.69) |
| fight4 @150 s (two robots) | 2 (0.93, 0.78) | 2 (0.96, 0.94) |
| fight5 @140 s (two robots) | 2 (0.91, 0.86) | 2 (0.95, 0.88) |
| fight5 @195 s (one robot in view) | 1 (0.94) | 1 (0.86) |

- It already works on this camera's monochrome frames, so its training data
  very likely included frames from these clips or this camera. Which clips
  cannot be known.
- It misses the large lifter in fight2 at both sizes. Six frames are a
  spot-check, not a score, but the miss is on an unusual robot, which is where
  generated variety is meant to help.
- Native-size inference is not better than 640. Use 640.
- Its own validation score is at ceiling, so any gain has to be measured on
  frames it has not seen.

### Robot meshes
`augment/data/meshes/Meshy_AI_Nemesis_1_0918034213_texture_obj/` and
`.../Meshy_AI_Gigabyte_0918035233_texture_obj/` (git-ignored). Used elsewhere
for RGBTrack 6D pose testing.

| | Nemesis | Gigabyte |
|---|---|---|
| Format | OBJ + MTL + one 2048x2048 RGB texture | same |
| Size | 15,445 vertices, 30,739 triangles | 13,696 vertices, 30,216 triangles |
| Extent (x, y, z) | 0.339, 0.120, 0.354 | 0.448, 0.120, 0.448 |
| Looks like | black body, red and gold markings, four forks, red disc | grey chassis, black top plate, exposed drive parts |

- Generated by Meshy AI, so they are approximations, not CAD.
- y is up, the floor is y=0.
- **No real scale.** Both are exactly 0.12 units tall, which is a
  normalisation, not a measurement.
- A plain z-buffer render with the texture is enough to produce recognisable
  robots at the 100-250 px they occupy in a fight frame, with exact outlines.
  No renderer dependency is needed for that.

## Decisions
- Folder `augment/`, own manifest, own venv.
- **Two classes, `nemesis` and `non-nemesis`, YOLO boxes** (user, 2026-10-04,
  replacing the same day's "one class `robot`"). `finetune_color.pt` as
  provided is still one class; the two-class detector is the user's to train.
  The ids live in `scripts/classes.py` and are written into every label file.
- **Class order is `0 = nemesis`, `1 = non-nemesis`** (confirmed by the user,
  2026-10-04).
- **Cut-outs are coloured in their robot's real livery**, described to the
  model in words from the identity table. Every fifth sheet of a robot keeps
  an invented paint, to cover guests; only robots with five sheets or more
  get one, so in run1 that was 4 of 41 sheets and 6% of the robots. No wiki
  picture is downloaded or given to the model.
- **Every robot in these images is `non-nemesis`, and Nemesis is not in them.**
  They are the negative class's training images. An unlabelled Nemesis would
  teach the detector to ignore it, and one labelled from the approximate mesh
  would teach it the wrong robot. Gigabyte is another robot, so its mesh stays.
- **Real frames from the clips are `non-nemesis` too**, since Nemesis is in
  none of them: `extract.py` writes that class.
- **Robot scale is an approximate estimate** (user, 2026-10-04): mesh renders
  are sized to the footprint of real robots at the same floor position in the
  clips, with jitter.
- Robots come from two sources: real cut-outs from the clips, coloured by the
  model, and the Gigabyte mesh. The two robots in an image always come from
  different fights, so they are never the same robot twice.
- Keep the real camera view. The image model adds colour, new robots and
  combat effects; it does not invent whole scenes. Free text-to-image is not
  used: wrong viewpoint, no labels.
- The camera is fixed, so the empty arena is coloured once per plate and
  reused, not per image. Brightness stays from the real plate and only colour
  comes from the model, so geometry and texture stay real.
- Only the edited region is pasted back, through our own mask.
- Images are generated in colour, with a grey copy exported alongside. The
  detector is to run on a colour camera (see The target camera) in a similar
  overhead position; the only real test frames today are monochrome.
- Generation model: FLUX.2 klein 4B. Full precision through `diffusers` on
  CARC; 4 bits through mflux on this Mac, where the full-precision one does
  not fit in 16 GB. The planned bake-off against SDXL 1.0 inpainting is
  dropped: klein did both jobs in the Mac trial and its colour is calibrated
  afterwards.
- **The bulk set is generated on CARC, and the user commits, pushes and
  submits** (user, 2026-10-04), although its model work would fit in about
  2.5 hours on the Mac. `mds/carc.md` is the procedure.
- Bulk generation and detector training run on USC CARC. All bulk images come
  from one model, whatever the Mac trial uses.
- **The full run goes up without a second smoke test** (user, 2026-10-12).
  The job checks every plate itself and sets a failing one aside before
  composing (`augment_report.py gate`). Making the set on the Mac instead was
  offered and not chosen.
- Proof test: one whole fight held out, chosen before any training. Arms, three
  seeds each: detector as is; fine-tuned on real frames only; real plus plain
  composites with no generative step; real plus generated. GO if generated
  beats real-only by more than the seed-to-seed spread.
- Outlines on real frames come from SAM 2.1 through Ultralytics, prompted
  frame by frame from what changed against the plate (see Real material).
- **Real frames are labelled only when everything robot-sized in view is
  moving and cleanly outlined.** Anything else is flagged for review, never
  guessed. A missing box on a visible robot does more harm in training and
  scoring than a missing frame.

## Open decisions
- **Two identities are not settled**: fight2's small robot (Chopper or
  OverKill) and fight5's dark robot (Witch Doctor, or Malice). The user may
  know the match-ups.
- **No more meshes will be supplied** (user, 2026-10-04); Gigabyte's stays the
  only one. Still worth having: a test frame from the new camera, and its
  lens and mounting.
- **Which fight is held out.** The detector's training set is unavailable, so
  no fight is provably unseen. Proposed rule: hold out the fight the detector
  as provided does worst on. On the accepted frames that is fight2 (recall
  0.54), but fight2 has only 12 accepted frames, and fight1 has none. A
  held-out set worth scoring on needs fight2's flagged frames reviewed (596,
  with proposed boxes), or fight1's. Who reviews, and how many frames, is the
  user's call. The limit stays on record either way: the test cannot rule out
  that the detector has seen the held-out fight.
- **Cut-outs from fight1 and from flagged frames.** The 315 cut-outs come
  from accepted frames only, so fight1's three robots and every disabled or
  damaged robot are missing. A cut-out needs a clean outline, not a complete
  frame, so the library can be widened without review when composing starts.
- **Framing on the target camera.** The model is known; its lens, position and
  crop are not. No footage from it exists, so colour performance cannot be
  scored yet.

## Real material (step 4)
`scripts/extract.py`, per clip: lit range -> plate -> kept frames -> one SAM
outline per changed region -> checks -> labels, proposals and review sheets.

### How it works
- **Plate.** Per-pixel median of 120 frames from the first 40% of the lit
  segment. All five plates are clean empty arenas.
- **Kept frames.** Every 30th lit frame (about 3 per second), skipping any
  whose floor brightness is outside 0.7-1.4 of the plate's.
- **Changed pixels.** Brightness differs from the plate by more than 28 grey
  levels *and* the 21x21 neighbourhood no longer correlates with the plate
  (`TEXTURE_MAX = 0.6`).
- **Outlines.** Each changed region of 800 px or more gets one SAM 2.1 prompt:
  a padded box, a point deep inside the region, and two background points on
  the box's far corners. Only the largest connected part of the result is kept.
- **Two stages.** The outline pass is the slow one (SAM) and writes
  `outlines.json` and `masks/`. The review pass judges every frame from those
  in seconds, so a rule or the review lists can change without redoing SAM:
  `extract.py <clip> --review`.
- **Accepted** means exactly two outlines, both moving, both inside the arena,
  nothing else robot-sized in view, little change left unexplained, in a clip
  that has two robots, and not on the `REJECTED` list. Only accepted frames
  get a file in `labels/`.
- **Flagged** is everything else, with the reasons in `manifest.json`. Every
  frame also gets its boxes in `proposals/`, as a starting point for review.

### What was tried, and why it changed
1. **Plate difference on brightness alone, one box prompt per region.**
   Shadows and the start squares counted as change (the squares are lit
   differently before the match starts), and a robot waiting on its square got
   the square outlined instead. Fixed by the texture test and the point
   prompts.
2. **SAM 2 video tracking**, to carry each robot through contact. Dropped:
   - Cost grows with clip length here. A 90-frame trial ran at 1.6 s per frame
     on CPU and 2.1 s on MPS; full clips slowed to 7 s (fight1, MPS) and 22 s
     (fight2, CPU) per frame, and two hours did not finish one clip.
   - In fight2 the seeds landed on two saw slots that had changed state, and
     the tracker followed the slots.
   - In fight1 it lost one robot partway through and never recovered it.
   - Ultralytics 8.4.11 passes a mask prompt to the video model as `(1, H, W)`
     where the model reads `(1, 1, H, W)`. Worked around by subclassing
     `SAM2VideoPredictor.add_new_prompts`; the code is gone with the tracker.
   - The video predictor drops empty outlines from its result, so one outline
     back for two robots cannot be attributed.
   If contact frames are ever wanted in bulk, run tracking on a CARC GPU in
   short chunks, each seeded from an accepted frame.
3. **A static-clutter mask** (pixels changed in over 30% of a clip's frames),
   to ignore saw slots and settled debris. Dropped: in fight3 one robot is
   disabled for most of the fight, the mask swallowed it, and frames were
   accepted with that robot unlabelled.
4. **Plate over the whole lit segment.** The same disabled robot left a ghost
   in the median. Hence the first 40% only.
5. **Accepting one-robot frames.** On review many hid the other robot: small
   and still by the far wall, or tangled with the first inside one outline.
   Hence two outlines or nothing.
6. **fight2's lifter.** It is all arms and forks; a fork or its shadow became a
   region of its own and passed as the second robot. Hence the piece check
   (`FRAGMENT`) and the arena-edge check.

### Why a frame is flagged
| Reason | What it usually is |
|---|---|
| static object | A disabled robot, debris, or a hazard that changed state. These cannot be told apart automatically |
| only one robot found | The other is out of view, hidden, still, or inside the same outline |
| robots in contact | One outline where the previous frame had two close together |
| more than two robot-sized things | Debris or torn-off parts |
| a region got no clean outline | SAM's outline did not land on the changed region |
| unlabelled change | Over 35% of the changed pixels fall outside every outline |
| robot at the arena edge | The outline touches the arena mask's border, so the robot may continue past it |
| second outline may be a piece of the first | A tiny outline touching a much larger one |
| no robot outlined | Nothing robot-sized changed, or no outline passed |
| clip not trusted | The fight does not have exactly two robots (`UNTRUSTED`) |
| rejected on review | Passed every check, wrong when looked at (`REJECTED`) |

### Calibration
- Share of a correct outline on changed pixels, fight3, 60 outlines:
  0.58-0.98. The ones under 0.75 had dragged in floor or shadow. Hence
  `MIN_ON_CHANGE = 0.55` to reject and `CUTOUT_ON_CHANGE = 0.75` for cut-outs.
- The camera is not in the same place in every fight. Against fight4 the
  plates are shifted vertically by -12.6 px (fight1), -6.7 px (fight2), 0
  (fight3) and +4.8 px (fight5). A plate, a mask or a box from one fight does
  not transfer to another without registration.

### Results (2026-10-04)
| Clip | Lit seconds | Kept frames | Accepted | Flagged | Clean cut-outs | Outline pass |
|---|---|---|---|---|---|---|
| fight1 | 133 | 434 | 0 | 434 | 0 | 6.5 min |
| fight2 | 186 | 608 | 12 | 596 | 21 | 8.7 min |
| fight3 | 64 | 209 | 49 | 160 | 90 | 3.0 min |
| fight4 | 97 | 316 | 28 | 288 | 51 | 3.7 min |
| fight5 | 89 | 291 | 81 | 210 | 153 | 4.0 min |
| all | 569 | 1,858 | 170 | 1,688 | 315 | 26 min |

Flag reasons across all clips (a frame can carry several): static object 780,
only one robot found 674, more than two robot-sized things 668, a region got
no clean outline 630, unlabelled change 533, robot at the arena edge 520, clip
not trusted 434, robots in contact 310, no robot outlined 198, second outline
may be a piece of the first 7, rejected on review 2.

- **Only 9% of kept frames are labelled automatically.** They are the part of
  each fight where both robots are moving, apart, and inside the arena. Contact,
  a disabled robot, debris and edge frames are all in the flagged pile, with
  proposed boxes. A training or test set that covers real combat needs those
  reviewed by a person.
- **fight1 has three robots**: two large ones and a minibot. Seen at full
  resolution in frame 32000. The two-robot rule cannot vouch for any of its
  frames, so the clip is in `UNTRUSTED` and nothing from it is accepted.
- **Small boxes by the far wall are mostly real robots.** At sheet scale they
  looked like debris; cropped at full resolution, fight4's is a V-shaped robot
  of about 1,200 px and fight5's thin tall box is an upright robot. Judge
  small things on crops, not on sheets.
- **Review method.** Every accepted frame was compared with the detector's
  boxes. Where both labelled robots matched a detector box (IoU 0.5) and the
  detector added nothing, the frame was taken as right: two independent
  sources agree. The 63 frames where they disagreed were looked at, the
  doubtful ones at full resolution. Outcome: two frames rejected
  (`REJECTED`), fight1 dropped, the rest kept. This leans on the detector to
  pick what to look at, never to decide a label.

### The detector as provided, on the accepted frames
`imgsz=640`, `conf=0.25`, a robot counts as found at IoU 0.5.

| Clip | Labelled robots | Found | Missed | Extra boxes | Recall | Precision |
|---|---|---|---|---|---|---|
| fight2 | 24 | 13 | 11 | 8 | 0.54 | 0.62 |
| fight3 | 98 | 90 | 8 | 2 | 0.92 | 0.98 |
| fight4 | 56 | 50 | 6 | 0 | 0.89 | 1.00 |
| fight5 | 162 | 158 | 4 | 0 | 0.98 | 1.00 |
| all | 340 | 311 | 29 | 10 | 0.91 | 0.97 |

- Provisional: these are the easy frames, and few of them.
- fights 3, 4 and 5 are near ceiling, which fits the detector having trained
  on them. fight2 is where it fails: it breaks the lifter into pieces or
  misses it, and misses the small robot at a distance. In fight4 its misses
  are the V-shaped robot at the far wall.
- So the headroom is on unusual shapes and on small, distant robots.


## Mac trial (step 6, first part, 2026-10-04)
Twelve labelled colour images, start to finish on this machine:
`augment/out/trial-colour/` (`images/`, `images_grey/`, `labels/`,
`manifest.json`, `sheet.jpg`). `augment/out/trial-mono/` holds the same kind
of set on the monochrome plate. This is the second version; the first had
Nemesis in it, grey cut-outs and the model's raw colours.

### The model on this machine
- **Runtime.** FLUX.2 klein 4B at 4 bits through `mflux` 0.21.0
  (`RunPod/FLUX.2-klein-4B-mflux-4bit`, 4.6 GB), command
  `mflux-generate-flux2-edit`, which takes a picture to edit plus reference
  pictures. It lives in `augment/mlx/.venv` because mflux pins its own stack.
- **Bonsai Image 4B was ruled out before downloading it.** Its Mac runtime is
  a fork of mflux that exposes text-to-image only.
- **Cost.** About 2.5 to 3 minutes a run at 768 px, 7.8 minutes for two
  references at 1024x544. Every run peaks at 12.3 GB of this machine's 16.
  Fine for a few plates and sheets, not for a pass over every image.

### Colouring the plate (`colorize.py plate`)
- **Told only the arena's colours, the model invented.** Attempt 1 painted a
  large red field across bare floor and made both start squares blue.
- **Told what stays grey and which thing gets which colour, it behaved.**
  Attempt 2, from a ground-level still: grey floor, red left square, pale
  blue right square, yellow slot outlines, yellow and black bumpers. The
  centre logo stayed grey, which was wrong.
- **Given the overhead broadcast screenshot, it got the layout right.**
  Attempt 3 (seed 3, the current plate): both start squares and both B
  panels in the right colours, lettering white, floor charcoal. It also
  painted the far-wall platform's markings yellow, which the broadcast shows
  grey. 2.2 minutes.
- **Its colours are then calibrated to the broadcast.** The model's paint
  varies from run to run: one red came out at twice the measured strength,
  one blue came out cyan. Whatever falls in a paint's range of hues is turned
  to the measured hue and scaled to the measured strength. Result on the
  plate against the targets above: red square (+51, +40), left B (+53, +41),
  blue square (-7, -35), right B (-14, -31), floor (+2, -5).
- **Paint is strong colour that is also far from the floor's own colour**
  (since 2026-10-05). The floor's colour is the median over the arena,
  whatever cast the model gave it, and is moved to the measured floor colour.
  Each paint is measured on floor pixels only: the full model paints the
  bumpers a stronger yellow than the slot outlines.
- **Colour outside the arena is kept faint.** One run turned the whole left
  wall bright red.
- **Detail comes from the real plate.** The model's picture is a clean,
  simplified floor. The result keeps the plate's detail finer than 15 px and
  moves coarser brightness halfway to the model's. Keeping the plate's
  brightness outright made the red square pink: the camera is red-sensitive
  and sees both squares equally bright.
- **Still approximate.** The broadcast camera is not the fight camera: its
  white balance, exposure and compression are its own, and the lighting
  changes through a show. The plate matches one screenshot.

### Colouring the cut-outs (`colorize.py cutouts`)
- **Nine to a sheet.** Nine real cut-outs go on one grey 768x768 sheet and one
  model run colours all nine, each a different paint. 36 are done, nine from
  each of fights 2 to 5.
- **The model redraws the robots** as clean, detailed vehicles of roughly the
  same shape. Only its colour is taken; shape and detail stay the camera's.
- **Brightness moves halfway to the model's.** With the camera's brightness
  kept outright, a white robot painted red came out pink.
- **The model's own redrawn robots are not used.** They hold the overhead
  view and could become a source of new robot designs, cut from their grey
  background. Not tried.

### Composing (`compose.py`)
- **Two robots from different fights**, so never the same robot twice: two
  cut-outs, or the Gigabyte mesh and a cut-out (30% of images).
- **A cut-out is slid at most 220 px along its row and 90 px across rows**
  from where it was cut, and resized by the real robots' size at the new row.
  Cut-outs from other fights are brought to the plate's gain. The camera sits
  up to 17 px apart between fights; that is ignored here.
- **Gigabyte's mesh** is rendered into a bank of 48 sprites (16 headings at
  45, 55 and 65 degrees of elevation, chosen by floor row) by painter's
  algorithm. Lit by the mesh's own vertex normals and a texture averaged over
  each triangle; lit by face normals with one texel per triangle, the top
  plate showed every facet.
- **Layouts.** Half put the two robots within reach of each other. The nearer
  robot is drawn last; a robot less than 45% visible is redrawn elsewhere.
  Boxes cover the visible part.
- **Matched to the footage, measured on fight4.** A frame differs from the
  plate on untouched floor by 10.8 grey levels (std): the plate is a median
  and has no sensor noise. Renders are blurred by one pixel and shadows are
  added. The grey copy gets noise of the measured size. The colour image gets
  about half, with a little of it in colour: the colour camera is unknown.

### What the trial does not do yet
- Cut-out colours are the model's invention, not the robots' real liveries.
- No generated opponents and no sparks or smoke.
- Robots are placed on the arena mask shrunk by 70 px, which still includes
  the hazards by the far wall.
- One coloured plate (fight4) and 36 coloured cut-outs: little variety yet.
- Images are 1440x762, the old clips' frame, not the new camera's.

## Livery colouring and the bulk composer (2026-10-04)
Step 2 of the CARC plan, done and checked on the Mac. Output of the check:
`augment/out/mac-check/` (50 images, verdict GO), with the per-robot review
sheets at `report/liveries_1.jpg` and `liveries_2.jpg`. The earlier
`out/trial-colour/` and `out/trial-mono/` are the old layout and old paint.

### Naming the robots (`scripts/robots.py`)
- **A frame's two robots are always different robots**, so each frame can be
  read only two ways. Two groups are grown under that rule from six simple
  features of each outline (brightness, its spread, bright and dark share,
  height against width, size for its row). Clustering outlines one by one was
  not tried; a per-frame rule on one feature alone was too weak for fights 3
  and 4.
- **Each group is named by one feature**: the bigger is Mammoth, the taller the
  Slot Machine, the one with more bright pixels Tazbot, the brighter
  HyperShock.
- **Every labelled frame was then looked at with its two robots named.** Right
  in fights 2 and 5. Fight4: six frames the wrong way round (Whiplash from
  above does not look like Whiplash from the side) and one unnameable. Fight3:
  two frames of wreckage. These are in `SWAPPED` and `UNNAMED`.
- **311 clean, named cut-outs** from fights 2 to 5: HyperShock 77, Witch
  Doctor 76, Slot Machine 47, Nightmare 39, Whiplash 27, Tazbot 24, Chopper or
  OverKill 11, Mammoth 10.
- **Kraken: 14 cut-outs picked by eye** from fight1 (`HAND_PICKED`), out of 46
  outlines that were full-size, textured and free of flame. Most show it from
  behind near its start square. Fight1's frames stay unlabelled.
- **Fight4 has no unboxed minibot in its labelled frames.** Checked: the only
  moving change away from both boxes was shadow under Tazbot.

### Colouring in real liveries (`colorize.py cutouts`)
- One robot per sheet of nine; the prompt gives its livery in words. Every
  fifth sheet of a robot gets the model's own choice of paint instead.
- **All nine came out recognisably right** against the wiki pictures: Kraken
  green scales with a red name plate, Tazbot red with white spikes, Whiplash
  black with neon-yellow wedges and hubs, HyperShock yellow with pink and
  black spots, Witch Doctor purple top with lime sides and hubs, Slot Machine
  black cabinet with a lit panel on a pink base, Nightmare black with a silver
  disc, Chopper-or-OverKill bare silver on black tyres.
- **The model drifts from a livery unless told what not to do.** Told "black
  steel tubes", it painted Mammoth red. Told "matt black all over ... no red,
  orange or any other colored parts", black. Nightmare picked up a maroon tint
  on three of nine.
- About 3.2 minutes a sheet on the Mac. The whole set's model work is about 41
  sheets and 11 more plates: roughly 2.5 hours here, with the machine's memory
  nearly full throughout. So this set does not strictly need CARC; the later
  per-image effects pass does.

### The bulk composer (`compose.py`)
- **Draws on everything `colorize.py` has made**: every `plate_color_*.png` of
  fights 2 to 5 and every coloured cut-out. Two superseded plates left in
  `data/plates/fight4/` by the earlier trials were picked up this way; the
  report flagged both (paint overlap 0.56, colours off by up to 22) and they
  were deleted. Stale files in `data/plates/` or `data/sprites/cutouts/` end up
  in images, and `stage.sh` would copy them to CARC.
- **A robot is drawn first, then one of its cut-outs.** Drawn by cut-out,
  fight5's two robots would be in half the images.
- **The kind of image is fixed before placing.** A two-robot layout fails more
  often than a one-robot one, and redrawing the kind on each retry left only
  half the images with two robots.
- **Robots stand only near where a real robot stood** in that clip (within
  90 px), which keeps them off the far-wall hazards.
- **Sizes come from one fit over all clips.** One fight alone has too few
  robots (fight2: one huge, one small).
- 0.25 s an image. `--start` lets several processes fill one run.

### The report (`augment_report.py`)
Thresholds are fixed at its top. NO-GO on any label or file error or if a
folder does not load in Ultralytics; MARGINAL on any other failed check.

| Check | Mac check, 50 images |
|---|---|
| Label files, class ids, boxes inside the frame | no errors |
| Loads in Ultralytics (`check_det_dataset`) | colour and grey both |
| Mix of two-robot / one-robot / empty | 0.84 / 0.06 / 0.10, within chance of 0.85 / 0.10 / 0.05 at this size |
| Plate against the approved one and the broadcast colours | 1 plate, ok |
| Box width against real robots at the same row | 0.93 |
| Floor grid reached | 0.72 (not judged under 500 images) |
| Near-duplicates | 0 |
| Two-robot images with boxes touching | 0.55 |

The detector as provided, on these 50 images: finds 48 of 87 robots (0.55)
and adds 18 boxes. By robot: Witch Doctor 0.90, Tazbot 0.89, Kraken 0.83,
HyperShock 0.57, Slot Machine 0.57, Nightmare 0.50, Gigabyte 0.40,
Chopper-or-OverKill 0.20, Whiplash 0.14, Mammoth 0.00. Reported, not judged:
it says where the set differs most from what the detector knows, and Mammoth
is the robot it also missed in the real fight2 frames. It is not evidence that
training on these images helps.

### Not verified
- `stage.sh` and `fetch.sh` are not known to have talked to CARC: the user
  moved the inputs and results themselves. The upload's file list and rsync
  flags were rehearsed into a local folder.
- Whether training on the set improves detection on real fights: the proof
  test, still parked.

## CARC smoke run (2026-10-05)
`RUN=smoke,N=50,SHEETS=1,SEEDS=3`, job 12658940, one A100. Output:
`augment/out/smoke/` (56 images), log `logs/augment-12658940.out`.

**The job runs end to end on CARC.** The environment imports, the model loads
from the scratch cache with the hub offline, the CUDA branch of
`colorize.generate` works, eight compose processes fill one run, and the
report grades it. No label errors; both folders load in Ultralytics; mix
0.82 / 0.07 / 0.11; box width against real 0.89; no near-duplicates.

**Verdict MARGINAL, and the set is not usable: three of the four plates are
wrong, and 45 of the 56 images sit on them.** Only fight4's plate is right,
and it is the one made on the Mac and copied up.

| Plate | Made by | Paint overlap (bar 0.5) | Centre floor (a, b), target (+3, -5) |
|---|---|---|---|
| fight2 seed 3 | CARC, full model | 0.31 | (-8, -30) |
| fight3 seed 3 | CARC, full model | 0.32 | (-8, -30) |
| fight4 seed 3 | Mac, 4-bit build | 1.0 | (+2, -5) |
| fight5 seed 3 | CARC, full model | 0.32 | (-8, -30) |

Two separate faults, both on the CARC plates only:
- **Blue floor.** The lit part of the floor sits exactly on the blue paint's
  target colour, and the blue squares are pushed to a neon cyan. So
  `colorize.repaint` took the floor for blue paint: it calibrates on the median
  of everything bluish, and a large faintly blue floor outweighs two small
  squares. How blue the model's own floor was is not known: the raw
  `generated_3.png` files stayed on CARC.
- **Phantom robots on the start squares.** The broadcast screenshot has a
  robot waiting on each square, a blue one on the red square and a yellow and
  black one on the blue. The full model copied both across as coloured blobs.
  The real plates are empty there. Every image on those plates therefore holds
  two unlabelled robot-like objects. The report has no check for this; it
  failed these plates on the floor alone.
- The 4-bit build did neither on fight4 seed 3. One plate is no proof that it
  never would.

**Cut-out sheets from the full model are right on all nine robots**, and
sharper and more saturated than the Mac's. Drift: Nightmare's frame is maroon
on about five of nine, HyperShock has large red areas on two of nine, and the
Slot Machine got red wheel hubs.

The detector as provided finds 40 of 96 robots (0.42) and adds 16 boxes.
Reported, not judged.

### The fix (2026-10-05, on the Mac)
The three raw pictures were brought down
(`out/smoke/report/plates/<clip>_generated_3.png`). The model's own floor in
them is navy, (a, b) about (-1, -14), on every plate; the 4-bit build's was
(0, -2). The old correction then doubled it. Rebuilding the smoke run's plates
from these pictures with the old code reproduces the report's numbers exactly.

- **`colorize.repaint` tells paint from floor by distance from the floor's own
  colour**, and measures each paint on floor pixels only. A first version that
  used distance alone counted every neutral pixel (walls, the white lettering)
  as paint once the floor was navy; paint has to be strong as well.
- **`colorize.empty_reference` covers the two robots** in the screenshot with
  a colour gradient fitted to the square around each. A flat fill showed as a
  rectangle, and OpenCV's inpainting left blotches.
- **`augment_report.py` checks the start squares**: the share of each square's
  outline that is not its own colour (`SQUARE_GAPS = 0.09`). Empty squares
  measure 0.04 to 0.06; squares with a copied robot 0.13 to 0.19.
- **`generate.job` copies the model's raw plate pictures** to
  `report/plates/`, so a fetch brings them back.

The same four raw pictures through the new correction:

| Plate | Paint overlap | Floor (a, b) | Worst paint, off by | Start squares | Passes |
|---|---|---|---|---|---|
| fight2 seed 3, CARC | 0.68 | (+2, -6) | red 5.8 | red 0.17, blue 0.05 | No: robot on the red square |
| fight3 seed 3, CARC | 0.73 | (+2, -6) | yellow 6.3 | 0.13, 0.19 | No: a robot on each |
| fight4 seed 3, Mac | 1.0 | (+2, -6) | yellow 3.2 | 0.04, 0.06 | Yes |
| fight5 seed 3, CARC | 0.75 | (+2, -6) | yellow 6.3 | 0.13, 0.18 | No: a robot on each |

- The floor and every paint are now right on all four. fight4's approved plate
  moved by 0.7 grey levels on average and was looked at again.
- The three CARC plates still fail, now only on the new check: the robots are
  in the model's pictures and go only when CARC colours those plates again
  from the covered reference.
- **The covered reference has only been through the 4-bit build**: fight3
  seed 3 on the Mac, squares empty, plate passes (overlap 0.85, squares 0.05
  and 0.06). That shows the covered picture still works as a reference. It
  did not show what the full model does with it; the full run did (see Full
  run): no copied robots on any of its twelve plates.

### One visit instead of two (2026-10-12)
A second smoke test would only have looked at the plates. The plate checks
now run inside the job, so the full run is submitted directly.

- **The gate.** `augment_report.py gate` runs `check_plate` on every
  `plate_color_*.png` and renames a failing one
  `rejected_plate_color_<seed>.png`. `compose.py` picks plates up by the
  pattern `plate_color_*.png`, so it never sees one. `generate.job` calls the
  gate between colouring and composing.
- **Tested on the Mac** on plates rebuilt from the smoke run's three raw
  pictures plus the approved plate: the three are set aside (red-square gaps
  0.17, 0.13, 0.13), the approved one stays, a second pass changes nothing,
  and `compose.stage` then offers only fight4's plate.
- **The report lists what was set aside** (`plates_set_aside`, with each
  plate's numbers) and warns when a run of 500 images or more is composed on
  fewer than six plates (`PLATES_MIN`). A full run colours twelve.
- **The invented-paint branch has now run once**, on the Mac's 4-bit build:
  one sheet of nine Kraken cut-outs with `INVENTED_EVERY` set to 1, written
  to a scratch folder. Nine sprites and an index, every entry
  `"paint": "invented"`. By the numbers the nine got different paint (red,
  orange, yellow-green, green, purple and four left unpainted) and the
  background stayed grey. Looked at by numbers only; the sheet is
  `out/invented-test/sheet.jpg` (grey input, painted sheet, sprites).
- **A full run colours 41 cut-out sheets**, four of them in invented paint
  (the fifth sheet of HyperShock, Witch Doctor, Slot Machine and Nightmare).
- **The smoke run's nine sheets must not be reused, and no longer are**
  (found 2026-10-13, before the full run). `--sheets 1` spreads its nine over
  all of a robot's cut-outs, so its sheet 0 holds other cut-outs than a full
  run's sheet 0 under the same file name. `generate` skipped the existing
  coloured sheet, which would have put one cut-out's colours on another for
  81 of the 325 cut-outs. `colorize.cutouts` now deletes a coloured sheet
  whose grey sheet has changed. Tested with a stand-in for the model: a
  smoke-style run then a full run repaints sheet 0, every coloured sheet
  matches its grey sheet, and a second full run paints nothing.

Before the full run, the three bad raw pictures have to be deleted on CARC,
or they are reused and set aside again (`carc.md`, step 5). No plate from the
full model has passed yet; the gate is what stands between that and the set.

## Full run, `run1` (2026-10-15)
`RUN=run1`, defaults (3,000 images, all sheets, seeds 1 2 3). Output:
`augment/out/run1/` (2.6 GB). **Verdict GO**, no warnings. The job log was not
copied back, so its run time is not known.

| Check | Result | Bar |
|---|---|---|
| Images | 3,000 colour, 3,000 grey | |
| Label files, class ids, boxes inside the frame | no errors | none |
| Loads in Ultralytics | colour and grey | both |
| Mix two / one / empty | 0.845 / 0.110 / 0.046 | 0.85 / 0.10 / 0.05 +- 0.04 |
| Plates passing | 7 of 12 | 6 |
| Box width against real robots at the same row | 0.88 | 0.6 to 1.6 |
| Floor grid reached | 0.94 | 0.6 |
| Near-duplicates | 0 | 1% |
| Two-robot images with boxes touching | 0.50 | reported |

**Plates.** The covered reference worked: none of the twelve pictures the
full model painted has a robot on a start square.
- Kept: fight2 seed 1; fight3 seeds 1, 3; fight4 seeds 1, 3 (3 is the Mac's);
  fight5 seeds 1, 3. All seven look right by eye: charcoal floor, red square
  left, blue right, both empty, red and blue B, yellow outlines. fight2's
  images all sit on its one plate.
- Set aside: seed 2 in every fight, and fight2 seed 3. **Seed 2 painted the
  blue square in two tones**: its right-hand part a deeper blue whose hue
  falls outside the blue range, so after correction it stays violet (blue
  square gaps 0.14 to 0.29). fight2 seed 3 missed on the red square by a hair
  (0.094 against 0.09); by eye it looks fine.
- So the square check also catches uneven paint, not only copied robots.

**Cut-out sheets, all 41 looked at.** Liveries right on all nine robots.
- **Nightmare drifts**: 7 of its 36 livery cut-outs came out maroon, red,
  teal or brown instead of black, measured on the cells that hold a real
  cut-out. Mammoth (0 of 10) and Chopper-or-OverKill (0 of 11) held.
- **Mammoth shows brown patches in composites**: its outline includes the
  floor seen through its open frame, and the model paints that floor too.
- The four invented-paint sheets give each robot a different colour (red,
  orange, yellow, green, blue, white, black), as intended.
- **On a robot's last, part-filled sheet the model draws extra robots of its
  own** in the empty cells (the prompt says nine). Only cells holding a real
  cut-out are used, so they never reach an image.

**Robots in the images**: Tazbot 624, Slot Machine 654 (133 invented),
HyperShock 608 (90), Witch Doctor 602 (99), Kraken 594, Nightmare 521 (38),
Whiplash 514, Chopper-or-OverKill 433, Gigabyte 429, Mammoth 418. Fight2's
two robots are fewest because an image's two robots come from different
fights.

**The detector as provided**, on 300 of the images: finds 360 of 540 robots
(0.67) and adds 74 boxes; 0.42 on the smoke run, 0.91 on the real labelled
frames. Lowest on Mammoth 0.15 and Chopper-or-OverKill 0.37, the two it also
missed in real fight2 frames; highest on Kraken 0.88 and Witch Doctor 0.85.
Reported, not judged: it is not evidence that training on the set helps.

## Combat trial (2026-10-15, on the Mac)
Plan approved 2026-10-15: robots in real combat and robots with no footage,
1,000 images made and judged on this Mac before anything goes to CARC. The
user chose real combat poses and new robots, made from the footage with no
image-model pass per image. Sparks, flames, smoke and debris were not chosen.
Output: `augment/out/combat-trial/`, laid out as run1. **Verdict GO**, no
warnings.

### Run1's colours, rebuilt on the Mac
Run1's 41 coloured cut-out sheets and 12 raw plate pictures (from
`out/run1/report/`) were put back in `data/`; the plates and sprites were
rebuilt from them with no model call. The gate kept the same 7 plates. The
Mac trial's older sheets are in `sprites/cutouts_mac_2026-10-04/`.

### Real combat poses (`scripts/combat.py`)
- **Two robots in contact are one patch of change**, so `extract.py` outlined
  them as one robot or only one of them. `combat.py track` follows each pair
  into contact through the video, every 6th frame (kept frames are 30
  apart), prompting SAM 2.1 with each robot's previous box, a point on it and
  a background point on the other. It starts from the last kept frame where
  the two were apart and also runs backwards from the first one after.
- **What made it work.** A first pass found 23 touching frames in 46 contact
  stretches. SAM's grown boxes took in floor lettering and slots, which the
  "share on changed pixels" test then rejected; outlines are now trimmed to
  changed pixels and their surroundings (`tidy`), and a robot trimmed by more
  than a quarter is not used. With backward tracking: 134 candidate frames.
- **Every candidate was looked at**, with both outlines and names drawn.
  108 pairs kept (`PAIRS`): fight2 1, fight3 23, fight4 39, fight5 45.
  Dropped: an outline that took in the far-wall platform, a floor patch, a
  spark cloud, or most of the other robot. One fight3 run was named the wrong
  way round (followed back from a frame named by features alone) and is
  corrected in the list. fight5 includes Witch Doctor thrown onto its side.
- **Single robots** from set-aside frames outside contact: still (stopped,
  flipped, wrecked) or much wider than a floor robot at that row (lifted).
  30 kept of 128 candidates (`SINGLES`): Mammoth with its arms up, the
  smashed Slot Machine, stopped and flipped robots. Most candidates were
  floor slots or lettering, or debris.
- **Few fight2 pairs.** Mammoth's open frame and fight2's merged outlines
  lose the track; the one fight2 pair is Mammoth over Chopper-or-OverKill.
- Naming uses run1's groups (`robots.groups`, `robots.name`); `identify` still
  gives run1's names exactly.

### Colouring the poses
`colorize.py poses`: 31 sheets on the Mac, 246 sprites, every fifth sheet of
a robot in invented paint. All looked at; liveries right. Nightmare's
livery now says "no red, maroon, brown, teal": it stayed black on all three
of its sheets (it drifted on 7 of 36 in run1). Mammoth still picks up brown
where the floor shows through its frame.

### New robots (`colorize.py newbots`)
- Diablo, Ginsu, Malice, Chopper and OverKill, the roster robots with no
  footage, described in words from the public wiki's ShowBot pages
  (`robots.NEW_ROBOTS`); no picture was downloaded or given to the model.
  Chopper and OverKill are both drawn because which one fight2 holds is not
  known.
- Drawn by the model from text, nine views to a sheet on grey floor, two
  sheets each, then found on the sheet and cut out with SAM 2.1: 63 views,
  all looked at, all clean.
- **The prompt took two corrections.** Told "saw blades used as wheels", it
  drew Ginsu on tyres; told "nine on a grid", it drew each on a grey mat;
  then it drew Ginsu with legs. It now asks for "a heavy BattleBots fighting
  robot ... a low, heavy vehicle ... no legs, not a humanoid and not a toy".
- Checked beside renders of the Nemesis mesh: none looks like it.
- In images they are sized like real robots (1.01 of a real robot's width
  at the same row) and kept whole: two were first placed with their top just
  off the frame, which the report caught as a label error.

### The trial set
| Check | Result | Bar |
|---|---|---|
| Images | 1,000 colour, 1,000 grey | |
| Real pair / combat pose + robot / new robot + robot | 39.3% / 29.0% / 31.7% | 40 / 30 / 30 |
| Label errors; loads in Ultralytics | none; both | none; both |
| Plates | the 7 that passed for run1 | |
| Box width against real robots | 0.88 | 0.6 to 1.6 |
| Floor grid reached | 1.0 | 0.6 |
| Near-duplicates | 0 | 1% |
| Two-robot images with boxes touching | 0.68 (run1 0.50) | reported |

Each of the 108 real pairs appears in about 3.6 images.

The detector as provided, on 300 of the images: finds 0.69 of the robots
(run1 0.67) and adds 98 boxes. By source: real pairs 0.74, stopped robots
0.70, lifted robots 0.48, new robots 0.77, run1's calm cut-outs 0.61. By
robot: Mammoth 0.20 again the lowest. Lifted robots, which the detector
finds least, are mostly Mammoth. Reported, not judged.

### Not verified
- Whether training on these images helps: the proof test, still parked.
- The new robots are the model's reading of words, not the robots
  themselves.
- Pose colours come from the Mac's 4-bit build; a CARC run would repaint
  them with the full model.

## Steps
Done so far: vault documentation, inputs checked, scaffold, real material
from the clips (five plates, 170 labelled frames, 315 cut-outs), and three
Mac trials of 12 composed images. The proof test and its baseline are parked
at the user's choice (generation first), along with the review of flagged
real frames that only the proof test needs.

Current plan (approved 2026-10-04): a bulk `non-nemesis` set on CARC.

| # | Step | Where | Status |
|---|---|---|---|
| 1 | Documentation: class order, robot identities, this plan | Mac | Done 2026-10-04 |
| 2 | Record robot identities as data; make the code CARC-ready | Mac | Done 2026-10-04: see Livery colouring and the bulk composer. Not run on CARC |
| 3 | Commit and push `augment/`; copy inputs to CARC | Mac, the user | Handed over 2026-10-04: exact commands in `carc.md`. The upload was rehearsed locally (392 files, 80 MB) |
| 4 | CARC setup and smoke run: every plate, one cut-out sheet per robot, 50 images | CARC, the user | Run 2026-10-05, verdict MARGINAL: the job works, the cut-outs are right, three of four plates were wrong. Fixed on the Mac the same day. No second smoke run: the job gates plates itself (2026-10-12). See CARC smoke run |
| 5 | Full run, about 3,000 images, graded by `augment_report.py` | CARC, the user | Done 2026-10-15: `out/run1`, verdict GO. See Full run |
| 6 | Combat trial: real combat poses and new robots, 1,000 images | Mac | Done 2026-10-15: `out/combat-trial`, verdict GO. Scaling up waits on the user's look. See Combat trial |

What the full run is to produce:
- About 3,000 images at 1440x762, each with a grey copy.
- Mix: 85% two robots (half in contact), 10% one robot, 5% empty arena.
- Plates for fights 2 to 5, three colour seeds each. All 315 cut-outs plus
  Kraken's from fight1, in real livery with every fifth sheet in invented
  paint. Gigabyte from its mesh.
- Two ready-to-train folders, `colour/` and `grey/`, each with `images/`,
  `labels/` and a `data.yaml` naming both classes.
- A report with a GO / MARGINAL / NO-GO verdict and contact sheets.

Not in this run: sparks, smoke and model-generated new robots. The new robots
and real combat poses are in the combat trial (step 6); sparks and smoke are
not planned.

## Environment (measured 2026-10-04)
- Apple M4, 16 GB unified memory, macOS 27.2. PyTorch 2.14.1 sees MPS.
- `augment/.venv` is 1.4 GB with the full manifest installed and no model
  weights. `Flux2KleinPipeline` imports from diffusers 0.40.0.
- 21 GB free disk after both venvs, the extracted frames and the 4-bit klein
  weights. klein at full precision is about a 16 GB download in the diffusers
  layout (skip the duplicate single-file checkpoint at the repo root,
  another 7.8 GB).
- `augment/mlx/.venv` (1.3 GB) holds mflux 0.21.0 and MLX 0.32.3.
- SAM 2.1 runs through Ultralytics from the checkpoint already in the Hugging
  Face cache (`sam2.1_hiera_small.pt`, 184 MB), symlinked as
  `augment/data/weights/sam2.1_s.pt`. No extra package and no download.
- CARC differs from the Mac's pins on purpose (`scripts/carc/setup_env.sh`):
  `torch` 2.6.0 from the `cu124` index, because diffusers 0.40.0 needs 2.6 or
  newer; and `numpy` chosen by pip, because the pinned 2.5.3 needs Python 3.12
  and the CARC module is 3.11.9. All six scripts parse under 3.11.
- **CARC login nodes allow each user 64 processes (threads count), 4 cores and
  32 GB** (CARC notice, 2022-10-07). numpy's OpenBLAS starts one thread per
  core it sees, 32 there, is refused part-way and raises an interrupt in its
  own process, so `import cv2` or `import numpy` dies with
  `KeyboardInterrupt` although nobody pressed anything. Seen on the first
  `setup_env.sh` run. Any Python on a login node needs
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The limit is on login nodes, so
  `generate.job` is not under it.
- Ultralytics pulls in `opencv-python`, not the headless build. On CARC that
  build can fail to import for want of `libGL`; if it does, swap it for
  `opencv-python-headless` there, never both.
