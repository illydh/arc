# Splatting task context

Gaussian-splat reconstruction of the BattleBots arena interior from handheld
phone footage. Separate concern from `vision/` -- see below.

## Goal
Near-term (user's explicit scope): produce a **raw 3D Gaussian splat, "just to
have."** No metric accuracy, no mesh, no simulation integration required yet.

Stated longer-term intent: an accurate reconstruction of the 48ft x 48ft arena
for simulation. The downstream consumer (physics engine, renderer, robotics
sim) is **not chosen**, which is why the near-term deliverable was
deliberately scoped down.

## Relationship to `vision/`
Deliberately decoupled. `vision/` is a minimal OpenCV/NumPy tracking pipeline
whose conventions (see `vision/mds/AGENTS.md`) scope to that directory only.
Splatting needs PyTorch plus a CUDA-compiled rasterizer -- an unrelated, heavy
toolchain that should not leak into `vision/`'s dependency manifest.

## Source footage
`splatting/data/video_inside_battlebox.mp4`
- 2560x1440, 30 fps, HEVC, 423.38 s, 12,702 frames, ~953 MB.
- Handheld phone walkthrough of the arena interior.
- Metadata (ffprobe, 2026-10-04): Samsung phone, Android 16; HEVC Main 10,
  HLG transfer (`arib-std-b67`), BT.2020 primaries, ~18 Mbit/s. So 2560x1440
  is the recorded size, and `images/` holds the full definition.
- `process_video.py` extracts 8-bit JPEGs with no tone-mapping filter, so
  the frames carry HLG/BT.2020 values read as SDR. This is consistent
  across frames and does not affect geometry; its effect on the splat's
  colour has not been checked.
- Unrelated to `vision/`'s fight clips (1440x762, ~98 fps, dark/noisy). Do not
  confuse the two datasets.

### Capture structure (measured, 2026-09-17)
The capture is **not** a uniform interior walkthrough. Sampling stills across
the full duration shows three distinct phases:
- **~0-110 s: floor close-ups.** Camera pointed nearly straight down, walking
  the arena floor -- panel seams, hazard slots, paint. Close subject, low
  texture, large image-space motion per second.
- **~110-310 s: the arena interior proper.** Walls, yellow bumper rail,
  polycarbonate barriers, BATTLEBOTS lettering, the lit pit beyond. This is
  the only segment that depicts what "a splat of the arena interior" means.
- **~310-423 s: ceiling.** Roof trusses and lighting rig, camera pointed up.

Consequence: a single run over the whole video mixes three sub-scenes captured
sequentially, connected only by the pans between them. Segment selection
(`--start`/`--end`) matters more than total frame count. Measured mean
Laplacian variance confirms the split -- ~355 on the floor segment vs ~830 on
the interior segment.

## Toolchain incompatibilities (measured, 2026-09-17)
nerfstudio 1.1.5 predates the installed ffmpeg and COLMAP and is broken
against both. This is the single most expensive thing in this document to
re-derive; each was found by running into it.

1. **ffmpeg 9 removed `-vsync`**, which `ns-process-data video` hardcodes.
   Extraction aborts immediately. The replacement option is `-fps_mode`.
2. **COLMAP 4.2 renamed `SiftExtraction`/`SiftMatching`** to
   `FeatureExtraction`/`FeatureMatching`. nerfstudio still emits the 3.x
   names, so `colmap feature_extractor` rejects its arguments.
3. **COLMAP 4.2 reads only faiss vocab trees** (it dropped flann in May 2025).
   The tree `nerfstudio.process_data.colmap_utils.get_vocab_tree()` downloads
   is the legacy flann one and aborts the matcher with a `Check failed:
   file_version == 1 || file_version == 2`. Vocab-tree matching and sequential
   **loop detection are therefore unavailable** without supplying a
   faiss-format tree explicitly. (The useless ~150 MB download is cached at
   `~/Library/Application Support/nerfstudio/vocab_tree.fbow`.)

Also relevant: nerfstudio's `run_command` crashes with
`AttributeError: 'NoneType' object has no attribute 'decode'` when a COLMAP or
ffmpeg call fails under `--verbose`, masking the real error behind a traceback.

**Consequence:** `ns-process-data` cannot run on this machine at all.
`scripts/process_video.py` owns frame extraction and COLMAP directly, and
calls nerfstudio's `colmap_to_json` for the `transforms.json` + `sparse_pc.ply`
that `ns-train splatfacto` consumes. Nerfstudio remains the **training** stack;
only its data-processing entrypoint is unusable. Do not "fix" this by patching
`site-packages` or downgrading the system toolchain.

## Decisions
- **Raw splat over textured mesh, for now.** A mesh (photogrammetry:
  SfM + MVS + meshing) was recommended instead, because a mesh serves any
  downstream consumer -- physics sim, game engine, renderer -- whereas a splat
  is render-only, and splat -> collision geometry is a lossy open research
  problem (SuGaR and similar), not a solved conversion. User scoped to a raw
  splat anyway since no simulator target is fixed yet. **Revisit this the
  moment a simulation consumer needing collision geometry is chosen** -- the
  splat will not serve it.
- **Nerfstudio for training, local scripts for data processing.** The original
  rationale for Nerfstudio was that `ns-process-data video` folds extraction,
  frame selection and COLMAP into one command. That rationale is dead on this
  machine (see Toolchain incompatibilities). What survives is the reason that
  actually mattered: `splatfacto` trains via gsplat, which is pip-installable
  with no hand-built CUDA submodules, unlike graphdeco-inria 3DGS which needs
  `diff-gaussian-rasterization` and `simple-knn` compiled against a matching
  CUDA toolkit.
- **Frame selection is automatic**, and is now genuinely blur-aware.
  `ns-process-data` never did blur rejection -- its `min_blur_score` is
  Polycam-only, and video mode just uses ffmpeg's `thumbnail` filter, which
  picks a *representative* frame per batch, not a sharp one. The earlier note
  in this document claiming Laplacian-variance blur rejection was wrong.
  `scripts/process_video.py` scores Laplacian variance on a cheap 320x180
  decode pass and keeps the sharpest frame of each window.
- **JPEG (`-q:v 2`), not PNG, for extracted frames.** PNG at 2560x1440 runs
  ~6 MB/frame, and the frame counts this footage needs make that tens of GB.

## Open decisions
- **Metric scale.** Deferred, not solved. Monocular SfM recovers geometry only
  up to an unknown global scale. When scale is eventually needed, the arena
  floor is a 12x12 grid of 4ft tiles totalling 48ft x 48ft (confirmed in
  `vision/mds/context.md`); those grid intersections are the natural scale
  reference, and are the same references `vision/calib` already uses for its
  homography. Not required for a raw "just to have" splat; required before any
  simulation use.
- **Static-scene assumption.** Vanilla 3DGS assumes nothing moves between
  frames. The contact sheet shows an empty arena but a lit, active pit area
  visible *through* the polycarbonate walls, with people and equipment beyond.
  Those will produce floating "ghost" artifacts. This is a *silent* failure --
  it degrades the result rather than raising an error. Masking the
  through-wall background is an option if the splat looks bad, not before.
- **Faiss vocab tree for loop closure.** Would enable `--vocab-tree`, loop
  detection, and drift correction on a walk that revisits corners. Not sourced
  yet; `process_video.py --vocab-tree PATH` is wired and waiting for one.
- ~~GPU target for training~~ **resolved: USC CARC (SLURM).** Used for run 1
  (training job 12394520, export job 12399735). Cloud rental (RunPod/Lambda)
  remains the fallback if queue latency or environment setup proves painful.
- ~~Python environment location~~ **resolved: repo-root `.venv`.** The
  question was moot: nerfstudio 1.1.5 and its PyTorch tree are already
  installed there (the venv is 2.0 GB, not the 168 MB previously recorded).
  The isolation concern is real but unrealised -- `vision/` is not installed
  into it. **Hazard:** the root venv has both `opencv-python` 5.0.0.93 and
  `opencv-python-headless` 4.10.0.84, which unpack into the same `cv2/`
  directory and have clobbered each other (the importable `cv2` reports
  4.10.0). Splatting only needs headless. Resolve before running `vision/`
  from this venv, since it needs `cv2.imshow`.

## Plan
- **Phase 0 -- environment. DONE.** COLMAP 4.2.0 (Homebrew, no GPU), ffmpeg
  9.0.1, nerfstudio 1.1.5 + torch 2.14.0 + gsplat 1.4.0 in the root `.venv`.
- **Phase 1 -- data processing, local, CPU.** `scripts/process_video.py` ->
  `scripts/sfm_report.py`. Also the go/no-go gate: if SfM cannot register the
  arena, it fails here, before any GPU time is bought or queued. **Gate
  passed as MARGINAL** on the interior segment -- a trainable dataset exists
  at `data/processed/interior400`. See Phase 1 findings.
- **Phase 2 -- training, GPU only. DONE for run 1.** `ns-train splatfacto` on
  USC CARC via SLURM. Requires a CUDA host; cannot run locally (verified:
  `gsplat: No CUDA toolkit found. gsplat will be disabled.`). Scripts in
  `scripts/carc/`, runbook in `carc.md`. Run 1: job 12394520, 30,000
  iterations, 656 s of training, 1.03M Gaussians; exported by job 12399735
  after the `LD_LIBRARY_PATH` fix.
- **Phase 3 -- viewing.** Corrected: this is **not** local the way it was
  originally planned. `ns-viewer` renders through gsplat and `ns-export` loads
  the model onto a CUDA device, so **neither runs on the Apple Silicon box**.
  The export therefore happens on CARC at the end of the training job, and
  what comes back is `export/splat.ply`, which opens in a browser WebGL splat
  viewer with no CUDA and no install. `pymeshlab` is consequently needed in
  the **CARC** environment (`ns-export` imports it at module load), not
  locally -- it is in `scripts/carc/setup_env.sh`.
- **Phase 3 -- evaluation.** `scripts/splat_report.py` grades a `splat.ply`
  locally (CPU) and writes `report.json` plus pictures to `report/` beside the
  .ply. Exact held-out scores need the model rendered, which is CUDA-only, so
  `scripts/carc/eval.job` runs `ns-eval` on CARC and the report reads its
  output as an optional fourth argument. Without it, held-out scores come from
  the training log's last full eval (step 29,000 for run 1). See Phase 3
  findings.

## Phase 1 findings
Measured, not predicted. All runs CPU-only, `sequential` matcher, no loop
closure.

| Segment | Frames | Spacing | Result |
|---|---|---|---|
| Floor, 0-40 s | 12 | ~1.7 s | **0 registered** -- "No images with matches", no model at all |
| Floor, 0-40 s | 61 | ~0.66 s | **17/61** across 2 fragments (5 + 12). Reproj 0.59 px |
| **Interior, 110-310 s** | **428** | **~0.47 s** | **280/428 (65.4%)**, 4 sub-models but 83.8% dominance. 30,937 points, reproj **0.935 px**, track length 4.93. **17.2 min** wall clock. Verdict **MARGINAL** |

The interior run (`data/processed/interior400`) is a complete, trainable
nerfstudio dataset: `transforms.json` with 280 posed frames, OPENCV intrinsics
(fl 3072 px), and `sparse_pc.ply` for splatfacto seeding. Trajectory bounding
box is 8.29 x 2.94 x 8.33 in arbitrary units -- a roughly square footprint with
low height, which is the right shape for a square arena and a useful sanity
check that the geometry is not nonsense.

**The 103-frame unregistered run (frames 253-355) is content, not density.**
Those frames were shot from outside the box looking *through* the
polycarbonate walls: reflections, specular highlights, transparency, and a
bright pit/seating area beyond dominating the frame. This is exactly the
failure mode predicted for specular/transparent surfaces. Adding frames there
will not help; masking or excluding that stretch might.

Conclusions so far:

- **The original frame budget was far too small.** The prior note here said
  "do not feed all 12,702 frames, target a few hundred." A few hundred frames
  spread over 423 s is ~1.4 s apart -- *sparser* than a spacing that already
  failed. For handheld walking footage the binding constraint is inter-frame
  baseline, not total image count. Density has to be chosen per segment.
- **Low reprojection error on a tiny fragment is not success.** The 61-frame
  run produced a clean 0.59 px model of 5 images. Registration *ratio* and
  sub-model count are the signals that matter, which is why
  `sfm_report.py` leads with them.
- **The floor segment is the hardest possible content** and should not be
  used to judge the footage: close range, repetitive concrete, low parallax
  relative to subject distance.
- **The gate's fragmentation rule was miscalibrated and has been changed.**
  It originally failed a run on raw sub-model count (>=3 -> NO-GO), which
  rejected the interior run despite one coherent 280-image model plus three
  12-22 image orphans. `sfm_report.py` now judges *dominance* -- the share of
  registered frames sitting in `sparse/0`, the model that
  `colmap_to_json` actually consumes -- with NO-GO below 50%. Raw sub-model
  count is still reported.

## Phase 3 findings (run 1)
Measured by `scripts/splat_report.py` on `out/splat.ply` (export of run
`2026-09-27_171857`). Held-out scores are final as of 2026-10-04: they come
from `eval.job` (`ns-eval` on checkpoint `step-000029999`, outputs in
`out/eval/`). The provisional scores from the training log at step 29,000
were 21.96 dB / 0.799 / 0.350, so the re-score changed nothing material.
Lengths are model units: nerfstudio scales the scene so the largest camera
coordinate is 1.0.

| Check | Result |
|---|---|
| Integrity | 1,006,172 Gaussians, file size matches the header, no NaN/Inf |
| Held-out (28 views) | PSNR 22.00 dB (spread 4.62), SSIM 0.808 (0.125), LPIPS 0.348 (0.151); within 0.25 dB of its best from step 11,000 |
| Training views | PSNR 29.0 dB averaged over the last 1,000 steps (gap 7.0 dB) |
| Per view | 13.2 to 29.8 dB. 9 views at 25 dB or above, 10 at 20-25, 9 below 20. Worst: `frame_00373` 13.2, `00383` 14.9, `00363` 15.0, `00140` 16.5, `00427` 16.6 |
| By position in the walk | Frames before the unregistered stretch (< 253): 22 views, mean 23.4 dB. Frames after it (> 355): 6 views, mean 16.8 dB |
| Floaters | 17.9% by count (15.1% outside the mapped region, 13.0% isolated, 0.5% in walked air); 20.2% by visual weight |
| Extent | Full box 37.8 x 39.1 x 26.3; 1-99% box 9.2 x 10.8 x 4.8; camera path box 1.75 x 1.83 x 0.40 |
| Floor | Present: covers 90.1% of the camera-path footprint, 4.2 degrees off level |
| Mix | 51.6% solid, 12.5% faint, 2.4% oversized, 5.4% needle-shaped |
| **Verdict** | **MARGINAL** (final). Misses GO on PSNR (22.0 < 25), SSIM (0.808 < 0.85) and floater weight (20.2% > 5%). Clear of NO-GO on all three counts |

Observations:

- **Two separate failure modes, seen in the 28 `ns-eval` pairs.**
  (1) Views after the unregistered stretch (`00363`-`00427`) look up and out
  through the polycarbonate; five of the six are smeared by large dark blobs.
  (2) About seven otherwise good interior views (`00060`, `00110`, `00120`,
  `00140`, `00150`, `00180`, `00220`) have a blurred blob in the foreground
  while the scene behind it is right. That is the signature of the splats
  sitting in the walked air, which are 0.5% of the count but 15.9% of the
  visual weight. Interior views without such a blob score 25-30 dB.
- **Dropping the through-the-glass views alone would not reach GO.** The 22
  earlier views average 23.4 dB. The foreground floaters have to go too.
- **`splat_clean.ply` removes the walked-air splats but is unscored.** Whether
  the cleanup raises held-out scores is untested: nothing here renders a
  .ply, and `ns-eval` scores the checkpoint, not the file.
- **The arena itself reconstructs.** The top-down render shows the square
  wall outline with the camera path inside it, and views across the floor
  score 28-30 dB.
- **The weak views are the predicted ones.** `00363`, `00373`, `00383` and
  `00427` look out through the polycarbonate at the pit, seating and roof.
  The renders are smeared with large dark blobs. This is the
  specular/transparent failure mode that also stopped frames 253-355 from
  registering in Phase 1. `00140` is a blurred floor shot with the
  operator's shadow in frame, which breaks the static-scene assumption.
- **The model is not level.** The fitted floor is 4.2 degrees off the model's
  z axis, so `orientation_method: up` (the average camera up vector) is off
  by that much. This matters for simulation, not for viewing.
- **There is a hole in the floor near the arena centre** (about 0.3 x 0.4
  units with no opaque splats at any height). Cause not yet checked.
- **Haze extends to about 19 units** around a scene whose camera path spans
  about 1.8. Most of it sits outside the mapped region and is cheap to crop
  in a viewer. The splats in the walked air are few (0.5% of the count) but
  carry most of the floater visual weight, because they sit right in front of
  the lens.
- **The training-view PSNR of 34.7 dB recorded earlier was one image** at
  step 29,990. Averaged over the last 1,000 steps it is 29.0 dB.
- **Visual weight definition.** The plan defined floater weight as opacity x
  world-space area. Before the first run this was changed to opacity x solid
  angle from the nearest training camera, because world area lets a handful
  of huge distant splats dominate: the 100 largest splats hold 45% of all
  world-area weight. Under the world-area definition the floater share is
  99.3%, which would have tripped NO-GO on that one number.

### Cleanup (2026-10-03)
`scripts/splat_clean.py` writes `out/splat_clean.ply` beside the original,
which it leaves untouched. It removes exactly the floaters `splat_report.py`
counts (same function) and rotates the model about the origin so the fitted
floor is level. Run 1: 180,103 removed (17.9%), 826,069 kept (205 MB), floor
4.18 -> 0.00 degrees.

- **A rotation has to rotate more than positions.** Each splat's orientation
  quaternion is left-multiplied by the levelling rotation. Its higher-order
  spherical-harmonic colour (`f_rest_*`, channel-major) is rotated per SH
  band with matrices fitted on random directions against gsplat's own basis.
  Rotating positions alone would make colours shift as the view orbits.
  Checked on the written file: colour by view, splat covariances and
  positions all match the original to float32 precision.
- **The crop is an axis-aligned box** around the SfM cloud (1st-99th
  percentile, grown 25%), so part of the pit and seating beyond the walls
  stays. A tighter arena-only crop needs a decision on whether that content
  is wanted.
- **`splat_report.py` does not apply to the cleaned file.** The report maps
  cameras into the export's original frame; the cleaned file is levelled.
  Grade the original export and clean afterwards.

## Run 2 timing (measured 2026-10-05, from the job logs)
| Run | Step time at full training size | 30,000 steps |
|---|---|---|
| Run 1: defaults, 1280x720 | 14 ms | 11 min |
| Run 2a: three settings + `color-corrected-metrics`, 1280x720 | 247 ms | 1 h 54 min |
| Run 2b: same, 2560x1440 | 1,224 ms | 8 h 25 min; the whole job just under 9 h (rerun, job 12679934, 2026-10-08) |

- **The step time moves in three stages** because splatfacto trains at a
  quarter of the image size until step 3,000, half until 6,000, and full
  size after (`num_downscales=2`, `resolution_schedule=3000`). Run 2a: 32,
  73, then 247 ms. Run 2b: 68, 246, then 1,221 ms. After step 6,000 it is
  flat and does not follow the Gaussian count, so the cost is per pixel.
- **`color_corrected_metrics=True` is the cost.** With it on,
  `get_metrics_dict` calls `color_correct` on every training step: 15
  least-squares solves over every pixel of the image. Run 2a's profiler
  shows it directly: scoring one held-out image took 0.256 s against
  0.021 s in run 1, and that path uses neither the bilateral grid nor pose
  refinement. The extra 0.235 s matches the extra 233 ms per training step.
- **The flag only reports a number.** It is not part of the loss, so
  training without it gives the same model. The colour-corrected scores can
  be computed afterwards from the saved real-vs-render pairs instead.
  Without it, a full-size run should take a fraction of the 9 hours; that is
  an estimate, not a measurement.
- **Projection for run 2b as configured:** 2 h for the first 10,000 steps
  plus 20,000 x 1.22 s = 6.8 h, about 9 h, then export and eval. Inside the
  12-hour limit with about 3 hours to spare.

## Run 2a findings (2026-10-05)
`run2a` = run 1's data and size (1280x720) with camera pose refinement, the
bilateral grid and scale regularization on. Graded by `splat_report.py` from
`out/splatfacto/run2a/` (job 12658961). Same 28 held-out views as run 1.

| | Run 1 | Run 2a |
|---|---|---|
| Gaussians exported | 1,006,172 | 898,595 |
| Held-out PSNR / SSIM / LPIPS | 22.00 / 0.808 / 0.348 | 20.48 / 0.762 / 0.333 |
| Held-out PSNR, colour-corrected | 23.04 | 22.23 |
| Training-view PSNR | 29.0 | 31.8 |
| Needle-shaped splats | 5.4% | 0.0% |
| Splats in walked air: count / visual weight | 0.45% / 15.9% | 0.24% / 6.0% |
| Floaters: count / visual weight | 17.9% / 20.2% | 22.5% / 15.2% |
| Outside the mapped region, count | 15.1% | 20.2% |
| Floor: cover / tilt | 90.1% / 4.2 deg | 88.9% / 4.7 deg |
| **Verdict** | MARGINAL | **MARGINAL** |

- **The plain scores went down, and two measurement effects explain it.**
  Both were expected (see `submit_run2.sh`), and both are now measured.
  (1) Colour: the bilateral grid corrects training photos only.
  Colour-correcting each render before scoring recovers 1.75 dB for run 2a
  and 1.04 dB for run 1. `splat_report.py` now reports this for every run
  from the saved pairs; its figure for run 2a (22.23) matches `ns-eval`'s
  own `cc_psnr` (22.24). (2) Pose: refinement moved the training cameras by
  0.12 degrees on average and 0.56 at most, and the held-out cameras are
  not refined. Matched points between each render and its photo sit a median
  0.6 px apart in run 1 and 2.2 px in run 2a (up to 7.7 px), which is what
  those angles predict at this focal length.
- **With both taken out, run 2a is slightly ahead.** Warping each render
  onto its photo with a homography, then colour-correcting: 23.64 -> 24.00 dB
  over the 25 views that could be aligned in both runs. Interior views
  24.61 -> 24.78 (21 views); through-the-glass views 18.54 -> 19.93 (4
  views). This check was done by hand and is not in `splat_report.py`. A
  homography cannot remove a sideways camera shift exactly, so it
  under-corrects.
- **What got better.** Needle-shaped splats are gone. The splats in the
  walked air carry 6% of the visual weight, down from 16%. The foreground
  blob is gone from `frame_00180` (20.0 -> 26.0 dB aligned) and `00110`. The
  large dark blobs in the through-the-glass views are gone.
- **What did not, or got worse.** `frame_00140`, `00150` and `00220` still
  have a foreground blob. New colour artifacts: floaters with a green cast
  (`00140`, `00150`, `00170`), a teal patch on the floor (`00070`), a ghost of
  the cage over the floor (`00120`). The through-the-glass views trade dark
  blobs for a pale ghost of the arena interior and stay under 21 dB. More
  haze sits outside the mapped region. The sharpest interior views score
  lower even after alignment (`00100`: 30.4 -> 26.9 dB).
- **Net: a modest, mixed gain, not a step change.** The through-the-glass
  views remain the weakest part, which training settings alone have now
  been shown not to fix.

## Run 2b findings (2026-10-08)
`run2b` = run 2a's settings at full size (2560x1440). Rerun as job 12679934:
30,000 steps, exit status 0, 846,007 Gaussians, 831,880 exported. Graded by
`splat_report.py` from `out/splatfacto/run2b/`. Same 28 held-out views.

**Full size gave no measurable gain over half size, at 4.7 times the job
time** (just under 9 h against run 2a's 1 h 54 min). The image size stays
1280x720.

`ns-eval` scored run 2b at 2560x1440 and the others at 1280x720, so its own
figures do not compare. All three were rescored at one size from the saved
real-vs-render pairs (by hand, not in `splat_report.py`; the per-view numbers
are in `out/splatfacto/run2b/export/report/like_for_like.json`).

| 28 held-out views | Run 1 | Run 2a | Run 2b |
|---|---|---|---|
| As `ns-eval` scored it: PSNR / SSIM / LPIPS | 22.00 / 0.808 / 0.348 | 20.48 / 0.762 / 0.333 | 20.55 / 0.832 / 0.387 |
| At half size: PSNR | 22.00 | 20.48 | 20.62 |
| At half size: PSNR, colour-corrected | 23.04 | 22.23 | 22.27 |
| At half size: SSIM (torchmetrics) | 0.798 | 0.751 | 0.758 |
| At full size, runs 1 and 2a enlarged 2x: PSNR | 21.89 | 20.49 | 20.55 |
| At full size: PSNR, colour-corrected | 22.92 | 22.23 | 22.18 |
| Half size, render aligned to its photo, colour-corrected (24 views) | 24.22 | 24.56 | 24.44 |
| Interior 22 views, half size: PSNR | 23.42 | 21.79 | 21.94 |
| Through-the-glass 6 views, half size: PSNR | 16.78 | 15.68 | 15.76 |
| Seven foreground-blob views, half size: PSNR | 19.76 | 20.30 | 20.57 |
| Views at 25 dB or above | 9 | 1 | 1 |
| Render-to-photo offset, median px at half size | 0.65 | 2.16 | 2.0 |

| Splat file | Run 1 | Run 2a | Run 2b |
|---|---|---|---|
| Gaussians exported | 1,006,172 | 898,595 | 831,880 |
| Training-view PSNR | 29.0 | 31.8 | 31.4 |
| Held-out PSNR within 0.25 dB of its best from step | 11,000 | 15,000 | 10,000 |
| Needle-shaped splats | 5.4% | 0.0% | 0.0% |
| Splats in walked air: count / visual weight | 0.45% / 15.9% | 0.24% / 6.0% | 0.08% / 5.0% |
| Floaters: count / visual weight | 17.9% / 20.2% | 22.5% / 15.2% | 19.9% / 17.4% |
| Isolated splats, visual weight | 4.1% | 8.7% | 11.9% |
| Full box | 37.8 x 39.1 x 26.3 | 32.7 x 35.2 x 26.8 | 50.2 x 56.3 x 37.8 |
| Floor: cover / tilt | 90.1% / 4.2 deg | 88.9% / 4.7 deg | 89.1% / 5.0 deg |
| Camera refinement: mean / max | none | 0.12 / 0.56 deg | 0.11 / 0.44 deg |
| **Verdict** | MARGINAL | MARGINAL | **MARGINAL** |

- **2b against 2a, view by view at half size: +0.14 dB plain (standard error
  0.13) and +0.03 dB colour-corrected.** Better in 16 of 28 views. Inside the
  noise.
- **`ns-eval`'s higher SSIM (0.832) and worse LPIPS (0.387) are size
  effects.** At the same size SSIM is 0.758 against 0.751.
- **The full-size render holds no more fine detail than run 2a's render
  enlarged.** Edge energy relative to the full-size photo: 0.68 for run 2b,
  0.70 for run 2a enlarged. In full-size crops the two cannot be told apart
  (`report/detail.jpg`). The photos are the limit: they are video frames, soft
  from motion and compression, so the extra pixels carry little.
- **What differs from 2a.** Better: the green cast is gone from `00140`,
  `00150` and `00170` (`00140` +2.0 dB, `00150` +0.8), and the teal patch on
  `00070` is smaller. Worse: `00220`'s foreground blob is larger and darker
  (-1.3 dB), `00393` has new junk at the top (-1.4), `00363` gains a grey
  smear over its right half. The cage ghost on `00120` remains. More haze far
  out: the full box is half as large again.
- **The through-the-glass views are unchanged**: 15.8 dB, all six bad.
- **The last 20,000 steps bought nothing on held-out views**: the score was
  within 0.25 dB of its best from step 10,000, which is 6.8 of the 8.4 hours.
- The like-for-like aligned figures here (24.22 / 24.56 for runs 1 / 2a over
  24 views) differ a little from the 2026-10-05 hand check (23.64 / 24.00 over
  25 views): different crop and matcher, same direction and size of gap.
- Comparison sheets, photo beside all three runs:
  `out/splatfacto/run2b/export/report/compare_1.jpg` to `compare_4.jpg`.

## Run 3 (planned 2026-10-05; approved and prepared 2026-10-12; not yet run)
Train on the interior photos only. Same photos, same camera poses, no new
frames and no SfM redo: the only change is which photos training uses. Added
on 2026-10-12: a fourth arm on a denser photo set cut from the same stretch
of video, which does need its own camera solve.

Decided by the user on 2026-10-12: the interior-only scope is accepted (see
Costs and limits); all three settings arms run; a denser set at about five
photos a second is added; everything goes up in one CARC visit, together with
the augment full run.

### Why (measured 2026-10-05 on `interior400` and the run 1 / 2a splats)
- **The registered frames are two groups.** Frames 31-252 (222 frames, video
  124-227 s) are the walk inside the arena, cameras pitched 10 to 25 degrees
  down. Frames 356-404 and 420-428 (58 frames, 276-310 s) come after the
  103-frame stretch that never registered; their cameras point level or up
  (median pitch -4 and +13 degrees) through the polycarbonate.
- **The two groups are barely tied together.** Of 30,937 SfM points, 218 are
  seen by both groups; 3,927 are seen only by the late group. The late
  cameras' positions relative to the interior hang on those 218 points.
- **The late group carries most of the foreground junk.** Of the visual
  weight of the splats sitting in the walked air, 77% is next to the 58 late
  cameras in run 1 and 90% in run 2a. They are 21% of the cameras.
- **Their held-out views are the bad ones**: 16.8 dB mean in run 1 against
  23.4 dB for the interior views. Run 2a showed training settings do not fix
  them.
- This is evidence by association: it shows where the junk sits, not that
  removing the late frames cleans the interior views. Arm 3a is the test.
- **Training uses 200 of the video's 12,702 frames**, which the user
  questioned (2026-10-12). Where they go: 6,000 frames are the interior
  stretch (110-310 s; the rest is floor close-ups and ceiling);
  `process_video.py` kept one sharp frame per 0.47 s, 428; SfM placed 280; 222
  of those are inside the arena; 22 are held out. Only the 0.47 s spacing was
  a choice, and it is at the sparse end. Arm 3d tests it.

### Design
- **A second dataset file beside the original**,
  `data/processed/interior400/transforms_interior.json`, passed to
  `ns-train` as `--data`. nerfstudio accepts a `.json` there and honours
  `train_filenames` / `val_filenames` / `test_filenames` in it (all three
  are required; training reads `val`, `ns-eval` reads `test`).
  - `frames`: all 280, unchanged. nerfstudio orients and scales the scene
    from every frame listed, before applying the lists, so keeping all 280
    keeps run 1's model frame exactly. Deleting the late frames from the
    file instead changes the frame (scale 0.2199 against 0.2103).
  - `train_filenames`: the 200 frames of 31-252 that run 1 trained on.
  - `val_filenames` = `test_filenames`: run 1's 22 interior held-out frames
    (`00040`, `00050` ... `00250`). The 6 late held-out views are dropped.
  - `ply_file_path`: `sparse_pc_interior.ply`, the 27,010 SfM points seen by
    at least one interior frame. The 3,927 late-only points are not seeded.
- Checked in a scratch copy with nerfstudio's own dataparser: the file
  loads, the three lists give 200 / 22 / 22 images from `images_2/`, and the
  model frame matches run 1's `dataparser_transforms.json` exactly.
- **Arms**, parallel jobs at 1280x720, 30,000 steps,
  `color-corrected-metrics` off (`splat_report.py` computes it):

  | Label | Dataset | Settings | Compares with | Why |
  |---|---|---|---|---|
  | `run3a` | `interior400`, pinned lists | run 1's defaults | run 1 | Isolates the photo-list change. No colour or pose effect on the scores |
  | `run3b` | same | 3a + scale regularization | `run3a` | The one run 2 setting with a clear gain and no scoring side effect |
  | `run3c` | same | run 2a's three settings | `run3b`, run 2a | Whether pose refinement and the bilateral grid help once the late frames are gone. Needs colour- and alignment-corrected scoring |
  | `run3d` | `interior_dense` | 3b's | `run3b` | Whether more of the video helps. Isolates photo density |

- Each arm should take about as long as run 1 (11 minutes of training);
  `submit_run3.sh` asks for 2 hours on the `sbatch` line, not `train.job`'s 12.

### Baselines on the 22 interior held-out views
| | Run 1 | Run 2a |
|---|---|---|
| PSNR, mean | 23.42 dB | 21.79 dB |
| PSNR, colour-corrected | 24.31 dB | 23.19 dB |
| Views at 25 dB or above | 9 | 1 |
| Seven foreground-blob views (`00060`, `00110`, `00120`, `00140`, `00150`, `00180`, `00220`), mean | 19.76 dB | 20.30 dB |
| Other 15 views, mean | 25.1 dB | 22.5 dB |
| Splats in walked air, visual weight | 15.9% | 6.0% |

### What counts as success (fixed before the run; rules of thumb)
For `run3a` against run 1, plain PSNR, same 22 photos:
1. The seven blob views gain at least 1.0 dB on average (to 20.8 dB or more).
2. The splats in walked air carry at most half of run 1's visual weight
   (8% or less), measured against all 280 cameras and the original cloud so
   the yardstick is run 1's.
3. The other 15 views do not lose more than 0.5 dB on average.

All three met: interior-only becomes the dataset for later runs (image size
is settled: half, see Run 2b findings). Not met: the
late frames were not the cause, and the next lever is denser SfM or masks.
The GO / MARGINAL / NO-GO bars stay as they are, applied to the 22 views.

For `run3d` against `run3b`, plain PSNR on the same 22 photos (fixed
2026-10-12): a clear gain is 1.0 dB or more on average, a modest one 0.5 to
1.0, and under 0.5 is none. Runs 2a and 2b, which should score alike, differ
by 0.14 dB with a standard error of 0.13.

### Costs and limits
- **The model loses what only the late frames saw**: the upper cage, the
  roof and the pit seen looking up and out. Accepted by the user on
  2026-10-12. The earlier splats, which have them, are kept.
- 200 training photos instead of 252.
- Scores over all 28 views (runs 1, 2a) do not compare with scores over 22.
  Comparisons use the 22.
- The camera poses are unchanged, so any pose error inside the interior
  group stays.

### The denser set, `interior_dense` (built 2026-10-12)
`scripts/interior_dataset.py dense`. The same stretch of video, 110 to 232 s,
at five photos a second: every third frame scored for sharpness, the sharper
of each pair kept.

| | `interior400` | `interior_dense` |
|---|---|---|
| Video stretch | 110-310 s | 110-232 s |
| Spacing | 0.47 s | 0.2 s |
| Photos extracted | 428 | 555 (533 new + the 22 held-out) |
| Placed by SfM | 280 (65.4%) | 532 (95.9%) |
| Training / held out | 252 / 28 (run 3: 200 / 22) | 510 / 22 |
| SfM points | 30,937 (27,010 seen from inside) | 74,003 |
| Reprojection error | 0.935 px | 0.988 px |
| Images per point | 4.93 | 6.32 |
| `sfm_report.py` | MARGINAL | **GO** |
| Wall clock | 17.2 min | 33 min |

- **Same test photos.** Run 1's 22 interior held-out photos are copied in as
  files, at their place in the walk, and SfM placed all 22. Their half-size
  versions are pixel-identical to the ones runs 1 and 2 were scored on.
- **The test is not made easier.** A new photo closer in time to a held-out
  photo than that photo's nearest training neighbour in the interior-only set
  is left out (77 of 610 picks). Checked from the written lists: the nearest
  training photo is as far as today's for 11 held-out photos and farther for
  the other 11.
- **It reaches further than `interior400` did.** Placed from 114.7 s on;
  `interior400` placed nothing before 124 s. The first 23 photos (110-114 s)
  form their own 20-photo fragment and are not used.
- **COLMAP's model 0 is not the largest one.** It numbers sub-models in the
  order it starts them. Here model 0 was a two-photo false start and the
  532-photo model was number 1, so the pipeline, which reads `sparse/0`,
  reported 2 of 555 and the gate said NO-GO. `process_video.py` now moves the
  largest sub-model to `sparse/0` before the final bundle adjustment
  (`largest_first`). This dataset was finished from the solve that had
  already run: the swap, the bundle adjustment and the dataset files, with
  the same functions, not a second 33-minute solve.
- **New model frame.** Its own position and scale, so picture scores on the
  22 photos compare with the other arms directly and floater figures only
  roughly.
- Where each `interior400` photo sits in the video was not stored. It is made
  again from the blur scores and checked against the three statistics in
  that set's `run.json`, which match exactly.

### Steps
1. ~~Dataset file (local).~~ Done 2026-10-12: `interior_dataset.py lists`.
   280 frames listed; 200 / 22 / 22; held-out names equal run 1's first 22;
   nerfstudio loads all three splits; model frame identical to run 1's;
   27,010 seed points.
2. ~~Denser set (local).~~ Done 2026-10-12, see above.
3. ~~Job scripts.~~ Done 2026-10-12. `train.job` takes an optional
   `TRANSFORMS` file name (default: today's behaviour). `submit_run3.sh`
   checks the inputs are on CARC, skips labels that exist, skips `run3d`
   with a message if its photos are missing, and submits the four arms with
   a 2-hour limit. Dry-run of the real scripts against stand-in commands;
   nerfstudio's own parser accepts each arm's arguments.
4. ~~Report.~~ Done 2026-10-12. `splat_report.py` reads the held-out names
   from the dataset file the run trained from when it pins them, and adds a
   per-view score with the render laid over its photo first (SIFT matches,
   homography, then colour correction), which is what makes a
   camera-refined arm comparable. Rerun on runs 1, 2a and 2b: no existing
   figure moved. Over the 24 views matched in all three: 24.24 / 24.57 /
   24.36 dB.
5. **User.** Commit and push; copy `splatting/out/carc-run3-inputs.tar.gz`
   (60 MB: the two interior files, and the denser set's half-size photos,
   poses and seed points) to CARC and unpack it; one submit command; bring
   the four run folders back. Commands in `carc.md`, Run 3.
6. **Grade.** Report per arm, comparison sheets against run 1 and 2a on the
   22 views, the success criteria above, `splat_clean.py` on the best arm,
   docs.

### After run 3, not part of it
- ~~Denser frames.~~ Moved into run 3 as arm `run3d` (2026-10-12).
- **Masks** for the background seen through the walls. splatfacto applies a
  per-image mask to the loss; making the masks is the work.
- ~~Full size.~~ Dropped 2026-10-08: `run2b` showed no gain (Run 2b findings).
- **Dropping blurred or shadowed interior frames.** Not in run 3, to keep
  one change per arm.

## Risks to reconstruction quality
- **Arena surfaces are adversarial for SfM.** Painted steel floor is
  low-texture; polycarbonate walls are specular and partly transparent. These
  are known COLMAP feature-matching failure modes, not merely harder cases.
  Partly confirmed: the floor segment barely reconstructs.
- **Coverage and parallax.** A single walking path at head height reconstructs
  well near that path only; off-path viewpoints are weak. The capture is one
  pass with no orbiting.
- **No loop closure** (see Toolchain incompatibilities), so a long walk will
  drift and may fragment into multiple sub-models.

## Environment constraints (macOS)
- Apple Silicon, macOS 27.x. **No CUDA.** This single constraint is what splits
  the pipeline into a local-CPU front half (Phases 0-1) and a remote-GPU back
  half (Phase 2).
- Installed COLMAP is 4.2.0 built **without GPU support**. Feature matching
  runs on CPU: slower, not broken. Roughly 2.5 s/image for SIFT extraction at
  2560x1440.
- Use COLMAP's **sequential** matcher, not exhaustive. The input is one
  continuous video walk, so sequential matching is both correct and materially
  cheaper -- which matters on CPU.

## Resolved: `fpsample` wheel build failure
Worth recording because the diagnosis is expensive to re-derive and the
underlying trap is still on disk.

- **Symptom.** `pip install nerfstudio` aborted building the `fpsample` wheel:
  `fatal error: 'cstddef' file not found` while compiling a pybind11 extension.
- **Root cause -- not a Python packaging problem at all.** The active toolchain
  was `/Library/Developer/CommandLineTools`, whose bundled libc++ overlay at
  `usr/include/c++/v1/` contained only 11 internal `__`-prefixed stub headers
  (mtime Nov 2023) despite the package reporting version 27.0. clang's include
  search used that directory and never reached the SDK's complete header set at
  `SDKs/MacOSX.sdk/usr/include/c++/v1/`, where `cstddef` did in fact exist.
  Confirmed by a bare `#include <cstddef>` failing standalone, outside any build
  system -- so *every* C++ extension build would have failed identically, not
  just `fpsample`.
- **Resolution -- resolved externally, without intervention.** Between
  2026-09-12 and 2026-09-17 a full `/Applications/Xcode.app` was installed
  (replacing `Xcode-old.app`, Xcode 26.6) and `xcode-select` switched to it;
  the OS went 27.0 -> 27.2 and clang 2100.3.33.1 -> 2100.3.34.2. The include
  path now resolves through the SDK's complete libc++. Verified: `fpsample`
  1.0.2 builds and installs, and a multi-header STL compile exits 0.
- **Latent trap, still present.** The stale Command Line Tools tree remains on
  disk with the same 11-header stub, and `softwareupdate --list` still offers
  "Command Line Tools for Xcode 27.0" (Recommended, ~507 MB). Nothing uses that
  path while Xcode.app is the active toolchain, but pointing `xcode-select` back
  at it -- via `xcode-select --reset`, or by moving/deleting Xcode.app --
  reproduces the failure exactly. Installing that pending update would clear the
  trap. NOT done: it needs sudo and was not authorized.

## Current state
- `splatting/` holds `requirements.txt`, `mds/{AGENTS,context,carc}.md`,
  `scripts/{process_video,sfm_report,interior_dataset,splat_report,splat_clean}.py`,
  `scripts/carc/` (Phase 2 remote ops), and `data/`.
- `data/` is git-ignored (`.gitignore` line 221), so processed runs stay out of
  version control. `mds/` is tracked, matching `vision/mds/`.
- `data/processed/interior400/` holds the dataset runs 1 and 2 trained on:
  428 extracted frames, 280 posed, ~300 MB including the 2x/4x/8x
  downscales. Beside its `transforms.json` sit `transforms_interior.json`
  and `sparse_pc_interior.ply`, run 3's interior-only lists and seed points.
- `data/processed/interior_dense/` holds run 3's denser set: 555 extracted
  frames, 532 posed, 730 MB with the COLMAP files.
- `out/` (git-ignored) holds run 1 as copied down from CARC by hand:
  `out/splat.ply`, `out/splatfacto/2026-09-27_171857/` (config, checkpoint,
  tfevents), `out/report/` from `splat_report.py`, and `out/splat_clean.ply`
  from `splat_clean.py`. The local layout is flatter than CARC's
  `out/<RUN>/...`.
- Run 1 evaluated: MARGINAL, final (see Phase 3 findings). `out/eval/` holds
  the `ns-eval` metrics and renders.

### Next steps, in order
1. ~~Re-score run 1 on CARC.~~ Done 2026-10-04; the verdict is final.
2. **Look at the .ply in a browser viewer** (see `carc.md`, Viewing) to check
   the report's pictures against the real thing.
3. ~~Give each run its own output label.~~ Done 2026-10-04: the three jobs
   take a required `LABEL` and keep everything for a run under
   `out/<RUN>/splatfacto/<LABEL>/`; `train.job` now also runs the eval.
   Checked by a dry run with stand-in commands and by feeding the resulting
   `ns-train` arguments to nerfstudio's parser. Not yet run on CARC.
4. **Run 2 is chosen: `scripts/carc/submit_run2.sh`.** Two parallel jobs with
   camera pose refinement, the bilateral grid and scale regularization on:
   `run2a` at half size, `run2b` at full size. A re-shoot is not possible.
   Two of those settings act on training photos only, so held-out scores can
   drop without the result being worse; `color-corrected-metrics` is on to
   discount the colour part, and the real-vs-render pairs decide the rest.
   `splat_report.py` does not read the `cc_` scores yet.
   Status 2026-10-05: `run2a` finished (job 12658961, 917,163 Gaussians,
   898,595 exported, eval written). `run2b` was cancelled at the 2-hour
   limit at step 10,000 (job 12658962). `train.job` now asks for 12 hours
   and `submit_run2.sh` skips runs that already exist; `run2b` is to be
   resubmitted after moving its partial folder aside. See Run 2 timing.
   Status 2026-10-08: the rerun finished (job 12679934) and is graded.
5. **Run 3 is approved and prepared, not yet run: see Run 3.** Four arms:
   three on the interior photos only (frames 31-252, through a second
   dataset file with pinned photo lists, no SfM redo) and one on a denser
   photo set of the same stretch. Everything is tested locally; the user
   commits, copies one 60 MB archive to CARC and submits
   (`scripts/carc/submit_run3.sh`).
6. ~~Grade `run2b`.~~ Done 2026-10-08: MARGINAL, no gain from full size.
   Later runs stay at 1280x720. See Run 2b findings.
