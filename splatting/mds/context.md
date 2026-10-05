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
  `scripts/{process_video,sfm_report}.py`, `scripts/carc/` (Phase 2 remote
  ops), and `data/`.
- `data/` is git-ignored (`.gitignore` line 221), so processed runs stay out of
  version control. `mds/` is tracked, matching `vision/mds/`.
- `data/processed/interior400/` holds the one usable dataset: 428 extracted
  frames, 280 posed, ~300 MB including the 2x/4x/8x downscales. Run 1 was
  trained on it.
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
3. **Give each run its own output label** in `train.job`, `export.job` and
   `eval.job` before run 2. As written, run 2 overwrites run 1's `export/`
   and `eval/` on CARC and the "most recent run" lookup stops finding run 1.
4. **Then choose run 2's changes.** The findings point at the through-the-glass
   views and the foreground floaters, not the arena interior. Train-only
   options, aimed at the floaters: camera pose refinement, per-image
   appearance correction, a needle penalty. Front-half options, which need
   SfM redone and are aimed at the through-the-glass views: exclude or mask
   that stretch (roughly `--start 110 --end 228`), or raise density on the
   segments that did register. A train-only run keeps the same 28 held-out
   views, so it compares directly with run 1. Redoing SfM changes the frame
   set, and with it the held-out views.
