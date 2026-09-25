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
- ~~GPU target for training~~ **resolved: USC CARC (SLURM).** Not built yet --
  Phase 2 is the next round of work. Cloud rental (RunPod/Lambda) remains the
  fallback if queue latency or environment setup proves painful.
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
- **Phase 2 -- training, GPU only.** `ns-train splatfacto` on USC CARC via
  SLURM. Requires a CUDA host; cannot run locally (verified: `gsplat: No CUDA
  toolkit found. gsplat will be disabled.`). **Scaffolding written, not yet
  executed** -- `scripts/carc/` + the runbook in `carc.md`. No CARC access
  exists from this machine, so every cluster-specific value in those scripts
  is a marked placeholder.
- **Phase 3 -- viewing.** Corrected: this is **not** local the way it was
  originally planned. `ns-viewer` renders through gsplat and `ns-export` loads
  the model onto a CUDA device, so **neither runs on the Apple Silicon box**.
  The export therefore happens on CARC at the end of the training job, and
  what comes back is `export/splat.ply`, which opens in a browser WebGL splat
  viewer with no CUDA and no install. `pymeshlab` is consequently needed in
  the **CARC** environment (`ns-export` imports it at module load), not
  locally -- it is in `scripts/carc/setup_env.sh`.

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
  frames, 280 posed, ~300 MB including the 2x/4x/8x downscales. It is what
  Phase 2 should be pointed at.
- Nothing trained yet. No GPU work started.

### Next steps, in order
1. **Run Phase 2.** Work through `carc.md`: preflight on the login node to
   fill in the account/partition/module placeholders, `setup_env.sh`,
   `stage.sh interior400`, a 500-iteration smoke job, then the real run.
2. **Look at the .ply before improving anything.** The Phase 1 gate was
   MARGINAL, but whether 65% registration and the through-the-glass gap
   actually matter is only answerable by looking at a render. Do not spend
   more CPU on SfM speculatively.
3. Only if the splat looks poor, revisit the front half: exclude the
   through-the-glass stretch (roughly `--start 110 --end 228`), or raise
   density on the segments that did register.
