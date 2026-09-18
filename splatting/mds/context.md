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
whose conventions (see `vision/AGENTS.md`) scope to that directory only.
Splatting needs PyTorch plus a CUDA-compiled rasterizer -- an unrelated, heavy
toolchain that should not leak into `vision/`'s dependency manifest.

## Source footage
`splatting/data/video_inside_battlebox.mp4`
- 2560x1440, 30 fps, HEVC, 423.38 s, 12,702 frames, ~953 MB.
- Handheld phone walkthrough of the arena interior.
- Unrelated to `vision/`'s fight clips (1440x762, ~98 fps, dark/noisy). Do not
  confuse the two datasets.

## Decisions
- **Raw splat over textured mesh, for now.** A mesh (photogrammetry:
  SfM + MVS + meshing) was recommended instead, because a mesh serves any
  downstream consumer -- physics sim, game engine, renderer -- whereas a splat
  is render-only, and splat -> collision geometry is a lossy open research
  problem (SuGaR and similar), not a solved conversion. User scoped to a raw
  splat anyway since no simulator target is fixed yet. **Revisit this the
  moment a simulation consumer needing collision geometry is chosen** -- the
  splat will not serve it.
- **Nerfstudio over raw COLMAP CLI + graphdeco-inria 3DGS.** Rationale:
  simplest stack was an explicit, repeated user requirement.
  `ns-process-data video` folds frame extraction, blur filtering, and COLMAP
  SfM into one command; `splatfacto` trains via gsplat, which is
  pip-installable with no hand-built CUDA submodules (the reference
  implementation requires manually compiling `diff-gaussian-rasterization` and
  `simple-knn` against a matching CUDA toolkit).
- **Frame selection is automatic.** Rule-based sampling plus Laplacian-variance
  blur rejection, both inside `ns-process-data`. No manual per-frame picking.
  An optional eyeball pass to delete obvious junk frames (lens covered, phone
  pointed at the ceiling) is a nice-to-have only.

## Open decisions
- **GPU target for training.** Unresolved. Options: pay-per-use cloud rental
  (RunPod / Lambda / Paperspace), USC on-prem HPC via SLURM, or default to
  cloud if undecided. Orchestration should be parameterized by remote target
  rather than hardcoded to one platform, since the user has access to both and
  has committed to neither.
- **Python environment location.** Repo-root `.venv` (Python 3.12.11, ~168 MB,
  effectively just numpy + the fpsample/pybind11 below) versus a dedicated
  `splatting/.venv`. Installing nerfstudio into the root venv drops the whole
  PyTorch tree into the same environment scope as `vision/`, contrary to the
  isolation intent above. `vision/` does not appear to be installed into that
  venv currently.
- **Metric scale.** Deferred, not solved. Monocular SfM recovers geometry only
  up to an unknown global scale. When scale is eventually needed, the arena
  floor is a 12x12 grid of 4ft tiles totalling 48ft x 48ft (confirmed in
  `vision/context.md`); those grid intersections are the natural scale
  reference, and are the same references `vision/calib` already uses for its
  homography. Not required for a raw "just to have" splat; required before any
  simulation use.

## Plan
- **Phase 0 -- environment.** COLMAP and ffmpeg: DONE (Homebrew). Nerfstudio:
  NOT DONE (see current state).
- **Phase 1 -- data processing, local, CPU.**
  `ns-process-data video --data splatting/data/video_inside_battlebox.mp4
  --output-dir <dir>`. Also serves as the go/no-go gate: if SfM cannot register
  the arena, it fails here, before any GPU time is bought.
- **Phase 2 -- training, GPU only.** `ns-train splatfacto --data <processed>`.
  Requires a CUDA host; cannot run locally.
- **Phase 3 -- viewing, local.** Nerfstudio's Viser web viewer against the
  checkpoint, or export `.ply` to a standalone WebGL splat viewer. WebGL only,
  no CUDA needed.

## Risks to reconstruction quality
Identified during planning, none yet tested against the footage:
- **Static-scene assumption.** Vanilla 3DGS assumes nothing moves between
  frames. People, moving equipment, or flickering lights produce floating
  "ghost" artifacts. This is a *silent* failure -- it degrades the result
  rather than raising an error. Whether this capture is of an empty, static
  box is UNCONFIRMED.
- **Arena surfaces are adversarial for SfM.** Painted steel floor is
  low-texture; polycarbonate walls are specular and partly transparent. These
  are known COLMAP feature-matching failure modes, not merely harder cases.
  Phase 1 is the real test of whether this footage is reconstructable at all.
- **Coverage and parallax.** A single walking path at head height reconstructs
  well near that path only; off-path viewpoints, floor, and ceiling will be
  weak. How much orbiting or height variation the capture contains is unknown.
- **Frame count.** Do not feed all 12,702 frames. Target a few hundred.
  Matching cost is roughly quadratic in image count, and near-duplicate frames
  contribute no additional geometry.

## Environment constraints (macOS)
- Apple Silicon, macOS 27.x. **No CUDA.** This single constraint is what splits
  the pipeline into a local-CPU front half (Phases 0-1) and a remote-GPU back
  half (Phase 2).
- Installed COLMAP is 4.2.0 built **without GPU support**, confirming the
  CPU-only path. Feature matching runs on CPU: slower, not broken.
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
- `splatting/` holds only `requirements.txt` (one line: `nerfstudio`) and
  `data/video_inside_battlebox.mp4`. No scripts, no processed data, no outputs.
- Installed in repo-root `.venv`: `fpsample` 1.0.2, `pybind11` 3.1.0, numpy.
  **`nerfstudio` itself is NOT installed** -- the original run aborted at
  `fpsample` before fetching it.
- COLMAP 4.2.0 (no GPU) and ffmpeg present via Homebrew.
- Immediate next step: settle the environment-location question above, install
  nerfstudio, then run Phase 1 as the go/no-go gate.
