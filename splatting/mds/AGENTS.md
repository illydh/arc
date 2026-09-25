# Agent instructions

See `context.md` for project background, toolchain findings, and open
decisions.

## Stack
Python 3.12, Nerfstudio (COLMAP + splatfacto), COLMAP CLI, ffmpeg, OpenCV,
NumPy. PyTorch and gsplat arrive with nerfstudio but only do anything on a
CUDA host, which this machine is not.

These conventions are not `vision/`'s. That directory is deliberately minimal
OpenCV/NumPy with no heavy dependencies; this one needs a full ML toolchain.
Neither requirements manifest should leak into the other.

## Layout
```
data/      source video, extracted frames, COLMAP output -- all git-ignored
mds/       context.md, carc.md + this file (tracked, mirroring vision/mds/)
scripts/   dev harnesses for the local CPU half
scripts/carc/  staging, SLURM job and retrieval for the remote GPU half
```
No packages yet: the local half of the pipeline is two scripts and there is
only one pipeline. Add modules when there is something real to share.

## Conventions
- Minimal, direct implementations. No speculative abstraction, no config
  options for cases that don't exist yet.
- No docstrings/comments unless they explain a non-obvious constraint. The
  toolchain mismatches below are exactly that -- do not strip those comments,
  they are the reason the code looks the way it does.
- Validate against the real capture in `data/`, not synthetic data. Arena
  surfaces (low-texture painted steel, specular polycarbonate) are the actual
  hard part.

## Toolchain constraints
nerfstudio 1.1.5 predates the installed ffmpeg and COLMAP, and is broken
against both. Do not work around this by patching `site-packages` or by
downgrading the system toolchain:

1. ffmpeg 9 removed `-vsync`, which `ns-process-data video` hardcodes. The
   replacement is `-fps_mode`.
2. COLMAP 4.2 renamed `SiftExtraction`/`SiftMatching` to
   `FeatureExtraction`/`FeatureMatching`. nerfstudio still emits the 3.x names.
3. COLMAP 4.2 reads only faiss vocab trees (it dropped flann in May 2025). The
   tree nerfstudio downloads is the legacy flann one, so vocab-tree matching
   and sequential loop detection need an explicitly supplied faiss tree.

Consequence: `ns-process-data` cannot run here at all. `scripts/process_video.py`
owns frame extraction and COLMAP, and calls nerfstudio's `colmap_to_json` for
the `transforms.json` + `sparse_pc.ply` that `ns-train splatfacto` consumes.
Nerfstudio is still the training stack; it is only the data-processing
entrypoint that is unusable.

## Running
Dependencies are in `requirements.txt`, installed into the repo-root `.venv`.
Append new ones as they are introduced.

```
.venv/bin/python splatting/scripts/process_video.py [--frames N] [--matcher M]
.venv/bin/python splatting/scripts/sfm_report.py splatting/data/processed/<run>
```

`process_video.py` writes a nerfstudio-format dataset to
`data/processed/<name>/`; `sfm_report.py` grades the reconstruction and writes
`report.json` beside it. Both are CPU-only.

Training is remote and so is the `.ply` export -- `ns-train`, `ns-viewer` and
`ns-export` all put the model on a CUDA device, so none of them run on this
machine. See `carc.md` for the SLURM runbook.

## Keeping context current
`context.md` and this file are working documents, not one-time setup.
Update them whenever:
- scope changes (a phase is added, dropped, or reordered)
- an open decision gets resolved
- a pipeline stage is completed or its design changes from what's documented
- a new toolchain constraint or assumption is found that a future session
  would otherwise have to re-derive

Stale context is worse than missing context -- don't leave a resolved open
decision or an outdated description in place.
