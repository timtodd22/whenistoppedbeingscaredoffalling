# Rendering THE LONG RAIN on a local GPU

The cloud session that built this trailer has 4 CPU cores and no GPU, so it renders Blender
shots on twos/threes at ~90 s per frame. On an RTX-class GPU (OptiX), every shot can be rendered
at every frame with ~6x the samples in minutes.

## Requirements
- Blender **4.2 LTS** (other 4.x versions likely work)
- `ffmpeg` on PATH (for packing the shots)
- Python 3 (any; the runner only calls Blender and ffmpeg)
- Git LFS (`git lfs install`)

## Render (from the repo root, on branch `claude/wizardly-clarke-bww088`)

    git pull
    python trailer/blender/render_all.py --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"

- `BR_GPU=1` is set by the runner: OptiX (falls back to CUDA/HIP/Metal), every frame, samples x6
  (`BR_SAMPLES=10` for even cleaner frames), finer fog steps.
- One shot only: `--only city_descent` (names: shot01, shot03, city_descent, city_hologram,
  city_aerial, shot17). Re-pack without rendering: `--pack-only`.
- Frames land in `trailer/renders/<shot>/` (git-ignored); each shot is packed to
  `trailer/renders_gpu/<shot>.mov` (ProRes 4444 12-bit, tracked by Git LFS).

## Hand back

    git add .gitattributes trailer/renders_gpu
    git commit -m "GPU renders"
    git push

The cloud session then pulls the branch; `trailer/edit/assemble.py` uses `renders_gpu/<shot>.mov`
automatically in place of the CPU renders, and the final trailer is assembled, graded, muxed with
the binaural mix and delivered.

## If you are a Claude Code session on this machine
Run the render command above (ask for the Blender path if it is not obvious), check the first
packed shot plays, then commit and push `trailer/renders_gpu/` as shown. Do not change the scene
scripts unless asked; the edit, grade and audio are done in the cloud session.
