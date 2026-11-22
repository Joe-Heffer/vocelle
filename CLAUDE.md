# vocelle: notes for Claude Code

Pipeline: `vocelle analyze` (Python 3.12 venv via uv) → `analysis/<track>.json` → `vocelle render`
which shells out to `blender -b --factory-startup --python src/vocelle/blender/run.py -- ...`.

## Two Pythons
- `src/vocelle/analysis/`, `cli.py`, `synth.py` run in the uv venv (librosa, numpy, scipy).
- `src/vocelle/blender/`, `src/vocelle/scenes/` run **inside Blender's bundled Python** (bpy + numpy
  only). Never import librosa/scipy there. `features.py` is stdlib-only so both sides can use it.

## Conventions
- Curves are per video frame, 0..1 (except `bar_index`, `section_index`, `section_label`, `melody_midi`).
  Curve index `i` ↔ Blender frame `i + 1` ↔ time `i / fps` (`drive.FRAME_OFFSET`).
- Changing the feature file format → bump `features.SCHEMA_VERSION` and update docs/FEATURES.md.
- Scenes get music in three ways (see `blender/drive.py`): keyframed custom props on the `VOCELLE`
  empty, drivers (`drive.add_driver`, simple expressions only), and **history meshes** sampled in
  Geometry Nodes with Sample Index (lets geometry read the past: travelling waves, trails).
- Build node trees with `blender/nodes.py` `Tree` helpers; Blender 5.x APIs: compositor is
  `scene.compositing_node_group`, actions are layered (use `anim_utils.action_get_channelbag_for_slot`),
  Curve to Mesh needs an explicit `Scale` input for thickness, Glare type is a menu socket.
- Visual iteration: render stills (`vocelle render ... --still 9 16 --res 960x540`) and view the PNGs
  before rendering video. `--save-blend` writes the built scene for inspection in the Blender UI.

## Commands
- `uv run pytest`, `uv run ruff check src tests`
- `uv run python -m vocelle.synth audio/demo.wav`: synthetic song with known tempo/chords/sections

Hardware here: AMD RX 580 (Polaris). EEVEE works headless via Vulkan; Cycles GPU (HIP) does not
support Polaris, so Cycles would be CPU-only.

## Render performance
Per-frame cost is dominated by single-threaded Geometry Nodes/depsgraph work, not shading, so one Blender
process leaves the GPU and most cores idle. `vocelle render --workers N` splits the frame range into N
chunks (`cli.split_frames`), renders each in its own Blender process (video-only, `--frame-start/--frame-end/
--no-audio/--threads`), then stitches with ffmpeg concat and muxes audio. Finished chunks are kept in
`renders/.<out>-chunks/` (`.done` markers) so an interrupted run resumes. Each worker loads the scene, so VRAM
bounds N. Measured on the RX 580 box: 6 s preview @960x540, 166 s (1 worker) vs 67 s (3 workers).
