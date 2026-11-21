# Research notes (2026-11)

## Prior art in this space
- **Blender native audio.** Blender 5.2 added a *Sample Sound Frequencies* Geometry Nodes node,
  and 5.3 extends it ([BlenderArtists demo](https://blenderartists.org/t/audio-visualization-using-blender-5-3-alphas-sample-sound-frequencies-node/1647493),
  [GSoC 2024 proposal](https://devtalk.blender.org/t/gsoc-2024-sample-sound-node/33927)). It reads the **spectrum only**: no beats,
  harmony or structure. The older route is *Graph Editor → Bake Sound to F-Curves* (one band per F-curve).
- **Add-ons:** [Sound Nodes](https://github.com/negdo/Sound_Nodes) (analysis → keyframes),
  [BeatDriver](https://awesome.ecosyste.ms/projects/github.com%2Fclonephaze%2Fbeatdriver-add-on-for-blender),
  [A2B](https://superhivemarket.com/products/a2b-51/docs) (realtime pitch/onset/BPM/flux into GN),
  Sound Reaktor and Pixelplay Beat Reactor (commercial). Most map *loudness/bands* to parameters.
- **Talk:** [Making a spectrogram with Geometry Nodes, Blender Conference 2025](https://www.blendernation.com/2025/10/20/making-a-spectrogram-with-geometry-nodes-by-simon-henry-at-blender-conference-2025/).
- Previous project [musicvis](https://github.com/Joe-Heffer/musicvis): Demucs + librosa → per-frame JSON →
  HyperFrames/three.js. vocelle keeps the offline-JSON idea, moves rendering to Blender and adds musical features.

## Music analysis tools
| Need | Tool | Notes |
|---|---|---|
| stems | [Demucs](https://github.com/adefossez/demucs) `htdemucs`, `htdemucs_ft`, `htdemucs_6s` | still the standard open separator; `extra: stems` |
| beats + downbeats | [Beat This!](https://pypi.org/project/beat-this/) (ISMIR 2024, [paper](https://arxiv.org/abs/2407.21658)) | SOTA without DBN, full mix; `extra: beats`. Fallback: librosa + chord-change downbeat heuristic |
| structure | [All-In-One](https://github.com/mir-aidj/all-in-one) (`allin1`), fork [`all-in-one-infer`](https://pypi.org/project/all-in-one-infer/) (2026, integrated separation) | functional labels: intro/verse/chorus. Current fallback: Foote novelty on SSM |
| notes | [Basic Pitch](https://github.com/spotify/basic-pitch) | polyphonic audio→MIDI with pitch bends; `extra: notes` |
| f0 | librosa pYIN (in use), CREPE | monophonic melody contour on vocal stem |
| chords | template + Viterbi (in use); [lv-chordia](https://pypi.org/project/lv-chordia/) (large vocab, ISMIR 2019) | upgrade path for 7ths/extended chords |
| general | librosa, Essentia, madmom | madmom is hard to install on modern Python |

## Best practices we follow
1. **Analyse offline, render deterministically.** Every frame is a pure function of the feature file.
   Look-ahead becomes possible: wind-ups before downbeats, pull-backs before drops.
2. **Separate first, then analyse each stem.** Drums drive rhythmic motion, bass drives mass/height,
   vocals drive the melody line, harmonic stems drive chords. Mixing them muddies every signal.
3. **Shape the signal.** Use dB, percentile normalisation, asymmetric attack/release, and decaying
   impulses instead of raw onsets. Raw FFT bands look jittery.
4. **Hierarchical musical time.** Beat phase, bar phase, section phase. Phase-locked motion (rotation
   per bar, breathing per beat) reads as "in the groove" much better than reacting to peaks.
5. **One musical dimension per visual channel.** Harmony → colour, rhythm → scale/light, melody →
   position, structure → camera. When everything reacts to everything, the result looks noisy.
6. **Harmony as colour space.** The circle of fifths gives a perceptually sensible hue wheel: related
   keys and chords get nearby colours, and a modulation visibly shifts the palette.
7. **Editing follows structure.** Cut on downbeats at section boundaries, and repeat framings for
   repeated sections so the video has form, not just texture.
8. **Space as time.** Geometry Nodes `Sample Index` into a per-frame "history mesh" lets each object read
   the music at a different time, giving travelling waves, trails and echoes. This is cheap and works in EEVEE.

## Rendering notes
- EEVEE (Blender 5.2) renders headless with `-b` on Linux via Vulkan/Mesa, including the RX 580 here
  (~0.2–0.5 s/frame at 540p). Cycles HIP does not support Polaris, so Cycles is CPU-only on this machine.
- Bloom moved to the compositor (Glare node, type *Bloom*). In 5.x the compositor is a node group
  assigned to `scene.compositing_node_group`.
- AgX view transform + "Punchy" look handles bright emissive colours without clipping to white.
- Audio: add the track as a VSE sound strip at frame 1 and render with FFMPEG/AAC. Partial renders
  (`--start`) stay in sync because the strip lives on the global timeline.

## Innovative directions
See issues labelled [`idea`](https://github.com/Joe-Heffer/vocelle/issues?q=label%3Aidea): Tonnetz flythrough,
tension-driven physics, stem-force particle simulations, melody sculptures, visual leitmotifs, an automatic
editing grammar, timbre embeddings → materials, groove/microtiming easing, lyric typography, stereo-field
placement, and more.
