# Vocelle

[![CI](https://github.com/Joe-Heffer/vocelle/actions/workflows/ci.yml/badge.svg)](https://github.com/Joe-Heffer/vocelle/actions/workflows/ci.yml)

A toolkit for **expressive, music-reactive 3D music videos** rendered in Blender (EEVEE).

Most audio visualisers react to *loudness*. vocelle analyses the *music*, including beats and
downbeats, tempo, key, chords, harmonic tension, melody, song sections and per-instrument stems,
and turns it into animation curves and events that drive procedural Blender scenes.

```
audio.wav ──► vocelle analyze ──► analysis/<track>.json ──► vocelle render ──► renders/<track>.mp4
              (stems, rhythm,       (per-frame curves +       (Blender -b: keyframes,
               harmony, melody,      events: beats, chords,    Geometry Nodes, EEVEE,
               structure)            notes, sections)          bloom, audio mux)
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Blender ≥ 5.2 on `PATH` (or `$BLENDER`), and ffmpeg.

```bash
uv sync                                   # core: librosa/numpy/scipy
uv sync --extra stems --extra beats       # optional: Demucs stems + Beat This! (PyTorch)

uv run python -m vocelle.synth audio/demo.wav           # a synthetic test song with known chords
uv run vocelle analyze audio/demo.wav --no-separate --sections 4
uv run vocelle render analysis/demo.json --still 9 16 22 --res 960x540     # preview stills
uv run vocelle render analysis/demo.json --start 14 --duration 10 --res 960x540 --quality preview
uv run vocelle render analysis/demo.json                                     # full 1080p video
```

For a real track, put it in `audio/` and run `vocelle analyze audio/track.wav`. Demucs runs
automatically if installed. You can also pass existing stems with `--stems DIR`.

## Demo scene: Harmonic Field

`src/vocelle/scenes/harmonic_field.py`: a disc of hexagonal columns on a mirror floor.

| Music | Visual |
|---|---|
| bass energy, kick hits | column height, **travelling outward as ripples** (each column samples the music from *distance × delay* frames ago) |
| chord root (circle of fifths) | hue of the glow, spreading outward on each chord change |
| chroma (12 pitch classes) | 12 angular sectors ordered by fifths; each swells while its note sounds |
| melody pitch / voicing | a glowing ribbon whose height traces the melody, trailing through time |
| downbeats | lens punch, overhead light flash |
| section labels | camera framing chosen per label, so a returning chorus gets the same shot |
| energy build before a drop | camera pulls back (look-ahead, offline only) |

## Layout

```
src/vocelle/analysis/   audio → features (envelopes, spectral, rhythm, harmony, melody, structure, stems)
src/vocelle/blender/    runs inside Blender: run.py entry, drive.py (keyframes/history meshes/drivers),
                        render.py (EEVEE, bloom, output), nodes.py (node-tree builder)
src/vocelle/scenes/     one module per scene, each exposing build(scene, features, ...)
docs/FEATURES.md        feature-file reference      docs/RESEARCH.md   techniques & sources
```

Ideas for future work are tracked as [GitHub issues](https://github.com/Joe-Heffer/vocelle/issues?q=label%3Aidea).
