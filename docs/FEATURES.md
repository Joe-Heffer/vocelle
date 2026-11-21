# Feature file (`analysis/<track>.json`), schema v1

Top level:

| key | description |
|---|---|
| `schema_version`, `track`, `fps`, `duration`, `n_frames` | metadata. Curves have `n_frames` values |
| `stems` | stem names analysed (empty if mix only) |
| `rhythm` | `{source, tempo, beats_per_bar, beats[], downbeats[]}` (seconds) |
| `key` | `{tonic, tonic_pc, mode, confidence}` |
| `chords[]` | `{start, end, root, root_pc, quality: maj/min, label, rgb}` |
| `melody` | `{source, notes[{start,end,pitch,velocity}], range_midi}` |
| `sections[]` | `{index, start, end, label: A/B/C…, energy 0..1}`. Equal labels = repeated material |
| `drops[]` | section starts with a big energy jump |
| `onsets[]`, `<stem>_onsets[]` | onset times |
| `curves{}` | name → per-frame list |

## Curves

| name | meaning |
|---|---|
| `<src>_rms` | loudness (dB-normalised, attack/release smoothed). `<src>` = `mix` or a stem name |
| `<src>_sub/low/mid/high` | band energies (20–80, 80–250, 250–2k, 2k–12k Hz) |
| `<src>_flux` | spectral flux (onset strength) |
| `<src>_bright` | spectral centroid (timbre brightness) |
| `<src>_onset` | decaying impulse at each onset, weighted by strength |
| `beat_phase`, `bar_phase` | 0→1 sawtooth through each beat / bar. Use for cyclic motion locked to the groove |
| `beat_pulse`, `downbeat_pulse` | decaying impulses on beats / downbeats |
| `downbeat_windup` | ease-in ramp during the beat *before* each downbeat (anticipation) |
| `bar_index` | integer bar number |
| `chord_r/g/b` | chord colour (root on circle of fifths → hue; minor darker) |
| `chord_major` | 1 major, 0 minor |
| `tension` | harmonic tension: root distance from tonic on circle of fifths + chroma roughness |
| `chroma_flux` | rate of harmonic change |
| `chroma_00..11` | pitch-class energies (C..B), raw (normalise per frame if needed) |
| `melody_pitch` | melody contour normalised to the melody's own range (held through rests) |
| `melody_voiced` | probability the melody is sounding |
| `melody_note_on` | impulse at each note start |
| `melody_midi` | raw MIDI pitch (not normalised) |
| `section_phase` | 0→1 through the current section |
| `section_index`, `section_label` | integer section number / label index (A=0) |
| `section_energy` | relative loudness of the current section |
| `section_change` | impulse at each section boundary |
| `energy` | slow song-level energy |
| `energy_build` | look-ahead: how much louder the next 8 s are than the last 8 s (peaks before drops) |
| `drop_pulse` | long impulse at each drop |
