"""Synthetic test music with known ground truth (tempo, chords, melody, sections).

Used by the test-suite and to try scenes without a real track:
    uv run python -m vocelle.synth audio/demo.wav
"""

from __future__ import annotations

import sys

import numpy as np

SR = 22050
NOTE = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def click_track(bpm: float = 120, seconds: float = 20, sr: int = SR) -> np.ndarray:
    y = np.zeros(int(seconds * sr))
    t = np.arange(int(0.05 * sr)) / sr
    click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 80)
    for b in np.arange(0, seconds, 60 / bpm):
        i = int(b * sr)
        y[i:i + len(click)] += click[: len(y) - i]
    return y


def tone(midi: float, dur: float, sr: int = SR, harmonics: int = 4, decay: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2 * np.pi * hz(midi) * k * t) / k for k in range(1, harmonics + 1))
    env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.02)
    if decay:
        env *= np.exp(-t * decay)
    return y * env


def chord(root: str, quality: str, dur: float, octave: int = 4, sr: int = SR) -> np.ndarray:
    r = 12 * (octave + 1) + NOTE[root]
    ivs = (0, 4, 7) if quality == "maj" else (0, 3, 7)
    return sum(tone(r + i, dur, sr) for i in ivs) / 3


def song(bpm: float = 110, sr: int = SR) -> tuple[np.ndarray, dict]:
    """~35 s: intro (pads) -> verse (drums+bass) -> chorus (louder, melody) -> verse."""
    beat = 60 / bpm
    bar = 4 * beat
    prog_a = [("C", "maj"), ("G", "maj"), ("A", "min"), ("F", "maj")]
    prog_b = [("F", "maj"), ("C", "maj"), ("G", "maj"), ("E", "min")]
    plan = [("intro", prog_a, 0.0), ("verse", prog_a, 1.0), ("chorus", prog_b, 1.6), ("verse", prog_a, 1.0)]
    total = len(plan) * len(prog_a) * bar
    y = np.zeros(int(total * sr) + sr)
    truth = {"bpm": bpm, "chords": [], "sections": []}
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    rng = np.random.default_rng(0)
    snare = rng.normal(0, 1, int(0.15 * sr)) * np.exp(-np.arange(int(0.15 * sr)) / sr * 25) * 0.5
    melody_a = [72, 74, 76, 79, 76, 74, 72, 71]
    melody_b = [77, 79, 81, 84, 83, 79, 76, 74]
    pos = 0.0
    for name, prog, level in plan:
        truth["sections"].append({"start": pos, "label": name})
        for ci, (root, q) in enumerate(prog):
            i0 = int(pos * sr)
            seg = chord(root, q, bar, sr=sr) * 0.35
            y[i0:i0 + len(seg)] += seg
            truth["chords"].append({"start": pos, "end": pos + bar, "label": root + ("m" if q == "min" else "")})
            if level > 0:
                b = tone(12 * 3 + NOTE[root], bar, sr, harmonics=3) * 0.35 * level
                y[i0:i0 + len(b)] += b
                for k in range(4):
                    j = int((pos + k * beat) * sr)
                    y[j:j + len(kick)] += kick * 0.9 * level
                    if k % 2:
                        y[j:j + len(snare)] += snare * level
            if level > 1.2:
                mel = melody_b if name == "chorus" else melody_a
                for k in range(2):
                    note = mel[(ci * 2 + k) % len(mel)]
                    m = tone(note, bar / 2, sr, harmonics=2, decay=0.5) * 0.3
                    j = int((pos + k * bar / 2) * sr)
                    y[j:j + len(m)] += m
            pos += bar
    y = y[: int(total * sr)]
    return (y / np.max(np.abs(y)) * 0.9).astype(np.float32), truth


if __name__ == "__main__":
    import soundfile as sf

    out = sys.argv[1] if len(sys.argv) > 1 else "audio/demo.wav"
    y, truth = song()
    sf.write(out, y, SR)
    print(f"wrote {out} ({len(y) / SR:.1f}s, {truth['bpm']} BPM)")
