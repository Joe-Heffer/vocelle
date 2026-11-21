"""Key, chords and harmonic tension.

* Key: Krumhansl–Schmuckler profile correlation over the mean chroma.
* Chords: 24 major/minor triad templates + no-chord, cosine similarity per frame,
  smoothed with a Viterbi pass (sticky self-transitions) so labels don't flicker.
* Colour: chord root placed on the circle of fifths -> hue, so harmonically close
  chords get close colours and a modulation visibly shifts the palette.
* Tension: distance of the chord root from the tonic on the circle of fifths, plus
  a chroma "roughness" term (energy on semitone / tritone intervals).
"""

from __future__ import annotations

import colorsys

import librosa
import numpy as np

from .envelopes import norm, smooth_ar, to_frames

PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
HOP = 2048  # ~93 ms at 22.05 kHz: harmony moves slowly


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, labels = [], []
    for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
        for root in range(12):
            t = np.zeros(12)
            t[[(root + i) % 12 for i in intervals]] = 1.0
            t[root] += 0.3  # root emphasis helps maj/min disambiguation of relative chords
            rows.append(t / np.linalg.norm(t))
            labels.append((root, quality))
    return np.array(rows), labels


def fifths_index(pc: int) -> int:
    """Position of pitch class `pc` on the circle of fifths (C=0, G=1, D=2, ...)."""
    return (pc * 7) % 12


def chord_colour(root: int, quality: str) -> tuple[float, float, float]:
    hue = fifths_index(root) / 12.0
    sat, val = (0.85, 1.0) if quality == "maj" else (0.95, 0.75)
    if quality == "min":
        hue = (hue + 0.04) % 1.0  # minor chords lean slightly cooler/different
    return colorsys.hsv_to_rgb(hue, sat, val)


def estimate_key(chroma: np.ndarray) -> dict:
    mean = chroma.mean(axis=1)
    best = (-2.0, 0, "maj")
    for root in range(12):
        for quality, prof in (("maj", KS_MAJOR), ("min", KS_MINOR)):
            r = float(np.corrcoef(mean, np.roll(prof, root))[0, 1])
            if r > best[0]:
                best = (r, root, quality)
    r, root, quality = best
    return {"tonic": PITCHES[root], "tonic_pc": root, "mode": "major" if quality == "maj" else "minor",
            "confidence": round(r, 3)}


def _fifths_distance(a: int, b: int) -> int:
    d = abs(fifths_index(a) - fifths_index(b))
    return min(d, 12 - d)


def analyze_harmony(y: np.ndarray, sr: int, fps: float, n_frames: int) -> tuple[dict, dict]:
    rate = sr / HOP
    y_h = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP, bins_per_octave=36)
    chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine") if chroma.shape[1] > 20 else chroma
    key = estimate_key(chroma)

    T, labels = _templates()
    cn = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    sim = T @ cn  # (24, frames)
    energy = librosa.feature.rms(y=y_h, frame_length=HOP * 2, hop_length=HOP)[0][: chroma.shape[1]]
    silent = energy < (np.max(energy) * 0.02 if energy.size else 0)
    no_chord = np.where(silent, 1.0, 0.55)[None, :]
    scores = np.vstack([sim, no_chord])
    prob = np.exp(scores * 12.0)
    prob /= prob.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(len(labels) + 1, 0.97)
    path = librosa.sequence.viterbi(prob, trans)

    # Collapse into chord segments.
    chords = []
    times = librosa.frames_to_time(np.arange(len(path) + 1), sr=sr, hop_length=HOP)
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            s = int(path[start])
            if s < len(labels):
                root, quality = labels[s]
                rgb = chord_colour(root, quality)
                chords.append({
                    "start": round(float(times[start]), 3), "end": round(float(times[i]), 3),
                    "root": PITCHES[root], "root_pc": root, "quality": quality,
                    "label": PITCHES[root] + ("m" if quality == "min" else ""),
                    "rgb": [round(c, 4) for c in rgb],
                })
            start = i

    # Per-frame curves at video fps.
    t = np.arange(n_frames) / fps
    rgb = np.full((n_frames, 3), 0.15)
    major = np.zeros(n_frames)
    fifths = np.zeros(n_frames)
    for c in chords:
        m = (t >= c["start"]) & (t < c["end"])
        rgb[m] = c["rgb"]
        major[m] = 1.0 if c["quality"] == "maj" else 0.0
        fifths[m] = _fifths_distance(c["root_pc"], key["tonic_pc"]) / 6.0
    # Roughness: chroma energy on dissonant interval classes (m2, tritone).
    rough = np.zeros(chroma.shape[1])
    for k, w in ((1, 1.0), (6, 0.7), (2, 0.3)):
        rough += w * (cn * np.roll(cn, -k, axis=0)).sum(axis=0)
    rough_f = norm(to_frames(rough, rate, fps, n_frames))
    tension = smooth_ar(0.6 * fifths + 0.4 * rough_f, fps, 0.15, 0.6)

    sm = lambda x: smooth_ar(x, fps, 0.08, 0.08)  # noqa: E731 - soften colour cuts slightly
    curves = {
        "chord_r": sm(rgb[:, 0]), "chord_g": sm(rgb[:, 1]), "chord_b": sm(rgb[:, 2]),
        "chord_major": sm(major),
        "tension": norm(tension, 0, 100),
        "chroma_flux": norm(to_frames(np.r_[0, np.abs(np.diff(cn, axis=1)).sum(0)], rate, fps, n_frames)),
    }
    # 12 chroma curves too, for per-pitch-class visuals (e.g. 12 objects on a circle of fifths).
    chroma_f = np.stack([to_frames(chroma[p], rate, fps, n_frames) for p in range(12)])
    for p in range(12):
        curves[f"chroma_{p:02d}"] = chroma_f[p]
    return curves, {"key": key, "chords": chords}
