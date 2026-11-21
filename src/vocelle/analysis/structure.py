"""Song structure: section boundaries + repetition labels (A, B, A, C ...).

Chroma + MFCC + loudness on a 0.5 s grid -> self-similarity matrix -> Foote
checkerboard novelty peaks = boundaries, snapped to downbeats, then segments clustered so repeated sections share a label.
Repetition labels let scenes reuse a "visual leitmotif" every time a chorus returns.
"""

from __future__ import annotations

import string

import librosa
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .envelopes import norm, phase

HOP = 512


def _novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    n = len(ssm)
    g = np.outer(np.hanning(2 * half), np.hanning(2 * half))
    sign = np.ones((2 * half, 2 * half))
    sign[:half, half:] = sign[half:, :half] = -1
    kern = g * sign
    pad = np.pad(ssm, half, mode="edge")
    nov = np.array([(pad[i:i + 2 * half, i:i + 2 * half] * kern).sum() for i in range(n)])
    return np.maximum(nov, 0)


def _pick_peaks(nov: np.ndarray, k: int, min_gap: int) -> list[int]:
    """Greedy: take the k highest novelty peaks at least `min_gap` apart (and away from the ends)."""
    order = np.argsort(nov)[::-1]
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= k or nov[i] <= 0:
            break
        if i < min_gap // 2 or i > len(nov) - min_gap // 2:
            continue
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def analyze_structure(y: np.ndarray, sr: int, beats: list[float], downbeats: list[float],
                      fps: float, n_frames: int, n_sections: int | None = None) -> tuple[dict, dict]:
    duration = len(y) / sr
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    feats = np.vstack([librosa.util.normalize(chroma, axis=0), mfcc / (np.abs(mfcc).max() + 1e-9), norm(rms[0])[None]])
    # Uniform ~0.5 s grid (not beat-synced: beats can be missing in drum-less intros).
    step = max(1, int(round(0.5 * sr / HOP)))
    grid = np.arange(0, feats.shape[1], step)
    feats = librosa.util.sync(feats, grid, aggregate=np.mean)
    times = librosa.frames_to_time(grid, sr=sr, hop_length=HOP)[: feats.shape[1]]
    seconds = lambda s_: max(2, int(round(s_ / 0.5)))  # noqa: E731
    # Foote novelty: slide a checkerboard kernel along the diagonal of the self-similarity matrix.
    z = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-9)
    z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-9
    ssm = z.T @ z
    nov = _novelty(ssm, half=seconds(min(8.0, duration / 6)))
    min_gap = seconds(6.0)
    k = n_sections or int(np.clip(round(duration / 20.0), 2, 12))
    peaks = _pick_peaks(nov, k - 1, min_gap)
    starts = sorted({0.0, *[float(times[p]) for p in peaks]})
    if len(downbeats) > 2:  # snap to bar lines so camera cuts land on the "one"
        db = np.asarray(downbeats)
        tol = 0.6 * float(np.median(np.diff(db)))  # only snap when a bar line is close
        snapped = []
        for s_ in starts[1:]:
            near = float(db[np.argmin(np.abs(db - s_))])
            snapped.append(near if abs(near - s_) < tol else s_)
        starts = sorted({0.0, *snapped})
    ends = starts[1:] + [duration]

    # Label repeated sections: cluster segment-mean features.
    seg_feats = []
    for s, e in zip(starts, ends, strict=True):
        m = (times >= s) & (times < e)
        seg_feats.append(z[:, m].mean(axis=1) if m.any() else z[:, 0])
    seg_feats = np.array(seg_feats)
    if len(seg_feats) > 1:
        ids = fcluster(linkage(seg_feats, "average", metric="cosine"), t=0.5, criterion="distance")
    else:
        ids = np.array([1])
    label_map: dict[int, str] = {}
    labels = [label_map.setdefault(int(i), string.ascii_uppercase[len(label_map) % 26]) for i in ids]

    energy = librosa.feature.rms(y=y, hop_length=HOP)[0]
    et = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=HOP)
    sections = []
    for i, (s, e, lab) in enumerate(zip(starts, ends, labels, strict=True)):
        m = (et >= s) & (et < e)
        sections.append({"index": i, "start": round(s, 3), "end": round(e, 3), "label": lab,
                         "energy": float(energy[m].mean()) if m.any() else 0.0})
    emax = max((s["energy"] for s in sections), default=1.0) or 1.0
    for s in sections:
        s["energy"] = round(s["energy"] / emax, 3)

    sec_phase, sec_idx = phase(starts + [duration], fps, n_frames)
    t = np.arange(n_frames) / fps
    sec_energy = np.zeros(n_frames)
    sec_label = np.zeros(n_frames)
    for s in sections:
        m = (t >= s["start"]) & (t < s["end"])
        sec_energy[m] = s["energy"]
        sec_label[m] = string.ascii_uppercase.index(s["label"])
    curves = {"section_phase": sec_phase, "section_index": np.clip(sec_idx, 0, len(sections) - 1).astype(float),
              "section_energy": sec_energy, "section_label": sec_label}
    return curves, {"sections": sections}
