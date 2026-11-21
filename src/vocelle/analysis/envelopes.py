"""Turn raw analysis signals into perceptually useful animation curves.

Everything here works on numpy arrays sampled at the video frame rate.
"""

from __future__ import annotations

import numpy as np


def norm(x, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robust 0..1 normalisation (percentile clip, so outliers don't flatten the curve)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_frames(x, src_rate: float, fps: float, n_frames: int) -> np.ndarray:
    """Resample a signal with `src_rate` samples/second to `n_frames` frames at `fps`."""
    x = np.asarray(x, dtype=float)
    t_src = np.arange(len(x)) / src_rate
    t_dst = np.arange(n_frames) / fps
    return np.interp(t_dst, t_src, x)


def smooth_ar(x, fps: float, attack: float = 0.01, release: float = 0.25) -> np.ndarray:
    """Asymmetric one-pole follower: fast rise, slow decay (like a VU meter / compressor)."""
    x = np.asarray(x, dtype=float)
    a_up = 1.0 - np.exp(-1.0 / max(attack * fps, 1e-6))
    a_dn = 1.0 - np.exp(-1.0 / max(release * fps, 1e-6))
    y = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        acc += (v - acc) * (a_up if v > acc else a_dn)
        y[i] = acc
    return y


def impulses(times, fps: float, n_frames: int, decay: float = 0.2, weights=None) -> np.ndarray:
    """Exponentially decaying impulse at each event time (0..1). Overlaps take the max."""
    out = np.zeros(n_frames)
    t = np.arange(n_frames) / fps
    weights = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    for ev, w in zip(times, weights, strict=True):
        i0 = int(np.ceil(ev * fps - 1e-9))
        if i0 >= n_frames:
            continue
        i0 = max(i0, 0)
        i1 = min(n_frames, i0 + int(decay * fps * 6) + 1)
        seg = w * np.exp(-(t[i0:i1] - ev) / decay)
        out[i0:i1] = np.maximum(out[i0:i1], seg)
    return np.clip(out, 0.0, 1.0)


def anticipation(times, fps: float, n_frames: int, lead: float = 0.5, weights=None) -> np.ndarray:
    """Ramp 0 -> 1 over `lead` seconds *before* each event, then snap to 0.

    Offline analysis knows the future: use this for wind-ups before a hit or a drop.
    """
    out = np.zeros(n_frames)
    t = np.arange(n_frames) / fps
    weights = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    for ev, w in zip(times, weights, strict=True):
        m = (t >= ev - lead) & (t < ev)
        ramp = w * ((t[m] - (ev - lead)) / lead) ** 2  # ease-in
        out[m] = np.maximum(out[m], ramp)
    return np.clip(out, 0.0, 1.0)


def phase(times, fps: float, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Sawtooth phase (0..1) between consecutive events and the integer event index per frame.

    Before the first event / after the last, the mean interval is extrapolated so the
    phase keeps running smoothly.
    """
    t = np.arange(n_frames) / fps
    times = np.asarray(times, float)
    if len(times) < 2:
        return np.zeros(n_frames), np.zeros(n_frames, dtype=int)
    period = float(np.median(np.diff(times)))
    pre = np.arange(times[0] - period, -period * 2, -period)[::-1]
    post_end = n_frames / fps + period * 2
    post = np.arange(times[-1] + period, post_end, period)
    grid = np.concatenate([pre, times, post])
    idx = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    ph = (t - grid[idx]) / (grid[idx + 1] - grid[idx])
    event_index = idx - len(pre)
    return np.clip(ph, 0.0, 1.0), event_index


def lookahead_build(energy, fps: float, window: float = 8.0) -> np.ndarray:
    """How much louder the next `window` seconds are than the last `window` seconds.

    High just before a drop/chorus -> use it to build tension (pull back, desaturate...).
    """
    e = np.asarray(energy, float)
    w = max(1, int(window * fps))
    c = np.concatenate([[0.0], np.cumsum(e)])
    n = len(e)
    i = np.arange(n)
    fut = (c[np.minimum(i + w, n)] - c[i]) / np.maximum(np.minimum(i + w, n) - i, 1)
    past = (c[i] - c[np.maximum(i - w, 0)]) / np.maximum(i - np.maximum(i - w, 0), 1)
    return norm(np.maximum(fut - past, 0.0), 0, 99.5)
