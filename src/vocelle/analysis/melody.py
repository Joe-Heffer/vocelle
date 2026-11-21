"""Melody: continuous f0 contour (pYIN) and discrete note events.

Run this on the vocal stem when available (Demucs), otherwise on the mix's harmonic
part. With `vocelle[notes]` installed, Spotify's Basic Pitch provides polyphonic note
events (start, end, pitch, velocity) as well.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from .envelopes import impulses, smooth_ar, to_frames

log = logging.getLogger(__name__)
HOP = 256


def _segment_notes(midi: np.ndarray, voiced: np.ndarray, rate: float, min_dur: float = 0.08) -> list[dict]:
    notes, start, cur = [], None, None
    rounded = np.where(voiced, np.round(midi), np.nan)
    for i, m in enumerate(np.r_[rounded, np.nan]):
        if start is not None and (np.isnan(m) or m != cur):
            if (i - start) / rate >= min_dur:
                notes.append({"start": round(start / rate, 3), "end": round(i / rate, 3), "pitch": int(cur), "velocity": 0.8})
            start = None
        if start is None and not np.isnan(m):
            start, cur = i, m
    return notes


def _basic_pitch_notes(path: Path) -> list[dict] | None:
    try:
        from basic_pitch.inference import predict  # type: ignore
    except ImportError:
        return None
    try:
        out = predict(str(path))
        events = out[-1]
        return [{"start": round(float(s), 3), "end": round(float(e), 3), "pitch": int(p), "velocity": round(float(v), 3)}
                for s, e, p, v, *_ in events]
    except Exception as exc:
        log.warning("basic-pitch failed (%s); using pYIN note segmentation", exc)
        return None


def analyze_melody(y: np.ndarray, sr: int, fps: float, n_frames: int, path: Path | None = None) -> tuple[dict, dict]:
    rate = sr / HOP
    f0, voiced, vprob = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                                     sr=sr, hop_length=HOP, frame_length=2048)
    midi = librosa.hz_to_midi(np.where(np.isfinite(f0), f0, np.nan))
    if np.isfinite(midi).any():
        lo, hi = np.nanpercentile(midi, 2), np.nanpercentile(midi, 98)
        # hold last pitch through unvoiced gaps so the contour doesn't drop to zero
        filled = midi.copy()
        last = np.nanmedian(midi)
        for i, v in enumerate(filled):
            last = v if np.isfinite(v) else last
            filled[i] = last
        pitch = np.clip((filled - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        lo = hi = 0.0
        pitch = np.zeros_like(midi)
        filled = np.zeros_like(midi)
    notes = (_basic_pitch_notes(path) if path is not None else None) or _segment_notes(midi, voiced, rate)
    curves = {
        "pitch": smooth_ar(to_frames(pitch, rate, fps, n_frames), fps, 0.03, 0.03),
        "voiced": smooth_ar(to_frames(vprob * voiced, rate, fps, n_frames), fps, 0.02, 0.2),
        "note_on": impulses([n["start"] for n in notes], fps, n_frames, decay=0.15),
        "midi": to_frames(filled, rate, fps, n_frames),  # raw MIDI number (not normalised)
    }
    return curves, {"notes": notes, "range_midi": [round(float(lo), 2), round(float(hi), 2)]}
