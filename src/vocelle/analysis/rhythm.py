"""Beat, downbeat and tempo tracking.

Uses Beat This! (ISMIR 2024, `pip install vocelle[beats]`) when available — it tracks
beats *and* downbeats on the full mix. Falls back to librosa's beat tracker plus a
simple downbeat-phase heuristic (assumes 4/4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from .envelopes import anticipation, impulses, phase

log = logging.getLogger(__name__)


def _beat_this(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from beat_this.inference import File2Beats  # type: ignore
    except ImportError:
        return None
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        beats, downbeats = File2Beats(checkpoint_path="final0", device=device, dbn=False)(str(path))
        return np.asarray(beats, float), np.asarray(downbeats, float)
    except Exception as exc:  # model download / runtime problems -> fall back
        log.warning("Beat This! failed (%s); falling back to librosa", exc)
        return None


def _librosa_beats(y: np.ndarray, sr: int, beats_per_bar: int = 4) -> tuple[np.ndarray, np.ndarray]:
    hop = 512
    # Percussive part only: sustained pads/chord changes otherwise pull the tracker to bar rate.
    env = librosa.onset.onset_strength(y=librosa.effects.percussive(y, margin=2.0), sr=sr, hop_length=hop)
    tempo, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    if 0 < tempo < 70 or tempo > 190:  # octave correction into the usual pop/rock/electronic range
        target = tempo * 2 if tempo < 70 else tempo / 2
        _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=target)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(beats) < beats_per_bar:
        return beats, beats[:1]
    # Downbeat heuristic: chords tend to change, and kicks tend to land, on the "one".
    # Score each bar phase by harmonic change (beat-synchronous chroma) + low-frequency onsets.
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=32)
    lo_at = low[np.clip(frames, 0, len(low) - 1)]
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr, hop_length=hop)
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)[:, 1:len(frames) + 1]
    sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    change = np.r_[0.0, 1.0 - (sync[:, 1:] * sync[:, :-1]).sum(axis=0)][: len(frames)]
    change = np.pad(change, (0, len(frames) - len(change)))

    def _z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    score = _z(lo_at) + 1.5 * _z(change)
    scores = [score[k::beats_per_bar].mean() for k in range(beats_per_bar)]
    k = int(np.argmax(scores))
    return beats, beats[k::beats_per_bar]


def track_beats(path: Path, y: np.ndarray, sr: int, use_beat_this: bool = True) -> dict:
    res = _beat_this(path) if use_beat_this else None
    source = "beat_this"
    if res is None:
        res = _librosa_beats(y, sr)
        source = "librosa"
    beats, downbeats = res
    tempo = float(60.0 / np.median(np.diff(beats))) if len(beats) > 1 else 0.0
    if len(downbeats) > 1 and len(beats) > 1:
        bpb = int(round(np.median(np.diff(downbeats)) / np.median(np.diff(beats))))
    else:
        bpb = 4
    return {
        "source": source,
        "tempo": round(tempo, 2),
        "beats_per_bar": max(bpb, 1),
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [round(float(b), 3) for b in downbeats],
    }


def rhythm_curves(rhythm: dict, fps: float, n_frames: int) -> dict[str, np.ndarray]:
    beats, downbeats = rhythm["beats"], rhythm["downbeats"]
    beat_phase, beat_idx = phase(beats, fps, n_frames)
    bar_phase, bar_idx = phase(downbeats, fps, n_frames)
    period = 60.0 / rhythm["tempo"] if rhythm["tempo"] else 0.5
    return {
        "beat_phase": beat_phase,
        "bar_phase": bar_phase,
        "bar_index": bar_idx.astype(float),
        "beat_pulse": impulses(beats, fps, n_frames, decay=period * 0.35),
        "downbeat_pulse": impulses(downbeats, fps, n_frames, decay=period * 0.8),
        "downbeat_windup": anticipation(downbeats, fps, n_frames, lead=period),
    }
