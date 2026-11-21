"""Loudness / spectral features for one signal (mix or stem)."""

from __future__ import annotations

import librosa
import numpy as np

from .envelopes import impulses, norm, smooth_ar, to_frames

BANDS = {"sub": (20, 80), "low": (80, 250), "mid": (250, 2000), "high": (2000, 12000)}
HOP = 512


def spectral_features(y: np.ndarray, sr: int, fps: float, n_frames: int) -> tuple[dict, dict]:
    """Return (curves, events) for a mono signal. Curves are 0..1 at `fps`."""
    rate = sr / HOP
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    curves: dict[str, np.ndarray] = {}

    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    curves["rms"] = smooth_ar(norm(to_frames(rms_db, rate, fps, n_frames), 5, 99.5), fps, 0.005, 0.12)

    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        if not m.any():
            curves[name] = np.zeros(n_frames)
            continue
        band_db = librosa.amplitude_to_db(S[m].mean(axis=0) + 1e-10, ref=np.max)
        curves[name] = smooth_ar(norm(to_frames(band_db, rate, fps, n_frames), 5, 99.5), fps, 0.005, 0.15)

    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr)
    curves["flux"] = norm(to_frames(onset_env, rate, fps, n_frames))

    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    curves["bright"] = smooth_ar(norm(to_frames(np.log2(cent + 1), rate, fps, n_frames)), fps, 0.05, 0.3)

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time")
    strength = np.interp(onsets * rate, np.arange(len(onset_env)), onset_env)
    strength = norm(strength, 0, 100) * 0.7 + 0.3 if len(strength) > 1 else np.ones(len(onsets))
    curves["onset"] = impulses(onsets, fps, n_frames, decay=0.12, weights=strength)
    events = {"onsets": [round(float(t), 3) for t in onsets]}
    return curves, events
