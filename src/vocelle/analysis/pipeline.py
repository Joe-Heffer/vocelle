"""Run the full analysis and write the feature file consumed by Blender scenes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

from ..features import SCHEMA_VERSION
from .envelopes import impulses, lookahead_build, smooth_ar
from .harmony import analyze_harmony
from .melody import analyze_melody
from .rhythm import rhythm_curves, track_beats
from .spectral import spectral_features
from .stems import separate
from .structure import analyze_structure

log = logging.getLogger(__name__)
SR = 22050


def _load(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def analyze(track: Path, fps: float = 30.0, stems_dir: Path | None = None, separate_stems: bool = True,
            model: str = "htdemucs", stems_root: Path = Path("audio/stems"), use_beat_this: bool = True,
            n_sections: int | None = None) -> dict:
    track = Path(track)
    y = _load(track)
    duration = len(y) / SR
    n = int(np.ceil(duration * fps))
    curves: dict[str, np.ndarray] = {}
    out: dict = {"schema_version": SCHEMA_VERSION, "track": track.name, "fps": fps,
                 "duration": round(duration, 3), "n_frames": n}

    if stems_dir is None and separate_stems:
        stems_dir = separate(track, stems_root, model)
    stems = {p.stem: _load(p) for p in sorted(Path(stems_dir).glob("*.wav"))} if stems_dir else {}
    for k, v in stems.items():  # pad/trim to mix length
        stems[k] = np.pad(v, (0, max(0, len(y) - len(v))))[: len(y)]
    out["stems"] = sorted(stems)

    log.info("spectral features")
    c, ev = spectral_features(y, SR, fps, n)
    curves.update({f"mix_{k}": v for k, v in c.items()})
    out["onsets"] = ev["onsets"]
    for name, sig in stems.items():
        c, ev = spectral_features(sig, SR, fps, n)
        curves.update({f"{name}_{k}": v for k, v in c.items()})
        out[f"{name}_onsets"] = ev["onsets"]

    log.info("rhythm")
    # Beat This! reads the mix from disk; the librosa fallback is cleaner on the drum stem.
    rhythm = track_beats(track, stems.get("drums", y), SR, use_beat_this)
    out["rhythm"] = rhythm
    curves.update(rhythm_curves(rhythm, fps, n))

    log.info("harmony")
    harm_src = sum(v for k, v in stems.items() if k not in ("drums", "vocals")) if stems else y
    c, ev = analyze_harmony(np.asarray(harm_src), SR, fps, n)
    curves.update(c)
    out.update(ev)

    log.info("melody")
    if "vocals" in stems and np.sqrt(np.mean(stems["vocals"] ** 2)) > 0.01:
        mel_src, mel_name = stems["vocals"], "vocals"
    else:
        mel_src, mel_name = librosa.effects.harmonic(stems.get("other", y)), ("other" if "other" in stems else "mix")
    mel_path = (Path(stems_dir) / f"{mel_name}.wav") if stems_dir and mel_name != "mix" else track
    c, ev = analyze_melody(mel_src, SR, fps, n, path=mel_path)
    curves.update({f"melody_{k}": v for k, v in c.items()})
    out["melody"] = {"source": mel_name, **ev}

    log.info("structure")
    c, ev = analyze_structure(y, SR, rhythm["beats"], rhythm["downbeats"], fps, n, n_sections)
    curves.update(c)
    out.update(ev)

    # Global, song-level curves.
    energy = smooth_ar(curves["mix_rms"], fps, 0.3, 1.5)
    curves["energy"] = energy
    curves["energy_build"] = lookahead_build(energy, fps, window=8.0)
    jumps = [s["start"] for a, s in zip(out["sections"], out["sections"][1:], strict=False)
             if s["energy"] - a["energy"] > 0.15]
    out["drops"] = jumps
    curves["drop_pulse"] = impulses(jumps, fps, n, decay=1.5)
    curves["section_change"] = impulses([s["start"] for s in out["sections"][1:]], fps, n, decay=0.6)

    out["curves"] = {k: np.round(np.asarray(v, float), 4).tolist() for k, v in sorted(curves.items())}
    return out


def write(features: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, separators=(",", ":")))
    return path
