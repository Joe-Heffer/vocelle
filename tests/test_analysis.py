import numpy as np
import pytest
import soundfile as sf

from vocelle import synth
from vocelle.analysis.harmony import PITCHES, analyze_harmony, estimate_key, fifths_index
from vocelle.analysis.melody import analyze_melody
from vocelle.analysis.pipeline import analyze
from vocelle.analysis.rhythm import track_beats

SR = synth.SR


def test_click_track_tempo(tmp_path):
    y = synth.click_track(bpm=120, seconds=15)
    r = track_beats(tmp_path / "x.wav", y, SR, use_beat_this=False)
    assert abs(r["tempo"] - 120) < 3
    assert np.allclose(np.diff(r["beats"]).mean(), 0.5, atol=0.03)


@pytest.mark.parametrize("root,quality,label", [("C", "maj", "C"), ("A", "min", "Am"), ("Eb", "maj", "Eb")])
def test_chord_recognition(root, quality, label):
    y = np.concatenate([synth.chord(root, quality, 3.0)] * 2)
    _, ev = analyze_harmony(y, SR, 30, int(6 * 30))
    main = max(ev["chords"], key=lambda c: c["end"] - c["start"])
    assert main["label"] == label


def test_key_c_major():
    y = np.concatenate([synth.chord(r, q, 1.0) for r, q in [("C", "maj"), ("F", "maj"), ("G", "maj"), ("C", "maj")]])
    import librosa

    key = estimate_key(librosa.feature.chroma_cqt(y=y, sr=SR))
    assert (key["tonic"], key["mode"]) == ("C", "major")


def test_circle_of_fifths():
    assert [PITCHES[(i * 7) % 12] for i in range(3)] == ["C", "G", "D"]
    assert fifths_index(7) == 1 and fifths_index(5) == 11


def test_melody_rising():
    y = np.concatenate([synth.tone(m, 0.5, harmonics=2) for m in (60, 64, 67, 72)])
    c, ev = analyze_melody(y, SR, 30, 60)
    p = c["pitch"]
    assert p[10] < p[25] < p[40] < p[55]
    assert [n["pitch"] for n in ev["notes"]][:4] == [60, 64, 67, 72]


def test_full_pipeline(tmp_path):
    y, truth = synth.song()
    path = tmp_path / "song.wav"
    sf.write(path, y, SR)
    f = analyze(path, fps=30, separate_stems=False, use_beat_this=False)
    assert abs(f["rhythm"]["tempo"] - truth["bpm"]) < 4
    n = f["n_frames"]
    assert all(len(v) == n for v in f["curves"].values())
    assert all(0 <= min(v) and max(v) <= 1.0001 for k, v in f["curves"].items()
               if k not in ("bar_index", "section_index", "section_label", "melody_midi"))
    labels = [c["label"] for c in f["chords"] if c["end"] - c["start"] > 1.0]
    assert {"C", "G", "Am", "F"} <= set(labels)
    assert len(f["sections"]) >= 2
    bar = 4 * 60 / truth["bpm"]
    db = np.asarray(f["rhythm"]["downbeats"])
    off = np.abs(((db + bar / 2) % bar) - bar / 2)  # distance to the true bar grid (starts at 0)
    assert np.median(off) < 0.1
