import numpy as np

from vocelle.analysis import envelopes as env


def test_norm_range():
    x = env.norm(np.r_[np.linspace(0, 10, 100), 1e6])
    assert x.min() == 0 and x.max() == 1


def test_impulses_peak_and_decay():
    y = env.impulses([1.0], fps=30, n_frames=90, decay=0.2)
    assert y[30] == 1.0 and y[29] == 0.0
    assert 0 < y[36] < 1 and y[36] > y[40]


def test_anticipation_rises_before_event():
    y = env.anticipation([2.0], fps=30, n_frames=90, lead=0.5)
    assert y[44] < y[50] < y[59] and y[60] == 0.0 and y[40] == 0.0


def test_phase_sawtooth():
    ph, idx = env.phase([0.0, 0.5, 1.0, 1.5], fps=30, n_frames=60)
    assert ph[0] == 0.0 and abs(ph[7] - 7 / 15) < 1e-6
    assert idx[16] == 1 and idx[46] == 3  # extrapolates past the last event


def test_smooth_ar_asymmetric():
    x = np.r_[np.zeros(10), np.ones(10), np.zeros(30)]
    y = env.smooth_ar(x, fps=30, attack=0.01, release=0.5)
    assert y[12] > 0.9 and y[25] > 0.5  # fast attack, slow release


def test_lookahead_build_peaks_before_jump():
    e = np.r_[np.full(300, 0.1), np.full(300, 1.0)]
    b = env.lookahead_build(e, fps=30, window=4)
    assert np.argmax(b) in range(280, 310)
