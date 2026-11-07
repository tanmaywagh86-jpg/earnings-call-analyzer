"""Check voice features against signals whose true answer we know."""

import numpy as np
import pandas as pd
import pytest

from concall.voice import pitch_stats, pitch_track, segment_features, summarise_call, timing_stats

SR = 16000
CFG = {"fmin_hz": 65, "fmax_hz": 400, "hop_length": 320, "pitch_method": "yin",
       "pause_min_s": 0.25, "long_pause_s": 1.0}


def harmonic_tone(f0_hz, seconds=2.0):
    """Voice-like tone: fundamental + harmonics. f0_hz can be a number or a per-sample array."""
    t = np.arange(int(SR * seconds)) / SR
    f = np.broadcast_to(np.asarray(f0_hz, dtype=float), t.shape)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return sum(np.sin(k * phase) / k for k in range(1, 5)).astype(np.float32) * 0.3


@pytest.mark.parametrize("method", ["yin", "pyin"])
def test_steady_pitch_is_recovered(method):
    f0, _ = pitch_track(harmonic_tone(150.0), SR, {**CFG, "pitch_method": method})
    stats = pitch_stats(f0)
    assert stats["f0_median_hz"] == pytest.approx(150, rel=0.03)
    assert stats["f0_std_st"] < 0.3          # steady tone -> almost no variation
    assert stats["voiced_ratio"] > 0.8


def test_glide_has_more_variation_than_steady_tone():
    t = np.arange(2 * SR) / SR
    glide = 120 * 2 ** (t / 2)               # 120 Hz -> 240 Hz = exactly 12 semitones
    stats = pitch_stats(pitch_track(harmonic_tone(glide), SR, CFG)[0])
    assert 8 < stats["f0_range_st"] < 12.5   # p90 - p10 of a linear 12-semitone ramp ~ 9.6
    assert stats["f0_std_st"] > 2


@pytest.mark.parametrize("level", [1e-6, 0.1])   # near-silence and loud hiss
def test_noise_has_no_pitch(level):
    rng = np.random.default_rng(0)
    f0, _ = pitch_track((level * rng.standard_normal(SR)).astype(np.float32), SR, CFG)
    assert pitch_stats(f0)["voiced_ratio"] < 0.1


def _words(spans):
    return [{"start": s, "end": e, "word": "w", "prob": 1.0} for s, e in spans]


def test_timing_pauses_and_rates():
    # 4 words in a 4 s segment; gaps of 0.1 s (not a pause), 0.5 s and 1.5 s (long)
    words = _words([(0.0, 0.5), (0.6, 1.0), (1.5, 1.9), (3.4, 4.0)])
    t = timing_stats(words, 0.0, 4.0, CFG)
    assert t["n_words"] == 4
    assert t["n_pauses"] == 2
    assert t["n_long_pauses"] == 1
    assert t["total_pause_s"] == pytest.approx(2.0)
    assert t["speaking_rate_wps"] == pytest.approx(1.0)
    assert t["articulation_rate_wps"] == pytest.approx(2.0)   # 4 words in 2 s of non-pause time


def test_segment_and_call_summary():
    y = harmonic_tone(200.0, seconds=3)
    row = {"duration_s": 3.0, **segment_features(y, SR, _words([(0, 1), (1.5, 3)]), 0, 3, CFG)}
    summary = summarise_call(pd.DataFrame([row, row]))
    assert summary["n_segments"] == 2
    assert summary["n_words"] == 4
    assert summary["f0_median_hz"] == pytest.approx(200, rel=0.03)
    assert summary["pauses_per_min"] == pytest.approx(2 / 6 * 60)
