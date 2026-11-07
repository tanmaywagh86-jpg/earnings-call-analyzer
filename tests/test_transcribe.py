import numpy as np

from concall.transcribe import SR, chunk_bounds


def test_short_audio_is_one_chunk():
    y = np.ones(SR * 60, dtype="float32")
    assert chunk_bounds(y, 600) == [0, len(y)]


def test_cuts_land_in_silence():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, SR * 1500).astype("float32")  # 25 min of "speech"
    y[SR * 605: SR * 606] = 0.0                          # a silent second near the 10-min mark
    b = chunk_bounds(y, 600)
    assert b[0] == 0 and b[-1] == len(y)
    assert SR * 605 <= b[1] <= SR * 606
    assert all(x < z for x, z in zip(b, b[1:]))          # strictly increasing, no empty chunks
