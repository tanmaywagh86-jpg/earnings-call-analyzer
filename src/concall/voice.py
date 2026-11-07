"""Voice (acoustic) features: HOW something was said, not WHAT was said.

We compute features per transcript segment (a sentence-ish chunk, usually 2-30 s), because:
  * segments are short, so pitch tracking needs little memory (a full 1-hour call at once
    would need several GB);
  * silence and hold music outside segments are skipped automatically;
  * in Phase 2 each segment gets a speaker label, and per-speaker features become a
    simple group-by over this table.

Features per segment
--------------------
Pitch (fundamental frequency, F0, via YIN or pYIN):
  f0_median_hz    typical pitch of the voice in this segment
  f0_std_st       pitch variation in semitones around the segment median. Semitones (a log
                  scale) make variation comparable between a low and a high voice:
                  st = 12 * log2(f0 / median_f0)
  f0_range_st     90th - 10th percentile, in semitones (robust "pitch range")
  voiced_ratio    share of frames with a detectable pitch

Timing (from Whisper word timestamps):
  speaking_rate_wps      words / segment duration (pauses included)
  articulation_rate_wps  words / (duration - pause time): how fast they talk when talking
  n_pauses, total_pause_s, mean_pause_s, n_long_pauses
                         gaps between consecutive words >= pause_min_s / long_pause_s

Loudness:
  rms_db_std      variation in loudness (dB) over voiced frames

Hypotheses we will test later: stressed or evasive speakers show more pauses, slower
articulation and changed pitch variation, especially in unscripted Q&A vs prepared remarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings

_EPS = 1e-10


def segment_features_path(call_id: str) -> Path:
    d = load_settings().path("features") / "voice_segments"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{call_id}.parquet"


# --------------------------------------------------------------------------- pitch

def pitch_track(y: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0 per frame with NaN where unvoiced, rms_db per frame)."""
    import librosa

    fmin, fmax, hop = cfg["fmin_hz"], cfg["fmax_hz"], cfg["hop_length"]
    frame_length = 1024  # 64 ms at 16 kHz: long enough to hold 2+ periods of a 65 Hz voice
    if len(y) < frame_length:
        return np.array([]), np.array([])

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + _EPS)

    if cfg.get("pitch_method", "yin") == "pyin":
        f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                     frame_length=frame_length, hop_length=hop)
        f0 = np.where(voiced, f0, np.nan)
    else:
        # YIN always returns *a* pitch, even for silence, hiss or "s"/"sh" sounds. We keep a
        # frame only if it is (a) loud relative to this segment, (b) not stuck at the search
        # limits, and (c) actually periodic: the waveform must repeat itself after one
        # detected period. (c) is what rejects steady background noise, which passes (a).
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                         frame_length=frame_length, hop_length=hop)
        n = min(len(f0), len(rms_db))
        f0, rms_db = f0[:n], rms_db[:n]
        loud = rms_db > (np.percentile(rms_db, 95) - 25)
        in_range = (f0 > fmin * 1.05) & (f0 < fmax * 0.95)
        periodic = periodicity(y, sr, f0, frame_length, hop) > cfg.get("min_periodicity", 0.5)
        f0 = np.where(loud & in_range & periodic, f0, np.nan)

    n = min(len(f0), len(rms_db))
    return f0[:n], rms_db[:n]


def periodicity(y: np.ndarray, sr: int, f0: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Normalised autocorrelation of each frame at a lag of one pitch period (0..1).

    Voiced speech repeats every period, so it scores ~0.7-1.0; noise scores ~0.
    Frames are centred like librosa's (center=True), so frame i starts at i*hop - frame_length//2.
    """
    import librosa

    padded = np.pad(y, frame_length // 2)
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop)
    out = np.zeros(len(f0))
    for i in range(min(len(f0), frames.shape[1])):
        lag = int(round(sr / f0[i])) if f0[i] > 0 else 0
        if not 0 < lag < frame_length // 2:
            continue
        a, b = frames[:-lag, i], frames[lag:, i]
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
        out[i] = np.dot(a, b) / denom if denom > _EPS else 0.0
    return out


def pitch_stats(f0: np.ndarray) -> dict:
    voiced = f0[~np.isnan(f0)]
    out = {"voiced_ratio": float(len(voiced) / len(f0)) if len(f0) else 0.0,
           "f0_median_hz": np.nan, "f0_std_st": np.nan, "f0_range_st": np.nan}
    if len(voiced) >= 10:  # need a handful of frames for stable statistics
        med = float(np.median(voiced))
        st = 12 * np.log2(voiced / med)
        p10, p90 = np.percentile(st, [10, 90])
        out.update(f0_median_hz=med, f0_std_st=float(np.std(st)), f0_range_st=float(p90 - p10))
    return out


# --------------------------------------------------------------------------- timing

def timing_stats(words: list[dict], start: float, end: float, cfg: dict) -> dict:
    duration = max(end - start, _EPS)
    n = len(words)
    gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])]) if n > 1 else np.array([])
    pauses = gaps[gaps >= cfg["pause_min_s"]]
    pause_time = float(pauses.sum())
    return {
        "n_words": n,
        "speaking_rate_wps": n / duration,
        "articulation_rate_wps": n / max(duration - pause_time, _EPS),
        "n_pauses": int(len(pauses)),
        "total_pause_s": pause_time,
        "mean_pause_s": float(pauses.mean()) if len(pauses) else 0.0,
        "n_long_pauses": int((pauses >= cfg["long_pause_s"]).sum()),
    }


# --------------------------------------------------------------------------- per segment / per call

def segment_features(y_seg: np.ndarray, sr: int, words: list[dict],
                     start: float, end: float, cfg: dict) -> dict:
    f0, rms_db = pitch_track(y_seg, sr, cfg)
    feats = pitch_stats(f0)
    voiced_db = rms_db[~np.isnan(f0)] if len(f0) else np.array([])
    feats["rms_db_std"] = float(np.std(voiced_db)) if len(voiced_db) >= 10 else np.nan
    feats.update(timing_stats(words, start, end, cfg))
    return feats


def extract_call(call_id: str, audio: Path, transcript: dict, force: bool = False) -> Path:
    """Compute features for every transcript segment of one call -> Parquet."""
    import soundfile as sf

    out = segment_features_path(call_id)
    if out.exists() and not force:
        return out

    cfg = load_settings().section("voice")
    y, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    rows = []
    for seg in transcript["segments"]:
        s, e = seg["start"], seg["end"]
        y_seg = y[int(s * sr): int(e * sr)]
        row = {"call_id": call_id, "segment_id": seg["id"], "start": s, "end": e,
               "duration_s": e - s, "text": seg["text"]}
        row.update(segment_features(y_seg, sr, seg["words"], s, e, cfg))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out


def utterance_features_path(call_id: str) -> Path:
    d = load_settings().path("features") / "voice_utterances"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{call_id}.parquet"


def extract_utterances(call_id: str, audio: Path, utts: pd.DataFrame, force: bool = False) -> Path:
    """Phase 2: same features, but per speaker-pure utterance, keeping speaker/role/section.

    Phase 1 worked on Whisper segments, which can mix two speakers (an analyst finishing and the
    CFO starting). Utterances come from diarization, so each one is a single voice.
    """
    import json

    import soundfile as sf

    out = utterance_features_path(call_id)
    if out.exists() and not force:
        return out
    cfg = load_settings().section("voice")
    y, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    rows = []
    for u in utts.itertuples(index=False):
        row = {"call_id": call_id, "utt_id": u.utt_id, "speaker": u.speaker, "role": u.role,
               "section": u.section, "start": u.start, "end": u.end, "duration_s": u.end - u.start}
        row.update(segment_features(y[int(u.start * sr): int(u.end * sr)], sr,
                                    json.loads(u.words_json), u.start, u.end, cfg))
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def summarise_call(seg_df: pd.DataFrame) -> dict:
    """Collapse segment/utterance rows to one row (duration-weighted where it makes sense).

    Used for the whole call (Phase 1) and per (role, section) group (Phase 4).
    """
    d = seg_df
    w = d["duration_s"].clip(lower=0)
    total = float(w.sum()) or _EPS

    def wmean(col: str) -> float:
        m = d[col].notna()
        return float(np.average(d.loc[m, col], weights=w[m])) if m.any() and w[m].sum() > 0 else np.nan

    speech_time = total - float(d["total_pause_s"].sum())
    return {
        "n_segments": len(d),
        "speech_duration_s": total,
        "n_words": int(d["n_words"].sum()),
        "speaking_rate_wps": d["n_words"].sum() / total,
        "articulation_rate_wps": d["n_words"].sum() / max(speech_time, _EPS),
        "pauses_per_min": d["n_pauses"].sum() / (total / 60),
        "long_pauses_per_min": d["n_long_pauses"].sum() / (total / 60),
        "mean_pause_s": float(d["total_pause_s"].sum() / max(d["n_pauses"].sum(), 1)),
        "f0_median_hz": wmean("f0_median_hz"),
        "f0_std_st": wmean("f0_std_st"),
        "f0_range_st": wmean("f0_range_st"),
        "rms_db_std": wmean("rms_db_std"),
        "voiced_ratio": wmean("voiced_ratio"),
    }
