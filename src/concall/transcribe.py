"""Speech -> text with Whisper (via faster-whisper), keeping per-word timestamps.

Word timestamps are the key design choice here. Later stages need them:
  * voice.py measures speech rate and pauses from the gaps between words;
  * Phase 2 assigns each word to a speaker by overlapping it with diarization turns.

Output: data/transcripts/<call_id>.json
{
  "call_id": ..., "model": "small", "device": "cpu", "language": "en",
  "duration_s": 3712.4, "elapsed_s": 905.1,
  "segments": [
    {"id": 0, "start": 12.3, "end": 17.9, "text": "...", "avg_logprob": -0.21,
     "no_speech_prob": 0.01,
     "words": [{"start": 12.3, "end": 12.6, "word": "Good", "prob": 0.98}, ...]},
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from .config import load_settings

log = logging.getLogger(__name__)


def transcript_path(call_id: str) -> Path:
    return load_settings().path("transcripts") / f"{call_id}.json"


def _pick_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). GPU if asked/available, else CPU int8."""
    cfg = load_settings().section("transcription")
    if requested in ("auto", "cuda"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:  # noqa: BLE001 - any CUDA problem means "use CPU"
            pass
        if requested == "cuda":
            log.warning("CUDA requested but not available; falling back to CPU")
    return "cpu", cfg.get("compute_type", "int8")


def _load_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _load_samples(audio: Path):
    """Our FLAC is already 16 kHz mono, so we hand Whisper the raw samples.

    This bypasses faster-whisper's own decoder (PyAV); faster-whisper 1.2.1 passes an argument
    that PyAV >= 15 removed (`metadata_errors`), and we don't need its resampling anyway.
    """
    import soundfile as sf

    y, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError(f"{audio} is {sr} Hz; run `prepare` to normalise to 16 kHz")
    return y if y.ndim == 1 else y.mean(axis=1)


SR = 16000


def chunk_bounds(y, chunk_s: float, search_s: float = 15.0, win_s: float = 0.5) -> list[int]:
    """Sample indices where to cut the audio: every ~chunk_s, moved to the quietest point nearby.

    faster-whisper builds the spectrogram of the WHOLE input up front (~900 MB for a 50-min call),
    which ran this 16 GB laptop out of RAM. Transcribing ~10-minute pieces caps that at ~180 MB.
    Cutting in the quietest half-second within +-search_s of the target avoids splitting a word.
    """
    import numpy as np

    n, step, win = len(y), int(chunk_s * SR), int(win_s * SR)
    bounds = [0]
    while n - bounds[-1] > step * 1.25:  # don't leave a tiny tail chunk
        target = bounds[-1] + step
        lo = max(bounds[-1] + win, target - int(search_s * SR))
        hi = min(n - win, target + int(search_s * SR))
        region = y[lo:hi + win].astype("float64") ** 2
        energy = np.convolve(region, np.ones(win), mode="valid")  # sliding-window energy
        bounds.append(lo + int(np.argmin(energy)) + win // 2)
    bounds.append(n)
    return bounds


def _seg_dict(seg, offset: float, seg_id: int) -> dict:
    return {
        "id": seg_id,
        "start": round(seg.start + offset, 3),
        "end": round(seg.end + offset, 3),
        "text": seg.text.strip(),
        "avg_logprob": round(seg.avg_logprob, 4),
        "no_speech_prob": round(seg.no_speech_prob, 4),
        "words": [
            {"start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
             "word": w.word.strip(), "prob": round(w.probability, 4)}
            for w in (seg.words or [])
        ],
    }


def _run(model, audio: Path, cfg: dict, call_id: str) -> tuple[dict, list[dict]]:
    y = _load_samples(audio)
    bounds = chunk_bounds(y, cfg.get("chunk_s", 600))
    segments: list[dict] = []
    language = None
    with tqdm(total=round(len(y) / SR), unit="s", desc=f"transcribe {call_id}") as bar:
        for a, b in zip(bounds, bounds[1:]):
            offset = a / SR
            segments_iter, info = model.transcribe(
                y[a:b],
                language=cfg.get("language"),
                beam_size=cfg.get("beam_size", 5),
                vad_filter=cfg.get("vad_filter", True),
                word_timestamps=True,
                condition_on_previous_text=False,  # stops Whisper repeating itself on long calls
                # Domain vocabulary. Whisper treats it as "previous text", which biases spelling
                # toward these terms; it fixed "CASA ratio" being heard as "cost ratio".
                initial_prompt=cfg.get("initial_prompt") or None,
            )
            language = language or info.language
            # transcribe() is lazy: the real work happens while we iterate.
            for seg in segments_iter:
                segments.append(_seg_dict(seg, offset, len(segments)))
                bar.update(max(0, round(seg.end + offset) - bar.n))
            bar.update(max(0, round(b / SR) - bar.n))
    meta = {"language": language, "duration_s": round(len(y) / SR, 2), "n_chunks": len(bounds) - 1}
    return meta, segments


def transcribe(call_id: str, audio: Path, force: bool = False) -> Path:
    out = transcript_path(call_id)
    if out.exists() and not force:
        return out

    cfg = load_settings().section("transcription")
    device, compute_type = _pick_device(cfg.get("device", "auto"))
    t0 = time.perf_counter()
    try:
        model = _load_model(cfg["model"], device, compute_type)
        meta, segments = _run(model, audio, cfg, call_id)
    except RuntimeError as e:
        # Typical on Windows: GPU visible but cuBLAS/cuDNN DLLs missing. Retry on CPU.
        if device != "cuda":
            raise
        log.warning("GPU transcription failed (%s); retrying on CPU", e)
        model = None  # release the GPU attempt before allocating the CPU model (memory is tight)
        import gc
        gc.collect()
        device, compute_type = "cpu", cfg.get("compute_type", "int8")
        model = _load_model(cfg["model"], device, compute_type)
        meta, segments = _run(model, audio, cfg, call_id)

    doc = {
        "call_id": call_id,
        "model": cfg["model"],
        "device": device,
        "compute_type": compute_type,
        **meta,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "segments": segments,
    }
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)  # atomic: a crash never leaves a half-written transcript behind
    return out


def load_transcript(call_id: str) -> dict:
    return json.loads(transcript_path(call_id).read_text(encoding="utf-8"))
