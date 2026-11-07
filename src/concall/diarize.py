"""Phase 2a: speaker diarization ("who spoke when") with pyannote.audio.

Diarization splits the audio into turns and clusters them by voice: SPEAKER_00, SPEAKER_01, ...
The labels are anonymous; roles.py later decides which label is the moderator, management or analysts.

How pyannote works (simplified):
  1. a segmentation model finds speech regions and speaker changes in ~10 s sliding windows;
  2. an embedding model turns each speaker-chunk into a vector ("voiceprint");
  3. clustering groups vectors that sound like the same person.

Output: data/diarization/<call_id>.json  {"call_id", "model", "device", "turns": [{start, end, speaker}]}

Needs HF_TOKEN in .env and accepting the model's terms on huggingface.co (free).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .config import load_settings, secret, torch_device

log = logging.getLogger(__name__)


class DiarizationUnavailable(RuntimeError):
    """Raised when pyannote can't run (no token / model terms not accepted). Callers fall back to text heuristics."""


def diarization_path(call_id: str) -> Path:
    return load_settings().path("diarization") / f"{call_id}.json"


def available() -> bool:
    return secret("HF_TOKEN") is not None


def _load_pipeline(model: str, token: str):
    from pyannote.audio import Pipeline
    try:
        return Pipeline.from_pretrained(model, token=token)          # pyannote.audio >= 4
    except TypeError:
        return Pipeline.from_pretrained(model, use_auth_token=token)  # pyannote.audio 3.x


def _annotation(output):
    """pyannote 4 returns an object holding several annotations; 3.x returns the Annotation itself.
    'exclusive' diarization has no overlapping turns, which makes word -> speaker assignment unambiguous."""
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        ann = getattr(output, attr, None)
        if ann is not None:
            return ann
    return output


def diarize(call_id: str, audio: Path, force: bool = False) -> Path:
    out = diarization_path(call_id)
    if out.exists() and not force:
        return out
    token = secret("HF_TOKEN")
    if not token:
        raise DiarizationUnavailable("HF_TOKEN not set in .env")

    import soundfile as sf
    import torch

    cfg = load_settings().section("diarization")
    device = torch_device(cfg.get("device", "auto"))
    try:
        pipeline = _load_pipeline(cfg["model"], token)
    except Exception as e:  # noqa: BLE001 - gated model / bad token / network
        raise DiarizationUnavailable(f"could not load {cfg['model']}: {e}") from e
    if pipeline is None:  # pyannote 3 returns None when the terms haven't been accepted
        raise DiarizationUnavailable(f"accept the terms at https://huggingface.co/{cfg['model']}")
    pipeline.to(torch.device(device))

    # Pass audio in memory: avoids pyannote's own decoder (torchcodec/FFmpeg DLLs on Windows).
    y, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(y if y.ndim == 1 else y.mean(axis=1)).unsqueeze(0)

    t0 = time.perf_counter()
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sr},
                          min_speakers=cfg.get("min_speakers"), max_speakers=cfg.get("max_speakers"))
        ann = _annotation(output)
        turns = [{"start": round(t.start, 3), "end": round(t.end, 3), "speaker": spk}
                 for t, _, spk in ann.itertracks(yield_label=True)]
    finally:
        # Hand the GPU back before later stages (FinBERT) load: PyTorch's caching allocator
        # otherwise keeps pyannote's memory reserved, which caused CUDA OOM on a 6 GB card.
        del pipeline
        if device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    doc = {"call_id": call_id, "model": cfg["model"], "device": device,
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "n_speakers": len({t["speaker"] for t in turns}), "turns": turns}
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(out)
    log.info("%s: %d turns, %d speakers (%s, %.0f s)", call_id, len(turns), doc["n_speakers"], device, doc["elapsed_s"])
    return out


def load_diarization(call_id: str) -> dict | None:
    p = diarization_path(call_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
