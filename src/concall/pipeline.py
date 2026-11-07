"""Orchestration: the per-call stage order, status, and the cross-call study.

Per-call stages (each skips if its output exists, unless force=True):
  prepare     audio.prepare            -> data/audio/clean/<id>.flac
  transcribe  transcribe.transcribe    -> data/transcripts/<id>.json
  voice       voice.extract_call       -> data/features/voice_segments/<id>.parquet   (Phase 1, whole call)
  diarize     diarize.diarize          -> data/diarization/<id>.json                  (optional: HF_TOKEN)
  roles       roles.build              -> data/utterances/<id>.parquet
  voice_utt   voice.extract_utterances -> data/features/voice_utterances/<id>.parquet
  text        text.score_call          -> data/text/<id>.parquet
  llm         llm.extract              -> data/llm/<id>.json                          (optional: ANTHROPIC_API_KEY)

Cross-call study:
  tone.build_table -> event.run_event_study -> event.run_regressions
"""

from __future__ import annotations

import logging

import pandas as pd

from . import audio, diarize, event, llm, manifest, roles, text, tone, transcribe, voice

log = logging.getLogger(__name__)

STAGES = ["prepare", "transcribe", "voice", "diarize", "roles", "voice_utt", "text", "llm"]

OUTPUTS = {
    "prepare": audio.clean_audio_path,
    "transcribe": transcribe.transcript_path,
    "voice": voice.segment_features_path,
    "diarize": diarize.diarization_path,
    "roles": roles.utterances_path,
    "voice_utt": voice.utterance_features_path,
    "text": text.text_path,
    "llm": llm.llm_path,
}


def status(call_id: str) -> dict[str, bool]:
    return {s: OUTPUTS[s](call_id).exists() for s in STAGES}


def _previous_llm(calls: pd.DataFrame, row: pd.Series) -> dict | None:
    """LLM output of the same company's previous quarter, for judging guidance direction."""
    q = manifest.quarter_index(row["fiscal_quarter"])
    prev = calls[(calls["nse_symbol"] == row["nse_symbol"])
                 & (calls["fiscal_quarter"].map(manifest.quarter_index) == q - 1)]
    return llm.load_llm(prev.index[0]) if len(prev) else None


def run_stage(stage: str, call_id: str, row: pd.Series, calls: pd.DataFrame, force: bool = False) -> None:
    clean = audio.clean_audio_path(call_id)
    if stage == "prepare":
        audio.prepare(call_id, row["audio_source"], force=force)
    elif stage == "transcribe":
        transcribe.transcribe(call_id, clean, force=force)
    elif stage == "voice":
        voice.extract_call(call_id, clean, transcribe.load_transcript(call_id), force=force)
    elif stage == "diarize":
        if not diarize.available():
            log.warning("%s: skipping diarization (HF_TOKEN not set); roles will use text rules", call_id)
            return
        diarize.diarize(call_id, clean, force=force)
    elif stage == "roles":
        roles.build(call_id, transcribe.load_transcript(call_id), diarize.load_diarization(call_id), force=force)
    elif stage == "voice_utt":
        voice.extract_utterances(call_id, clean, pd.read_parquet(roles.utterances_path(call_id)), force=force)
    elif stage == "text":
        text.score_call(call_id, pd.read_parquet(roles.utterances_path(call_id)), force=force)
    elif stage == "llm":
        if not llm.available():
            log.warning("%s: skipping LLM extraction (ANTHROPIC_API_KEY not set)", call_id)
            return
        llm.extract(call_id, row, pd.read_parquet(roles.utterances_path(call_id)),
                    _previous_llm(calls, row), force=force)
    else:
        raise ValueError(f"unknown stage {stage}")


def _mtime(stage: str, call_id: str) -> float | None:
    p = OUTPUTS[stage](call_id)
    return p.stat().st_mtime if p.exists() else None


def is_stale(stage: str, call_id: str) -> bool:
    """True if this stage's output is older than the newest output of any earlier stage
    (like `make`: e.g. roles were redone, so text features built from the old roles are stale)."""
    mine = _mtime(stage, call_id)
    if mine is None:
        return False
    upstream = [_mtime(s, call_id) for s in STAGES[:STAGES.index(stage)]]
    return any(t is not None and t > mine for t in upstream)


def run_call(call_id: str, force: bool = False, stages: list[str] | None = None) -> None:
    calls = manifest.load_manifest()
    row = calls.loc[call_id]
    for stage in stages or STAGES:
        stale = is_stale(stage, call_id)
        if stale and not force:
            log.info("%s: %s output is older than its inputs; rebuilding", call_id, stage)
        run_stage(stage, call_id, row, calls, force=force or stale)
    log.info("done: %s", call_id)


def ordered(calls: pd.DataFrame) -> pd.DataFrame:
    """Company by company, oldest quarter first, so 'previous quarter' outputs exist when needed."""
    key = calls["fiscal_quarter"].map(manifest.quarter_index)
    return calls.assign(_q=key).sort_values(["nse_symbol", "_q"]).drop(columns="_q")


def run_study(calls: pd.DataFrame) -> dict:
    features = tone.build_table(calls)
    if features.empty:
        raise RuntimeError("no calls have finished the text stage yet")
    events = event.run_event_study(calls.loc[features["call_id"]])
    regs = event.run_regressions(features, events) if not events.empty else pd.DataFrame()
    return {"n_calls": len(features), "n_events": len(events), "n_regression_rows": len(regs)}
