"""Phase 6: FastAPI service that serves per-call reports.

Run:   uvicorn concall.api:app --reload        then open http://127.0.0.1:8000/docs

Endpoints
  GET  /health                       liveness + which optional components are configured
  GET  /calls                        all calls in the manifest with their pipeline status
  GET  /calls/{call_id}/report       everything we know about one call (JSON)
  POST /calls/{call_id}/process      run the pipeline for one call in the background
  GET  /results/regressions          event-study regression table

Reports are read from the files the pipeline writes, so the API holds no state of its own and
can be restarted at any time.
"""

from __future__ import annotations

import json
import math

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException

from . import diarize, llm, manifest, pipeline, tone
from .config import load_settings

app = FastAPI(title="Earnings Call Voice & Sentiment Analyzer", version="1.0.0")


def _clean(obj):
    """JSON can't carry NaN/inf; make them null."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _calls() -> pd.DataFrame:
    return manifest.load_manifest()


def _call_row(call_id: str) -> pd.Series:
    calls = _calls()
    if call_id not in calls.index:
        raise HTTPException(404, f"unknown call_id {call_id}")
    return calls.loc[call_id]


@app.get("/health")
def health():
    return {"status": "ok", "diarization_configured": diarize.available(), "llm_configured": llm.available()}


@app.get("/calls")
def list_calls():
    return [{"call_id": cid, "company": r["company"], "fiscal_quarter": r["fiscal_quarter"],
             "status": pipeline.status(cid)} for cid, r in _calls().iterrows()]


@app.get("/calls/{call_id}/report")
def report(call_id: str):
    row = _call_row(call_id)
    out: dict = {"call": row.to_dict(), "status": pipeline.status(call_id)}

    fp = tone.features_path()
    if fp.exists():
        f = pd.read_parquet(fp)
        f = f[f["call_id"] == call_id]
        if len(f):
            out["features"] = f.iloc[0].to_dict()

    ev = load_settings().path("results") / "event_study.csv"
    if ev.exists():
        e = pd.read_csv(ev)
        e = e[e["call_id"] == call_id]
        if len(e):
            out["market_reaction"] = e.iloc[0].to_dict()

    doc = llm.load_llm(call_id)
    if doc:
        out["llm"] = {k: doc[k] for k in ("summary", "outlook", "guidance", "evasive_answers", "kpis")}

    roles_file = load_settings().path("utterances") / f"{call_id}.roles.json"
    if roles_file.exists():
        out["speaker_roles"] = json.loads(roles_file.read_text(encoding="utf-8"))
    return _clean(out)


@app.post("/calls/{call_id}/process", status_code=202)
def process(call_id: str, background: BackgroundTasks, force: bool = False):
    _call_row(call_id)
    background.add_task(pipeline.run_call, call_id, force)
    return {"call_id": call_id, "queued": True}


@app.get("/results/regressions")
def regressions():
    p = load_settings().path("results") / "regressions.csv"
    if not p.exists():
        raise HTTPException(404, "run `python -m concall study` first")
    return _clean(pd.read_csv(p).to_dict(orient="records"))

