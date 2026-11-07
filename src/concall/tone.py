"""Phase 4: one row per call with every signal, plus quarter-over-quarter (QoQ) changes.

Feature naming:  <who>_<signal>_<section>
  who      mgmt (management) | analyst
  section  prep (prepared remarks) | qa (Q&A) | all
Example: mgmt_tone_qa = word-weighted mean FinBERT tone of management sentences in Q&A.

Text signals    tone (P+ - P-), neg_share (share of sentences with P- > 0.5), hedges_per_1k
Voice signals   f0_std_st, articulation_rate_wps, pauses_per_min, long_pauses_per_min
LLM signals     guidance_score (+1 raised, 0 maintained/introduced, -1 lowered/withdrawn, averaged),
                evasive_rate (non_answer = 1, partial = 0.5, divided by # analyst questions),
                outlook_score (+1 positive, 0 neutral/mixed, -1 negative), yoy growth controls
Contrasts       <signal>_qa_minus_prep for management: how much they change when unscripted

Why QoQ changes? Each CEO/CFO has a personal baseline (some always sound upbeat, some always
hedge). Comparing a company with *itself* last quarter removes that baseline, so the change is
the informative part. d_<feature> = value this quarter - value last quarter (only if the two
quarters are consecutive; otherwise NaN).

Output: data/features/calls.parquet (+ .csv for Excel)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import llm, manifest, roles, text, voice
from .config import load_settings

SECTIONS = {"prep": "prepared", "qa": "qa"}
DIRECTION_SCORE = {"raised": 1, "maintained": 0, "introduced": 0, "lowered": -1, "withdrawn": -1}
OUTLOOK_SCORE = {"positive": 1, "neutral": 0, "mixed": 0, "negative": -1}
VOICE_SIGNALS = ["f0_std_st", "articulation_rate_wps", "pauses_per_min", "long_pauses_per_min"]
CONTRASTS = ["tone", "hedges_per_1k", "f0_std_st", "articulation_rate_wps", "pauses_per_min"]
QOQ_FEATURES = ["mgmt_tone_all", "mgmt_tone_qa", "mgmt_hedges_per_1k_all", "mgmt_f0_std_st_qa",
                "mgmt_articulation_rate_wps_qa", "mgmt_pauses_per_min_qa", "evasive_rate",
                "guidance_score", "outlook_score", "analyst_tone_qa"]


def features_path():
    return load_settings().path("features") / "calls.parquet"


def text_features(sent: pd.DataFrame) -> dict:
    out = {}
    for who, role in (("mgmt", "management"), ("analyst", "analyst")):
        for sec_key, sec in [*SECTIONS.items(), ("all", None)]:
            d = sent[(sent["role"] == role) & ((sent["section"] == sec) if sec else True)]
            words = d["n_words"].sum()
            out[f"{who}_tone_{sec_key}"] = float(np.average(d["tone"], weights=d["n_words"])) if words else np.nan
            out[f"{who}_neg_share_{sec_key}"] = float((d["p_neg"] > 0.5).mean()) if len(d) else np.nan
            out[f"{who}_hedges_per_1k_{sec_key}"] = float(d["n_hedges"].sum() / words * 1000) if words else np.nan
            out[f"{who}_words_{sec_key}"] = int(words)
    return out


def voice_features(vu: pd.DataFrame) -> dict:
    out = {}
    for sec_key, sec in [*SECTIONS.items(), ("all", None)]:
        d = vu[(vu["role"] == "management") & ((vu["section"] == sec) if sec else True)]
        summary = voice.summarise_call(d) if len(d) and d["duration_s"].sum() > 0 else {}
        for s in VOICE_SIGNALS:
            out[f"mgmt_{s}_{sec_key}"] = summary.get(s, np.nan)
    return out


def llm_features(doc: dict | None) -> dict:
    if not doc:
        return {}
    dirs = [DIRECTION_SCORE[g["direction"]] for g in doc["guidance"] if g["direction"] in DIRECTION_SCORE]
    ev = sum(1.0 if e["severity"] == "non_answer" else 0.5 for e in doc["evasive_answers"])
    return {
        "guidance_score": float(np.mean(dirs)) if dirs else 0.0,
        "n_guidance": len(doc["guidance"]),
        "n_guidance_lowered": sum(g["direction"] in ("lowered", "withdrawn") for g in doc["guidance"]),
        "n_analyst_questions": doc["n_analyst_questions"],
        "evasive_rate": ev / doc["n_analyst_questions"] if doc["n_analyst_questions"] else np.nan,
        "outlook_score": OUTLOOK_SCORE.get(doc["outlook"], 0),
        "yoy_revenue_growth_pct": doc["yoy_revenue_growth_pct"],
        "yoy_profit_growth_pct": doc["yoy_profit_growth_pct"],
    }


FUNDAMENTALS = ["yoy_revenue_growth_pct", "yoy_profit_growth_pct"]


def load_fundamentals() -> pd.DataFrame:
    """Hand-verified YoY growth per call (banks: net interest income and standalone PAT), from the
    published quarterly results. Empty cells mean "not comparable" (e.g. a loss or a merger in the base
    quarter). Returns an empty frame if the file doesn't exist."""
    try:
        p = load_settings().path("fundamentals")
    except KeyError:
        return pd.DataFrame(columns=["call_id", *FUNDAMENTALS])
    if not p.exists():
        return pd.DataFrame(columns=["call_id", *FUNDAMENTALS])
    return pd.read_csv(p, usecols=["call_id", *FUNDAMENTALS])


def add_fundamentals(rec: dict, fund: pd.DataFrame) -> dict:
    """The hand-verified table wins over the LLM's reading of the call: it comes from the published
    results, while the LLM (or a transcript) can mishear a number. A row in the table is
    authoritative even when a cell is empty (i.e. deliberately not comparable)."""
    row = fund[fund["call_id"] == rec["call_id"]]
    if len(row):
        for f in FUNDAMENTALS:
            v = row.iloc[0][f]
            rec[f] = float(v) if pd.notna(v) else np.nan
    return rec


def add_contrasts(row: dict) -> dict:
    for s in CONTRASTS:
        qa, prep = row.get(f"mgmt_{s}_qa", np.nan), row.get(f"mgmt_{s}_prep", np.nan)
        row[f"mgmt_{s}_qa_minus_prep"] = qa - prep
    return row


def add_qoq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["nse_symbol", "q_index"]).copy()
    for f in QOQ_FEATURES:
        if f not in df:
            continue
        prev = df.groupby("nse_symbol")[f].shift(1)
        consecutive = df.groupby("nse_symbol")["q_index"].diff() == 1
        df[f"d_{f}"] = np.where(consecutive, df[f] - prev, np.nan)
    return df


def build_table(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fund = load_fundamentals()
    for cid, row in calls.iterrows():
        tp, up = text.text_path(cid), voice.utterance_features_path(cid)
        if not tp.exists():
            continue  # call not processed through Phase 3 yet
        rec = {"call_id": cid, "company": row["company"], "nse_symbol": row["nse_symbol"],
               "sector": row["sector"], "fiscal_quarter": row["fiscal_quarter"],
               "q_index": manifest.quarter_index(row["fiscal_quarter"]),
               "call_date": row["call_date"], "call_time_ist": row["call_time_ist"]}
        utts = pd.read_parquet(roles.utterances_path(cid))
        rec["role_source"] = utts["role_source"].iloc[0]
        rec.update(text_features(pd.read_parquet(tp)))
        if up.exists():
            rec.update(voice_features(pd.read_parquet(up)))
        rec.update(llm_features(llm.load_llm(cid)))
        rec = add_fundamentals(rec, fund)
        rows.append(add_contrasts(rec))
    df = add_qoq(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if not df.empty:
        df.to_parquet(features_path(), index=False)
        df.to_csv(features_path().with_suffix(".csv"), index=False)
    return df
