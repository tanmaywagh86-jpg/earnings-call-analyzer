"""Phase 3b: LLM extraction with Claude. Guidance, evasive answers, sector KPIs, YoY growth.

Why an LLM here? These need *reading comprehension*, not word counting:
  * guidance: "we now expect credit growth of 16-17%, up from 14-15%" -> metric, value, direction=raised
  * evasiveness: did the answer actually address the analyst's question?
  * KPIs: "GNPA came in at 1.67%" -> {name: GNPA, value: 1.67, unit: %}

Reliability measures:
  * structured outputs: the response must validate against the Pydantic schema below, so the
    pipeline never parses free text;
  * every extracted item carries a verbatim quote, so a human can check it against the transcript;
  * the previous quarter's guidance is passed in, so "raised / maintained / lowered" is judged
    against what management actually said last time, not guessed.

Output: data/llm/<call_id>.json  (the validated Extraction plus model + token usage)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .config import load_settings, secret

log = logging.getLogger(__name__)

SECTOR_KPIS = {
    "Banks": "NIM, GNPA, NNPA, CASA ratio, credit cost, loan growth, deposit growth, PCR, RoA",
    "NBFC": "AUM growth, NIM, GNPA, NNPA, credit cost, RoA, disbursements",
    "IT": "constant-currency revenue growth, EBIT/operating margin, attrition, deal wins/TCV, headcount",
}
DEFAULT_KPIS = "revenue growth, EBITDA margin, PAT growth, volume growth, capex"


class GuidanceItem(BaseModel):
    metric: str = Field(description="What the guidance is about, e.g. 'loan growth FY26', 'EBIT margin'")
    value: str | None = Field(description="Guided value or range as stated, e.g. '16-17%'; null if qualitative")
    direction: Literal["raised", "maintained", "lowered", "introduced", "withdrawn", "unclear"] = Field(
        description="Versus the previous quarter's guidance (given below) or as management describes it")
    quote: str = Field(description="Verbatim sentence from the transcript")


class EvasiveAnswer(BaseModel):
    question: str = Field(description="One-line summary of the analyst's question")
    analyst: str | None = Field(description="Analyst name/firm if announced")
    answer_excerpt: str = Field(description="Verbatim excerpt of management's answer")
    severity: Literal["partial", "non_answer"] = Field(
        description="partial = addressed only part / deflected to generalities; non_answer = declined or ignored")
    reason: str


class KPI(BaseModel):
    name: str
    value: float | None
    unit: str = Field(description="%, bps, Rs crore, x, ...")
    period: str = Field(description="Period the value refers to, e.g. 'Q2FY26'")
    quote: str


class Extraction(BaseModel):
    summary: str = Field(description="3-4 sentence neutral summary of the call")
    outlook: Literal["positive", "neutral", "negative", "mixed"]
    guidance: list[GuidanceItem]
    n_analyst_questions: int = Field(description="Number of distinct analyst questions in Q&A")
    evasive_answers: list[EvasiveAnswer]
    kpis: list[KPI]
    yoy_revenue_growth_pct: float | None = Field(
        description="Headline YoY growth of the top line as stated (banks: net interest income; else revenue)")
    yoy_profit_growth_pct: float | None = Field(description="YoY growth in net profit (PAT) as stated")
    revenue_metric: str | None = Field(description="Which top-line metric the revenue growth refers to")


SYSTEM = """You are a sell-side equity research assistant analysing the transcript of an Indian listed \
company's quarterly earnings call. Extract information exactly as stated. Never infer numbers that \
are not said. Every quote must be copied verbatim from the transcript. Lines are tagged \
[section | role | speaker]; role labels come from an automatic classifier and can occasionally be wrong."""


def llm_path(call_id: str) -> Path:
    return load_settings().path("llm") / f"{call_id}.json"


def available() -> bool:
    return secret("ANTHROPIC_API_KEY") is not None or secret("ANTHROPIC_AUTH_TOKEN") is not None


def format_transcript(utts: pd.DataFrame) -> str:
    lines = [f"[{u.section} | {u.role} | {u.speaker or '?'}] {u.text}" for u in utts.itertuples(index=False)]
    return "\n".join(lines)


def build_prompt(row: pd.Series, utts: pd.DataFrame, previous: dict | None) -> str:
    kpis = SECTOR_KPIS.get(row["sector"], DEFAULT_KPIS)
    prev = "None available (first quarter in our sample)."
    if previous:
        prev = json.dumps(previous.get("guidance", []), ensure_ascii=False, indent=1)
    return f"""Company: {row['company']} (NSE: {row['nse_symbol']}), sector: {row['sector']}
Quarter: {row['fiscal_quarter']}  Call date: {row['call_date']}

Sector KPIs to look for: {kpis}

Guidance given in the PREVIOUS quarter's call (compare against this for 'direction'):
{prev}

Instructions:
- guidance: forward-looking targets or outlook statements by management only.
- evasive_answers: only answers that genuinely fail to address the question; an empty list is fine.
- kpis: values reported for this quarter.
- yoy growth: as stated by management or in the prepared remarks; null if not stated.

<transcript>
{format_transcript(utts)}
</transcript>"""


def extract(call_id: str, row: pd.Series, utts: pd.DataFrame, previous: dict | None, force: bool = False) -> Path:
    out = llm_path(call_id)
    if out.exists() and not force:
        return out
    import anthropic

    cfg = load_settings().section("llm")
    client = anthropic.Anthropic()
    try:
        # beta.messages.parse = structured output (validated against Extraction) + server-side
        # fallback: if a safety classifier wrongly declines a transcript, the API re-runs it on a
        # recommended fallback model instead of failing.
        response = client.beta.messages.parse(
            model=cfg["model"],
            max_tokens=cfg.get("max_tokens", 16000),
            system=SYSTEM,
            output_config={"effort": cfg.get("effort", "medium")},
            messages=[{"role": "user", "content": build_prompt(row, utts, previous)}],
            output_format=Extraction,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError("ANTHROPIC_API_KEY is invalid") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"rate limited; retry later ({e.message})") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("network error reaching the Claude API") from e

    if response.stop_reason == "refusal":
        raise RuntimeError(f"model declined the request: {response.stop_details}")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise RuntimeError(f"no valid structured output (stop_reason={response.stop_reason})")

    doc = {"call_id": call_id, "model": response.model,
           "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
           **response.parsed_output.model_dump()}
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)
    log.info("%s: %d guidance items, %d evasive answers, %d KPIs (%d in / %d out tokens)", call_id,
             len(doc["guidance"]), len(doc["evasive_answers"]), len(doc["kpis"]),
             doc["usage"]["input_tokens"], doc["usage"]["output_tokens"])
    return out


def load_llm(call_id: str) -> dict | None:
    p = llm_path(call_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
