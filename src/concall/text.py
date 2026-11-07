"""Phase 3a: sentence-level text signals. FinBERT sentiment + hedging-word counts.

FinBERT (ProsusAI/finbert) is BERT further trained on financial news (Financial PhraseBank).
General sentiment models misread finance: "liability", "provision" or "risk" are neutral there,
and "beat estimates" is positive. FinBERT outputs P(positive), P(negative), P(neutral) per sentence.

  tone = P(positive) - P(negative)          in [-1, +1]

Hedging = how often a speaker qualifies statements ("we believe", "might", "too early to say").
Counted with the lexicon in config/lexicons/hedging.txt; reported per 1,000 words in Phase 4.

Output: data/text/<call_id>.parquet, one row per sentence of every non-moderator utterance:
  call_id, utt_id, sent_id, speaker, role, section, text, n_words, p_pos, p_neg, p_neu, tone, n_hedges
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, load_settings, torch_device

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def text_path(call_id: str) -> Path:
    return load_settings().path("text") / f"{call_id}.parquet"


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Simple rule-based splitter; Whisper's punctuation is good enough for this."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= min_words]


@lru_cache(maxsize=1)
def hedge_regex() -> re.Pattern:
    cfg = load_settings().section("text")
    terms = []
    for line in (PROJECT_ROOT / cfg["hedging_lexicon"]).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            terms.append(r"\s+".join(map(re.escape, line.split())))
    terms.sort(key=len, reverse=True)  # longest first so "we will see" wins over "see"
    return re.compile(r"(?<![\w'])(?:" + "|".join(terms) + r")(?![\w'])", re.I)


def count_hedges(text: str) -> int:
    return len(hedge_regex().findall(text))


@lru_cache(maxsize=1)
def _finbert():
    from transformers import pipeline

    cfg = load_settings().section("text")
    device = torch_device(cfg.get("device", "auto"))
    return pipeline("text-classification", model=cfg["finbert_model"], top_k=None,
                    truncation=True, max_length=256, device=0 if device == "cuda" else -1)


def finbert_scores(sentences: list[str]) -> pd.DataFrame:
    if not sentences:
        return pd.DataFrame(columns=["p_pos", "p_neg", "p_neu"])
    cfg = load_settings().section("text")
    results = _finbert()(sentences, batch_size=cfg.get("batch_size", 16))
    rows = [{d["label"].lower(): d["score"] for d in r} for r in results]
    df = pd.DataFrame(rows).rename(columns={"positive": "p_pos", "negative": "p_neg", "neutral": "p_neu"})
    return df[["p_pos", "p_neg", "p_neu"]]


def score_call(call_id: str, utterances: pd.DataFrame, force: bool = False) -> Path:
    out = text_path(call_id)
    if out.exists() and not force:
        return out
    rows = []
    for u in utterances[utterances["role"] != "moderator"].itertuples(index=False):
        for k, sent in enumerate(split_sentences(u.text)):
            rows.append({"call_id": call_id, "utt_id": u.utt_id, "sent_id": k, "speaker": u.speaker,
                         "role": u.role, "section": u.section, "text": sent,
                         "n_words": len(sent.split()), "n_hedges": count_hedges(sent)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{call_id}: no non-moderator sentences to score")
    df = pd.concat([df, finbert_scores(df["text"].tolist())], axis=1)
    df["tone"] = df["p_pos"] - df["p_neg"]
    df.to_parquet(out, index=False)
    return out
