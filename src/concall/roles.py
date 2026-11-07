"""Phase 2b: turn words + speaker turns into labelled utterances.

Steps
-----
1. assign_speakers   each Whisper word gets the diarization speaker it overlaps most.
2. build_utterances  consecutive words of the same speaker form an utterance (one speaking turn).
3. find_qa_start     the moderator's "first question is from ..." marks where Q&A begins;
                     everything before is the prepared (scripted) remarks section.
4. label_roles       decide which anonymous speaker is the moderator, management or an analyst:
     moderator   most utterances containing operator phrases ("next question is from", "press star and one")
     analyst     speaks right after the moderator announces a questioner, or uses analyst phrases
                 ("thanks for taking my question"); little or no time in prepared remarks
     management  speaks in the prepared remarks, or answers right after an analyst
   This is a rule-based classifier: transparent and easy to audit. Every decision keeps its vote
   counts in the output so mistakes can be traced.

Without diarization (no HF_TOKEN) we fall back to the same rules applied per Whisper segment, using
only text: role_source = "text" instead of "diarization". Less accurate in Q&A, still useful.

Output: data/utterances/<call_id>.parquet
  call_id, utt_id, speaker, role, section (prepared|qa), start, end, n_words, text, words_json, role_source
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings

MODERATOR_RE = re.compile(
    r"ladies and gentlemen|hand (?:the )?(?:conference|call|floor|proceedings) over|"
    r"(?:next|first|last) question (?:is |comes )?(?:from|on the line)|we have (?:the )?next question|"
    r"press (?:star|\*) (?:and|then|followed by) (?:one|1|zero|0)|question[- ]and[- ]answer session|"
    r"listen[- ]only mode|the line for|please go ahead|over to you|(?:that|this) (?:was|is) the last question|"
    r"conclude (?:the|this) (?:conference|call)|you may now disconnect|joining us today|"
    r"rejoined? the (?:question )?queue|kindly mute|"
    # operator boilerplate seen in real ICICI calls (Whisper sometimes hears "star" as "Southern")
    r"as a reminder|participants'? lines|conference is being recorded|signal an operator|"
    r"touch[- ]?(?:tone|stone)|question queue|use handsets|press (?:star|\*|\w+) (?:and|then) (?:two|2)|"
    r"wait for a moment|may press star|anyone who wishes to ask",
    re.I,
)
ANNOUNCE_RE = re.compile(r"question (?:is |comes )?from|next question|line of ", re.I)
QA_START_RE = re.compile(
    r"question[- ]and[- ]answer session|first question (?:is |comes )?from|begin (?:the|with the) question", re.I)
ANALYST_RE = re.compile(
    r"thanks? (?:you )?for taking (?:my|the|our) questions?|my (?:first )?question is|"
    r"congratulations (?:on|for)|i have (?:a|two|few|a couple of) questions?|just a follow[- ]?up|"
    r"(?:a|one) (?:quick )?(?:question|clarification)|(?:could|can) you (?:please )?(?:give|help|explain|share|throw)",
    re.I,
)
PREPARED_MIN_S = 30.0   # a speaker with this much prepared-remarks time is management, not an analyst


def utterances_path(call_id: str) -> Path:
    return load_settings().path("utterances") / f"{call_id}.parquet"


# --------------------------------------------------------------------------- 1. words -> speakers

def transcript_words(transcript: dict) -> pd.DataFrame:
    rows = [{"seg_id": s["id"], "start": w["start"], "end": w["end"], "word": w["word"]}
            for s in transcript["segments"] for w in s["words"]]
    return pd.DataFrame(rows, columns=["seg_id", "start", "end", "word"])


def assign_speakers(words: pd.DataFrame, turns: list[dict], max_gap_s: float = 1.0) -> pd.Series:
    """Speaker with the largest time overlap per word; if none overlaps, the nearest turn within max_gap_s."""
    if not turns or words.empty:
        return pd.Series([None] * len(words), index=words.index, dtype=object)
    t_start = np.array([t["start"] for t in turns])
    t_end = np.array([t["end"] for t in turns])
    spk = np.array([t["speaker"] for t in turns], dtype=object)
    out = []
    for ws, we in zip(words["start"].to_numpy(), words["end"].to_numpy()):
        overlap = np.minimum(we, t_end) - np.maximum(ws, t_start)
        i = int(np.argmax(overlap))
        if overlap[i] > 0:
            out.append(spk[i])
            continue
        mid = (ws + we) / 2
        dist = np.where(mid < t_start, t_start - mid, np.where(mid > t_end, mid - t_end, 0))
        j = int(np.argmin(dist))
        out.append(spk[j] if dist[j] <= max_gap_s else None)
    return pd.Series(out, index=words.index, dtype=object)


# --------------------------------------------------------------------------- 2. utterances

def build_utterances(words: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive words with the same speaker. Unassigned words join the current utterance."""
    rows, cur = [], None
    for w in words.itertuples(index=False):
        spk = w.speaker if w.speaker is not None else (cur["speaker"] if cur else None)
        if cur is None or spk != cur["speaker"]:
            if cur:
                rows.append(cur)
            cur = {"speaker": spk, "words": []}
        cur["words"].append({"start": w.start, "end": w.end, "word": w.word})
    if cur:
        rows.append(cur)
    return _finalise(rows)


def utterances_from_segments(transcript: dict) -> pd.DataFrame:
    """Fallback without diarization: one utterance per Whisper segment, speaker unknown."""
    rows = [{"speaker": None, "words": s["words"]} for s in transcript["segments"] if s["words"]]
    return _finalise(rows)


def join_words(words: list[str]) -> str:
    """Whisper splits '11.9%' into tokens '11', '.9', '%'. Re-glue punctuation and hyphens."""
    text = " ".join(w for w in words if w)
    text = re.sub(r"\s+(?=[.,%?!:;)\]'’])", "", text)   # "11 .9 %" -> "11.9%", "word ," -> "word,"
    text = re.sub(r"(?<=\w)\s+-(?=\w)", "-", text)       # "year -on -year" -> "year-on-year"
    return text.strip()


def _finalise(rows: list[dict]) -> pd.DataFrame:
    out = []
    for i, r in enumerate(rows):
        ws = r["words"]
        out.append({"utt_id": i, "speaker": r["speaker"], "start": ws[0]["start"], "end": ws[-1]["end"],
                    "n_words": len(ws), "text": join_words([w["word"] for w in ws]),
                    "words_json": json.dumps(ws)})
    return pd.DataFrame(out, columns=["utt_id", "speaker", "start", "end", "n_words", "text", "words_json"])


# --------------------------------------------------------------------------- 3. sections

def find_qa_start(utts: pd.DataFrame) -> float:
    """Start time of the first utterance that opens Q&A; +inf if none is found (all prepared)."""
    hits = utts[utts["text"].str.contains(QA_START_RE)]
    return float(hits["start"].iloc[0]) if len(hits) else float("inf")


# --------------------------------------------------------------------------- 4. roles

def label_roles(utts: pd.DataFrame, qa_start: float) -> tuple[pd.DataFrame, dict]:
    """Return utterances with `role` + `section`, and the per-speaker evidence used."""
    utts = utts.copy()
    utts["section"] = np.where(utts["start"] < qa_start, "prepared", "qa")
    utts["is_mod_text"] = utts["text"].str.contains(MODERATOR_RE)
    utts["is_analyst_text"] = utts["text"].str.contains(ANALYST_RE)

    if utts["speaker"].isna().all():
        return _label_by_text(utts), {"method": "text"}
    utts["speaker"] = utts["speaker"].fillna("SPEAKER_UNASSIGNED")  # words no diarization turn covered

    ev: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    dur = utts["end"] - utts["start"]
    for spk, g in utts.groupby("speaker"):
        ev[spk]["mod_cues"] = float(g["is_mod_text"].sum())
        ev[spk]["analyst_cues"] = float(g["is_analyst_text"].sum())
        ev[spk]["prepared_s"] = float(dur[g.index][g["section"] == "prepared"].sum())
        ev[spk]["total_s"] = float(dur[g.index].sum())

    moderator = max(ev, key=lambda s: (ev[s]["mod_cues"], -ev[s]["total_s"]))
    if ev[moderator]["mod_cues"] < 2:
        moderator = None

    # Whoever speaks right after the moderator announces a questioner is (very likely) that analyst.
    seq = utts[["speaker", "text", "section"]].to_dict("records")
    for prev, nxt in zip(seq, seq[1:]):
        if prev["speaker"] == moderator and nxt["speaker"] != moderator and ANNOUNCE_RE.search(prev["text"]):
            ev[nxt["speaker"]]["announced"] += 1

    roles = {}
    for spk, e in ev.items():
        if spk == moderator:
            roles[spk] = "moderator"
        elif e["prepared_s"] >= PREPARED_MIN_S:
            roles[spk] = "management"
        elif e["announced"] + e["analyst_cues"] > 0:
            roles[spk] = "analyst"
        else:
            roles[spk] = None  # decided below from conversational position

    # Remaining speakers: answering right after an analyst -> management.
    for prev, nxt in zip(seq, seq[1:]):
        if roles.get(nxt["speaker"], "x") is None and roles.get(prev["speaker"]) == "analyst" and nxt["section"] == "qa":
            ev[nxt["speaker"]]["answers"] += 1
    for spk in ev:
        if roles[spk] is None:
            roles[spk] = "management" if ev[spk]["answers"] > 0 else "unknown"

    utts["role"] = utts["speaker"].map(roles)
    # Before Q&A, any non-moderator text is the scripted management presentation.
    utts.loc[(utts["section"] == "prepared") & (utts["role"] == "unknown"), "role"] = "management"
    evidence = {"method": "diarization", "moderator": moderator,
                "speakers": {s: {**{k: round(v, 1) for k, v in e.items()}, "role": roles[s]} for s, e in ev.items()}}
    return utts.drop(columns=["is_mod_text", "is_analyst_text"]), evidence


def _label_by_text(utts: pd.DataFrame) -> pd.DataFrame:
    """Text-only roles, following the rhythm of a concall Q&A (a small state machine):

      moderator announces "next question is from X"  -> state = analyst
      analyst asks; while segments contain '?' (or analyst phrases) we stay in analyst
      first segment after a question that is not itself a question -> state = management
      management answers until the next announcement, or an analyst follow-up phrase
    """
    roles, state, prev_question = [], "management", False
    questions = utts["text"].str.contains("?", regex=False).tolist()
    mod = utts["is_mod_text"].tolist()

    def question_ahead(i: int, k: int = 2) -> bool:
        """Is one of the next k segments a question, before the moderator speaks again?"""
        for j in range(i + 1, min(i + 1 + k, len(questions))):
            if mod[j]:
                return False
            if questions[j]:
                return True
        return False

    for i, u in enumerate(utts.itertuples(index=False)):
        text = u.text.strip()
        # Analysts often give context between questions ("Because loan growth is a tad lower...").
        # Treat a non-question as part of the question if a question follows within 2 segments.
        is_question = questions[i] or question_ahead(i)
        if u.is_mod_text:
            roles.append("moderator")
            if ANNOUNCE_RE.search(text) and u.section == "qa":
                state, prev_question = "analyst", False
            continue
        if u.section == "prepared":
            roles.append("management")
            continue
        if state == "analyst" and prev_question and not is_question and not u.is_analyst_text:
            state = "management"              # the questions are over; this is the answer
        elif state == "management" and u.is_analyst_text:
            state = "analyst"                 # e.g. "just a follow-up" after an answer
        roles.append(state)
        prev_question = is_question if state == "analyst" else False
    utts["role"] = roles
    return utts.drop(columns=["is_mod_text", "is_analyst_text"])


# --------------------------------------------------------------------------- driver

def build(call_id: str, transcript: dict, diarization: dict | None, force: bool = False) -> Path:
    out = utterances_path(call_id)
    if out.exists() and not force:
        return out
    if diarization and diarization.get("turns"):
        words = transcript_words(transcript)
        words["speaker"] = assign_speakers(words, diarization["turns"])
        utts, source = build_utterances(words), "diarization"
    else:
        utts, source = utterances_from_segments(transcript), "text"
    utts, evidence = label_roles(utts, find_qa_start(utts))
    utts.insert(0, "call_id", call_id)
    utts["role_source"] = source
    utts.to_parquet(out, index=False)
    out.with_suffix(".roles.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    return out
