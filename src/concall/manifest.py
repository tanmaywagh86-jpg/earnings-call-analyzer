"""The call manifest: a CSV with one row per earnings call.

Every later stage is keyed on `call_id`, so the manifest is the single source of truth for
which calls exist and when they happened. The date + time matter for the event study
(Phase 5): a call after market close (15:30 IST) moves the stock on the *next* trading day.

Columns
-------
call_id         {NSE_SYMBOL}_{FISCAL_QUARTER}, e.g. HDFCBANK_FY25Q2 (built automatically if blank)
company         display name
nse_symbol      NSE ticker without suffix (yfinance needs "<symbol>.NS")
sector          Banks / IT / ... (decides which sector metrics are extracted in Phase 3)
fiscal_quarter  Indian fiscal year: FY25Q1 = Apr-Jun 2024, FY25Q2 = Jul-Sep 2024, ...
call_date       YYYY-MM-DD
call_time_ist   HH:MM (24h); blank = unknown, treated as after market close
audio_source    http(s) URL (direct file or YouTube) or a local file path
notes           free text
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import load_settings

COLUMNS = [
    "call_id", "company", "nse_symbol", "sector", "fiscal_quarter",
    "call_date", "call_time_ist", "audio_source", "notes",
]
_QUARTER_RE = re.compile(r"^FY(\d{2})Q([1-4])$")


def make_call_id(symbol: str, fiscal_quarter: str) -> str:
    return f"{symbol.strip().upper()}_{fiscal_quarter.strip().upper()}"


def quarter_index(fiscal_quarter: str) -> int:
    """FY25Q2 -> 25*4 + 1 = 101. Consecutive quarters differ by exactly 1 (used for QoQ in Phase 4)."""
    m = _QUARTER_RE.match(fiscal_quarter.strip().upper())
    if not m:
        raise ValueError(f"bad fiscal_quarter: {fiscal_quarter}")
    return int(m.group(1)) * 4 + int(m.group(2)) - 1


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    path = path or load_settings().path("manifest")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    blank = df["call_id"].str.strip() == ""
    df.loc[blank, "call_id"] = [
        make_call_id(s, q) for s, q in zip(df.loc[blank, "nse_symbol"], df.loc[blank, "fiscal_quarter"])
    ]
    validate(df)
    return df.set_index("call_id", drop=False)


def youtube_id(source: str) -> str | None:
    m = re.search(r"(?:youtu\.be/|[?&]v=|/embed/)([A-Za-z0-9_-]{11})", source or "")
    return m.group(1) if m else None


def _source_key(source: str) -> str:
    """youtu.be/X, youtube.com/watch?v=X and youtube.com/embed/X all compare equal."""
    vid = youtube_id(source)
    return f"youtube:{vid}" if vid else source.strip()


def validate(df: pd.DataFrame) -> None:
    problems: list[str] = []
    dupes = df["call_id"][df["call_id"].duplicated()].tolist()
    if dupes:
        problems.append(f"duplicate call_id: {dupes}")

    # The same recording listed for two different calls is almost always a copy-paste error on the
    # company's website (this happened with ICICI Bank's Q4FY25 / Q1FY26 links).
    keys = df["audio_source"].map(lambda s: _source_key(s) if s.strip() else "")
    for key, group in df[keys != ""].groupby(keys[keys != ""]):
        if len(group) > 1:
            problems.append(f"same audio_source used by {group['call_id'].tolist()}: {key}")

    for _, row in df.iterrows():
        cid = row["call_id"]
        if not _QUARTER_RE.match(row["fiscal_quarter"].upper()):
            problems.append(f"{cid}: fiscal_quarter '{row['fiscal_quarter']}' should look like FY25Q2")
        try:
            if pd.isna(pd.Timestamp(row["call_date"])):
                raise ValueError
        except ValueError:
            problems.append(f"{cid}: bad call_date '{row['call_date']}'")
        if row["call_time_ist"] and not re.match(r"^\d{1,2}:\d{2}$", row["call_time_ist"]):
            problems.append(f"{cid}: call_time_ist '{row['call_time_ist']}' should be HH:MM")
    if problems:
        raise ValueError("manifest problems:\n  " + "\n  ".join(problems))
