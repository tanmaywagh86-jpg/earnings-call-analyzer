# Earnings Call Voice & Sentiment Analyzer

> **Status: In Progress.** All six phases are implemented and tested; real ICICI Bank / HDFC Bank calls are being processed. See [`BUILD_LOG.md`](BUILD_LOG.md) for the build journal and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the code structure.

Every quarter, listed companies hold **earnings calls** where management presents results and answers analysts' questions. Analysts listen not only to *what* executives say but *how* they say it: hesitation, evasive answers and shifts in tone.

This project analyzes earnings-call recordings of **NSE-listed companies** using both the **text** and the **voice** of management, then tests whether those signals relate to how the stock moves after the call.

---

## Pipeline

```
Concall audio ─► Whisper transcription ─► Speaker diarization ─► Management vs. Analyst Q&A
                                                                        │
                       ┌────────────────────────────────────────────────┤
                       ▼                                                ▼
              Text signals                                       Voice signals
   • FinBERT financial sentiment                       • Pitch variation
   • Hedging-language frequency                        • Speech rate
   • LLM-flagged evasive answers                       • Pause length
   • Guidance: raised / maintained / cut               • Scripted vs. Q&A contrast
                       │                                                │
                       └──────────────► Quarter-over-quarter tone change ◄┘
                                                    │
                                                    ▼
                         Event study vs. market (yfinance, Nifty 50)
               Cumulative abnormal returns · volatility · volume after the call
```

## Methodology

| Component | Approach |
|---|---|
| **Transcription** | OpenAI Whisper |
| **Speaker separation** | pyannote speaker diarization; speaker roles identified from moderator introductions |
| **Text sentiment** | FinBERT (finance-tuned BERT) |
| **Evasiveness & guidance** | LLM-based extraction of forward-looking statements, labeled raised / maintained / cut vs. the previous quarter |
| **Voice features** | Librosa: pitch (F0) variation, speech rate, pause length |
| **Market reaction** | **Abnormal return = stock return − Nifty 50 return**, accumulated over 1 / 5 / 30 trading days (CAR) |
| **Event timing** | Calls held after market close (3:30 PM IST) use the next trading day as day 0, to avoid look-ahead bias |
| **Controls** | YoY revenue and profit growth, to test whether tone adds information **beyond the reported numbers** |

### Sector-specific metrics extracted
- **Banks:** NPA, NIM, CASA ratio, credit cost
- **IT:** deal wins, attrition, operating margin
- **All sectors:** revenue growth, EBITDA margin

## Roadmap

- [x] **Phase 1:** Data collection pipeline (company-published recordings), Whisper transcription (GPU), Librosa voice features. *Sample so far: 20 processed calls: ICICI Bank (12), HDFC Bank (4), Axis Bank (4); 31 in the manifest*
- [x] **Phase 2:** Speaker diarization (pyannote, needs HF token) + rule-based role labelling; separate management remarks from analyst Q&A
- [x] **Phase 3:** Text signals: FinBERT sentiment, hedging language, LLM (Claude) evasiveness flags, guidance + KPI extraction
- [x] **Phase 4:** Quarter-over-quarter tone comparison per company
- [x] **Phase 5:** Event study: cumulative abnormal returns vs. Nifty 50, post-call volatility and volume, YoY growth controls
- [x] **Phase 6:** FastAPI service returning a per-call report; Docker

## Quick start

```bash
python -m concall run      # all stages for every call in data/manifest/calls.csv
python -m concall study    # feature table, event study, regressions -> data/results/
python -m concall serve    # API at http://127.0.0.1:8000/docs
```

## Tech Stack

Python · Whisper · pyannote · Librosa · FinBERT (Hugging Face Transformers) · LLMs · Pandas · NumPy · yfinance · FastAPI · Docker

## Limitations (planned to be reported honestly)

- A small sample of companies and quarters limits statistical power; results are exploratory, not a trading strategy.
- Correlation between tone and returns does not imply causation.
- Free consensus estimates for Indian stocks are limited, so YoY growth is used as a proxy for earnings surprise.

---

*Author: Tanmay Wagh · B.Tech AI & ML, Walchand College of Engineering, Sangli · Minor in FinTech*
