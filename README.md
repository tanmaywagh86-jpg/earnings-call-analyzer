# Earnings Call Voice & Sentiment Analyzer

[![docker](https://github.com/tanmaywagh86-jpg/earnings-call-analyzer/actions/workflows/docker.yml/badge.svg)](https://github.com/tanmaywagh86-jpg/earnings-call-analyzer/actions/workflows/docker.yml)

> **Status:** All six phases are implemented and tested, and the pipeline has been run end to end on **20 real earnings calls** (ICICI, HDFC and Axis Bank). See [Results](#results) below, [`BUILD_LOG.md`](BUILD_LOG.md) for the build journal and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the code structure.

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
- [x] **Phase 3:** Text signals: FinBERT sentiment, hedging language, LLM (Claude) evasiveness flags, guidance + KPI extraction *(LLM step is optional and needs `ANTHROPIC_API_KEY`; it was not run for the results below)*
- [x] **Phase 4:** Quarter-over-quarter tone comparison per company
- [x] **Phase 5:** Event study: cumulative abnormal returns vs. Nifty 50, post-call volatility and volume, YoY growth controls, **company fixed effects**
- [x] **Phase 6:** FastAPI service returning a per-call report; Docker

## Results

**Sample:** 20 calls by three private banks (ICICI Bank 12, HDFC Bank 4, Axis Bank 4), FY24–FY27, from recordings published by the companies. All are transcribed, diarized and role-labelled, and the market reaction is measured against the Nifty 50.

### 1. Management sounds different off-script (strong, consistent)

| Management, per call (median of 20) | Prepared remarks | Analyst Q&A | Calls where Q&A is higher |
|---|---|---|---|
| Hedging words per 1,000 words | 1.6 | **37.7** | 19 / 20 |
| Articulation rate (words / s) | 2.64 | 2.75 | 16 / 20 |
| Pauses per minute | 8.1 | 7.4 | 2 / 20 |
| Pitch variation (semitones) | 4.6 | 4.2 | 3 / 20 |

Executives hedge over **20× more** when answering live questions than when reading the script. In Q&A they also speak a little faster, with fewer pauses and flatter pitch. This scripted-vs-spontaneous contrast is the signal the project is built around, and it shows up in almost every call.

### 2. Does the call predict the stock's reaction? Not in this sample

Market reaction = cumulative abnormal return (CAR) vs. Nifty 50 from the first trading day after the call. Regressions use OLS with HC3 robust errors and standardised signals, fitted both **pooled** and with **company fixed effects** (each call compared only with the same bank's other calls).

| Day-0 CAR model | adj. R², pooled | adj. R², company FE |
|---|---|---|
| Bank identity only | n/a | **0.41** |
| + management tone + hedging | 0.21 | 0.34 |
| + pitch variation + pauses | 0.21 | 0.27 |

- **Pooled, hedging looked "significant"** (+1.4 pp of day-0 CAR per SD, p = 0.043; Spearman ρ = +0.55).
- **It was a company effect.** Axis Bank management hedges least and sounds most upbeat, and its four calls happened to be followed by bad days (mean day-0 CAR −4.0%, vs +0.9% for ICICI). With company fixed effects the hedging coefficient falls to +0.2 pp (p = 0.79). No signal reaches p < 0.2 in any fixed-effects model, and within-bank correlations are all |ρ| ≤ 0.30.
- Adding call features to the bank dummies *lowers* adjusted R²: in this sample, which bank it is explains more than anything said on the call.

**Takeaway:** there is no evidence yet that tone, hedging or voice predicts the stock's reaction once each bank is compared only with itself. With 20 calls and only 4 per bank for two of the banks, this is a low-power null result, not proof of no effect. The main lesson is about method: pooling companies without fixed effects produced a convincing-looking false positive.

Result files: [`data/results/`](data/results/) (`regressions.txt`, `correlations.csv`, `correlations_within.csv`, `event_study.csv`).

## Quick start

```bash
python -m concall run      # all stages for every call in data/manifest/calls.csv
python -m concall study    # feature table, event study, regressions -> data/results/
python -m concall serve    # API at http://127.0.0.1:8000/docs
```

## Tech Stack

Python · Whisper · pyannote · Librosa · FinBERT (Hugging Face Transformers) · LLMs · Pandas · NumPy · yfinance · FastAPI · Docker

## Limitations

- **Small sample:** 20 calls, 3 banks, one sector. Within-bank tests have little power; results are exploratory, not a trading strategy.
- **No earnings-surprise control yet:** the YoY growth controls are extracted by the optional LLM step, which was not run, so the models do not yet test "beyond the reported numbers". Free consensus estimates for Indian stocks are limited, so YoY growth is the planned proxy for surprise.
- **Speaker labels are automatic:** pyannote often splits one executive into several speakers (14–19 on Axis calls), and roles come from rules on the transcript. Voice features are sensitive to labelling errors; switching from text-only to diarized roles flipped the sign of one feature (see `BUILD_LOG.md`).
- **FinBERT** scores factual declines ("NPA declined by 0.4%") as negative, so tone is noisy.
- Correlation between tone and returns would not imply causation.
- 11 more calls (10 Axis Bank, ICICI FY26Q1 without audio) are in the manifest but not processed.

---

*Author: Tanmay Wagh · B.Tech AI & ML, Walchand College of Engineering, Sangli · Minor in FinTech*
