# Earnings Call Voice & Sentiment Analyzer

[![docker](https://github.com/tanmaywagh86-jpg/earnings-call-analyzer/actions/workflows/docker.yml/badge.svg)](https://github.com/tanmaywagh86-jpg/earnings-call-analyzer/actions/workflows/docker.yml)

> **Status:** All six phases are implemented and tested, and the pipeline has been run end to end on **all 31 earnings calls** in the manifest (ICICI, HDFC and Axis Bank). See [Results](#results) below, [`BUILD_LOG.md`](BUILD_LOG.md) for the build journal and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the code structure.

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

- [x] **Phase 1:** Data collection pipeline (company-published recordings), Whisper transcription (GPU), Librosa voice features. *Sample: 31 processed calls: ICICI Bank (13), HDFC Bank (4), Axis Bank (14)*
- [x] **Phase 2:** Speaker diarization (pyannote, needs HF token) + rule-based role labelling; separate management remarks from analyst Q&A
- [x] **Phase 3:** Text signals: FinBERT sentiment, hedging language, LLM (Claude) evasiveness flags, guidance + KPI extraction *(LLM step is optional and needs `ANTHROPIC_API_KEY`; it was not run for the results below)*
- [x] **Phase 4:** Quarter-over-quarter tone comparison per company
- [x] **Phase 5:** Event study: cumulative abnormal returns vs. Nifty 50, post-call volatility and volume, YoY growth controls, **company fixed effects**
- [x] **Phase 6:** FastAPI service returning a per-call report; Docker

## Results

**Sample:** 31 calls by three private banks (ICICI Bank 13, HDFC Bank 4, Axis Bank 14), FY24–FY27, from recordings published by the companies. All are transcribed, diarized and role-labelled, and the market reaction is measured against the Nifty 50.

### 1. Management sounds different off-script (strong, consistent)

| Management, per call (median of 31) | Prepared remarks | Analyst Q&A | Calls where Q&A is higher |
|---|---|---|---|
| Hedging words per 1,000 words | 2.1 | **19.9** | 30 / 31 |
| Articulation rate (words / s) | 2.73 | 2.79 | 27 / 31 |
| Pauses per minute | 9.0 | 8.1 | 5 / 31 |
| Pitch variation (semitones) | 4.6 | 4.3 | 6 / 31 |

Executives hedge about **10× more** when answering live questions than when reading the script, in 30 of 31 calls. In Q&A they also speak a little faster, with fewer pauses and flatter pitch. This scripted-vs-spontaneous contrast is the signal the project is built around, and it shows up in almost every call.

### 2. Does the *level* of tone predict the stock's reaction? No

Market reaction = cumulative abnormal return (CAR) vs. Nifty 50 from the first trading day after the call. Regressions use OLS with HC3 robust errors and standardised signals, fitted both **pooled** and with **company fixed effects** (each call compared only with the same bank's other calls).

- **The reported numbers don't explain it either.** Controls are YoY growth in net interest income and standalone profit, taken from the banks' published results ([`data/manifest/fundamentals.csv`](data/manifest/fundamentals.csv), sources per row). The controls-only model has adjusted R² of −0.06 to +0.02. Growth that the market already expected isn't a surprise, and free consensus estimates for Indian banks aren't available.
- Adding tone, hedging, pitch and pauses on top doesn't help. Every model (day 0, days 0–4, days 0–29; pooled or fixed effects) has adjusted R² between −0.35 and +0.02, and no call feature reaches p < 0.12 (none below 0.3 with fixed effects).
- **A false positive along the way:** with 20 calls (only 4 from Axis), hedging looked significant (p = 0.043). It was a company effect: those 4 Axis calls happened to be followed by bad days (mean day-0 CAR −4.0%). Company fixed effects removed it (p = 0.79), and with all 14 Axis calls the Axis mean is −1.0%, so the "bank effect" itself was largely noise. See `BUILD_LOG.md` steps 2.10–2.13.

### 3. Does the *change* in tone vs. last quarter? A weakening lead

Comparing a bank with **itself one quarter earlier** removes each management team's baseline style. With 30 calls (23 with a previous quarter), this looked promising: tone change vs. days 0–4 CAR had ρ = +0.58 (p = 0.004), and p = 0.038 with company fixed effects. It was noticed after looking at ~24 correlations, though, so it needed checking on calls that weren't used to find it.

Adding ICICI FY26Q1 created two new ICICI pairs, a small out-of-sample check, and they went against the lead. On 31 calls (25 pairs):

| Tone change vs. previous quarter → CAR | Spearman ρ | p | Company FE | Company FE + YoY growth |
|---|---|---|---|---|
| Day 0 | +0.44 | 0.027 | +1.1 pp / SD, p = 0.16 | +0.9 pp / SD, p = 0.29 |
| Days 0–4 | +0.51 | 0.010 | +1.9 pp / SD, p = 0.082 | +1.7 pp / SD, p = 0.13 |
| Days 0–29 | +0.15 | 0.49 | +0.5 pp / SD, p = 0.69 | +0.1 pp / SD, p = 0.94 |

It now rests almost entirely on **Axis Bank** (ρ = 0.85 over 13 pairs for days 0–4); for ICICI it fell from 0.43 to 0.18. Once growth and bank are controlled for, it is not significant.

**Takeaway:** off-script speech is clearly different from scripted speech, but neither the tone of a call, its change from last quarter, nor the reported growth predicts the stock's reaction in this sample. The tone-change effect may be specific to one bank or may be chance; only future quarters can tell. Method lessons: pooling companies without fixed effects produced a convincing false positive at N = 20, and the "best" feature found by scanning many weakened as soon as new data arrived.

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

- **Small sample:** 31 calls, 3 banks (only 4 from HDFC), one sector. Results are exploratory, not a trading strategy.
- **YoY growth is a weak proxy for surprise:** free consensus estimates for Indian banks aren't available, so the controls are YoY growth in NII and PAT. Two values are deliberately blank: HDFC FY25Q1 (base quarter predates the HDFC Ltd merger) and Axis FY24Q4 profit (base quarter was a loss from the Citibank India acquisition).
- **Speaker labels are automatic:** pyannote often splits one executive into several speakers (14–19 on Axis calls), and roles come from rules on the transcript. Voice features are sensitive to labelling errors; switching from text-only to diarized roles flipped the sign of one feature (see `BUILD_LOG.md`).
- **FinBERT** scores factual declines ("NPA declined by 0.4%") as negative, so tone is noisy.
- Correlation between tone and returns would not imply causation.
---

*Author: Tanmay Wagh · B.Tech AI & ML, Walchand College of Engineering, Sangli · Minor in FinTech*
