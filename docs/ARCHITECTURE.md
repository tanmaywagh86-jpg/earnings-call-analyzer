# Architecture

> Living document: describes the project **as it is now**. Updated at the end of every phase.
> For *how we got here* (decisions, problems, lessons) see [`../BUILD_LOG.md`](../BUILD_LOG.md).

**Location:** `D:\clduse\bankingpr\bnk` (moved off a full C: drive, see BUILD_LOG Step 1.9).
**Built:** all six phases. Optional parts switch on when keys are present in `.env`:
`HF_TOKEN` → real speaker diarization; `ANTHROPIC_API_KEY` → LLM guidance/evasiveness/KPIs.

---

## 1. The big picture

```
data/manifest/calls.csv  (which calls exist, when; YouTube/URL/local audio source)
          │
          ▼  ───────────────────────── per call (pipeline.run_call) ─────────────────────────
 PHASE 1  prepare     audio.py       yt-dlp / requests / local ──ffmpeg──► 16 kHz mono FLAC
          transcribe  transcribe.py  faster-whisper (GPU via torch's CUDA DLLs, else CPU)
                                     + finance-vocabulary prompt ──► words with timestamps
                                     (in ~10-min chunks cut at silences, to cap RAM)
          voice       voice.py       pitch (YIN + periodicity gate), pauses, speech rate per segment
 PHASE 2  diarize     diarize.py     pyannote: anonymous SPEAKER_00.. turns        [needs HF_TOKEN]
          roles       roles.py       words→speakers→utterances; moderator / management / analyst;
                                     prepared remarks vs Q&A   (text-only rules if no diarization)
          voice_utt   voice.py       same voice features per speaker-pure utterance
 PHASE 3  text        text.py        FinBERT P(+), P(−) per sentence; hedging lexicon counts
          llm         llm.py         Claude structured output: guidance (raised/maintained/lowered),
                                     evasive answers, sector KPIs, YoY growth     [needs API key]
          ▼  ───────────────────────── across calls (pipeline.run_study) ───────────────────────
 PHASE 4  tone.py     one row per call: <who>_<signal>_<section>, Q&A-minus-prepared contrasts,
                      QoQ changes d_<feature> (consecutive quarters only)
 PHASE 5  event.py    yfinance prices → day 0 (after-close rule) → AR vs Nifty 50 → CAR[0,0],[0,4],[0,29],
                      abnormal volume, volatility ratio → Spearman correlations + OLS (HC3) M1..M4
 PHASE 6  api.py      FastAPI: /calls, /calls/{id}/report, /calls/{id}/process, /results/regressions
          Dockerfile  CPU image; data/ and models/ mounted as volumes
```

## 2. Design principles

| Principle | What it means in the code |
|---|---|
| **One stage = one module = one output folder** | Rerun or replace a single stage without touching others. `pipeline.STAGES` lists the order. |
| **`call_id` is the join key everywhere** | `ICICIBANK_FY25Q2`. Every file is `<call_id>.<ext>`. |
| **Idempotent, resumable stages** | Skip a call if its output exists; `--force` redoes. Crashes/shutdowns lose nothing. |
| **Atomic writes** | Write `*.part`, then rename. A half-written file never looks finished. |
| **Failures are per call** | One broken link logs an error; the batch continues. |
| **Optional components degrade gracefully** | No `HF_TOKEN` → text-rule roles. No API key → LLM features absent; regressions use what exists. |
| **All tunables in `config/settings.yaml`, secrets in `.env`** | No magic numbers in code; no keys in git. |
| **Every automated judgement is auditable** | Role votes saved to `<id>.roles.json`; every LLM item carries a verbatim quote. |
| **Pure functions for the maths** | `day0_position`, `abnormal_returns`, `car`, `label_roles`, `add_qoq` take data in, return data out, so they're unit-tested with known answers. |

## 3. Folder map

```
bnk/
├── README.md, BUILD_LOG.md, docs/ARCHITECTURE.md
├── config/settings.yaml          all parameters and paths
├── config/lexicons/hedging.txt   hedging/uncertainty words (Loughran-McDonald-style)
├── .env.example                  template for HF_TOKEN / ANTHROPIC_API_KEY (copy to .env)
├── Dockerfile, .dockerignore, pyproject.toml, requirements.txt
├── src/concall/
│   ├── config.py      settings, .env, HF_HOME→models/, torch CUDA DLLs for Whisper
│   ├── manifest.py    load/validate calls.csv (dup recordings, dates), quarter_index
│   ├── audio.py       YouTube (yt-dlp) / URL / local → FLAC
│   ├── transcribe.py  Whisper
│   ├── voice.py       acoustic features (segments + utterances)
│   ├── diarize.py     pyannote
│   ├── roles.py       speaker roles + sections
│   ├── text.py        FinBERT + hedging
│   ├── llm.py         Claude extraction (Pydantic schema)
│   ├── tone.py        call-level features + QoQ
│   ├── event.py       event study + regressions
│   ├── pipeline.py    stage order, status, study driver
│   ├── api.py         FastAPI
│   └── cli.py         `python -m concall …`
├── tests/                         known-answer tests for every module's logic
├── models/                        downloaded models (git-ignored)
└── data/                          (git-ignored except manifest/)
    ├── manifest/                  calls.csv, companies.csv
    ├── audio/{raw,clean,meta}/    downloads, FLAC, source titles/durations
    ├── transcripts/ diarization/ utterances/ text/ llm/
    ├── features/                  voice_segments/, voice_utterances/, calls.parquet/.csv
    ├── market/                    cached prices
    └── results/                   event_study.csv, correlations.csv, regressions.csv/.txt
```

## 4. Data contracts

| File | Grain | Key columns |
|---|---|---|
| `calls.csv` | call | call_id, nse_symbol, sector, fiscal_quarter, call_date, call_time_ist, audio_source |
| `transcripts/<id>.json` | segment → words | start, end, text, words[start, end, word, prob] |
| `diarization/<id>.json` | turn | start, end, speaker |
| `utterances/<id>.parquet` | utterance | utt_id, speaker, role, section, start, end, text, words_json, role_source |
| `features/voice_utterances/<id>.parquet` | utterance | role, section, f0_std_st, articulation_rate_wps, n_pauses, … |
| `text/<id>.parquet` | sentence | role, section, p_pos, p_neg, p_neu, tone, n_hedges, n_words |
| `llm/<id>.json` | call | summary, outlook, guidance[], evasive_answers[], kpis[], yoy_*_growth_pct |
| `features/calls.parquet` | call | mgmt_tone_qa, mgmt_hedges_per_1k_all, mgmt_f0_std_st_qa, evasive_rate, guidance_score, d_* (QoQ) |
| `results/event_study.csv` | call | day0, car_0_0, car_0_4, car_0_29, abn_volume_0_4, vol_ratio_0_9 |
| `results/regressions.csv` | car × model × variable | coef_pct_per_sd, t_hc3, p, n, r2, adj_r2 |

## 5. How to run

```bash
cd D:\clduse\bankingpr\bnk
.venv\Scripts\activate
python -m concall validate               # check the manifest
python -m concall status                 # which stages are done per call
python -m concall run                    # all stages, all calls (resumable)
python -m concall run --call ICICIBANK_FY25Q2
python -m concall study                  # Phases 4-5 → data/results/
python -m concall serve                  # API → http://127.0.0.1:8000/docs
pytest                                   # tests
```

## 6. Known limitations

- Small sample (14 calls processed: 12 ICICI + 2 HDFC): results are exploratory, not a trading
  strategy. First batch found no robust signal (see BUILD_LOG "Batch results").
- Role labelling is rule-based; check `<id>.roles.json` when numbers look odd.
- Without `HF_TOKEN`, Q&A roles come from text cues only (question marks, "thanks for taking my question").
- YoY growth controls come from what management *says* on the call (LLM-extracted); there's no free
  structured source of Indian quarterly financials or consensus estimates.
- Voice pitch is affected by phone-line quality; analysts dial in, management often sits in a studio.
  That's why the voice features compare management with itself (Q&A vs prepared, QoQ).
- Memory: this laptop often has only ~2–3 GB free. Transcription is chunked (`chunk_s`) so a long
  call stays at ~200 MB of spectrogram; long calls on CPU are still slow (see BUILD_LOG 2.8).
