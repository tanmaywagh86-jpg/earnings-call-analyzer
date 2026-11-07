# Build Log

> A chronological journal of building this project: **what** was done, **why** each decision
> was made, problems hit, and concepts worth learning. Newest phase at the bottom.
> For the current structure of the code, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Phase 1: Audio → Transcript → Voice features  (started 2026-11-06)

### Goal
Turn an earnings-call recording into (a) a text transcript with the exact time of every word,
and (b) numbers describing *how* it was spoken: pitch variation, speech rate, pauses.

### Step 1.1: Environment check
| Item | Found | Consequence |
|---|---|---|
| Python | 3.12.10 | good; most ML libs support it |
| ffmpeg | installed | used to convert any audio format |
| GPU | RTX 3050 Laptop, 6 GB | could speed up Whisper ~5–10× |
| **Free disk on C:** | **6.6 GB of 218 GB** | **biggest constraint**; drives Decisions 1–3 |

Created an isolated virtual environment `bnk/.venv` so project packages don't clash with the
system Python. Installed with `pip --no-cache-dir` to avoid keeping a second copy of every wheel.

### Decision 1: `faster-whisper` instead of `openai-whisper`
- `openai-whisper` needs PyTorch. PyTorch with CUDA is ~4–5 GB installed; that alone would fill the disk.
- `faster-whisper` is the same Whisper model re-implemented on **CTranslate2**: ~4× faster,
  less memory, and **no PyTorch needed**. Same accuracy (same weights, converted).
- **Learn:** *quantisation*. `compute_type: int8` stores weights as 8-bit integers instead of
  32-bit floats: 4× smaller, much faster on CPU, with negligible accuracy loss.

### Decision 2: Run on CPU by default, GPU when possible
- `transcribe.py` asks CTranslate2 whether a CUDA GPU is usable. If the GPU run fails (on Windows
  it usually fails because the cuBLAS/cuDNN DLLs are missing), it **automatically retries on CPU**.
- Enabling the GPU later = installing NVIDIA's cuBLAS + cuDNN runtime (~1.5 GB). Postponed until disk is freed.
- **Expected speed:** model `small` on CPU ≈ 0.25–0.6× real time, i.e. **15–40 min per 1-hour call**.
  For 60–80 calls, that's an overnight batch job. Fine, because every stage resumes where it stopped.

### Decision 3: Store audio as 16 kHz mono FLAC, delete raw downloads
- Whisper and pyannote both work at 16 kHz mono, so we convert once and every stage reads the same file.
- FLAC is lossless (no compression artefacts that could distort pitch) and ~½ the size of WAV.
  80 one-hour calls ≈ 4 GB as FLAC vs ≈ 9 GB as WAV.

### Step 1.2: Project skeleton
- `src/` layout with one module per pipeline stage (see ARCHITECTURE.md §2 for the principles).
- `config/settings.yaml` holds every tunable number; `config.py` is the only file that reads it.
- `data/manifest/calls.csv` is the **registry of calls**. Every stage is keyed on `call_id`
  (`HDFCBANK_FY25Q2`). The date and time are stored now because the event study (Phase 5) needs them:
  - **Learn:** *look-ahead bias*. NSE closes at 15:30 IST. If a call happens at 16:00, the market can
    only react the **next** trading day. Using the same day's return would credit the call for
    price moves that happened *before* it.
- `companies.csv`: 18 candidate companies (6 banks, 6 IT, 6 other) for the sample.

### Step 1.3: `audio.py`
fetch (URL with streaming download + progress bar, or local path) → ffmpeg → `clean/<id>.flac`.
Downloads go to a `.part` file first and are renamed only when complete.

### Step 1.4: `transcribe.py`
- `word_timestamps=True`: every word gets a start/end time. Needed for pauses now and for
  matching words to speakers in Phase 2.
- `vad_filter=True`: **Voice Activity Detection** (Silero VAD, a small ONNX model) skips hold
  music and silence; faster, and prevents Whisper "hallucinating" text during music.
- `condition_on_previous_text=False`: on long audio, Whisper can get stuck repeating a phrase
  when it is fed its own previous output; turning this off is the standard fix.
- `language="en"`: Indian concalls are in English; auto-detection sometimes guesses Hindi on accented speech.

### Step 1.5: `voice.py`: the acoustic features
| Feature | How | Why it might matter |
|---|---|---|
| Pitch (F0) median | YIN algorithm on 64 ms frames, 20 ms hop | baseline voice level, needed to normalise |
| Pitch variation `f0_std_st` | std of `12·log2(f0/median)` | flat, monotone delivery vs animated; changes under stress |
| Pitch range `f0_range_st` | p90 − p10 in semitones | robust to occasional pitch-tracking errors |
| Speaking rate | words / segment duration | |
| Articulation rate | words / (duration − pause time) | speed while actually talking |
| Pauses | gaps between consecutive words ≥ 0.25 s; "long" ≥ 1 s | hesitation |
| Loudness variation | std of RMS energy (dB) on voiced frames | |

- **Learn:** *why semitones?* Pitch perception is logarithmic. A 20 Hz swing is large for a
  100 Hz voice but small for a 250 Hz voice. Semitones measure variation relative to the
  speaker's own voice, so speakers can be compared.
- **Learn:** *YIN vs pYIN*. YIN is fast but always outputs a pitch, even for silence or
  "sss". We gate it: keep frames that are loud (within 25 dB of the segment's loudest) and not
  stuck at the search limits. pYIN (probabilistic YIN) decides voiced/unvoiced itself; it's more
  accurate but ~10× slower. Switchable via `voice.pitch_method`.
- Features are computed **per transcript segment**, not on the whole call at once: a 1-hour
  call in one go would need several GB of RAM. It also makes Phase 2 trivial: label each
  segment with a speaker, then group by speaker.

### Step 1.6: CLI (`python -m concall ...`)
`status`, `validate`, `prepare`, `transcribe`, `voice`, `run`, `summary`.
Every stage **skips finished calls** (resume after a crash or a laptop shutdown) and **continues
past failures** (one broken URL doesn't kill an overnight batch).

### Step 1.7: Tests (`tests/`)
Signals with a known answer, so we know the maths is right before trusting real data:
- a steady 150 Hz tone must give a median ≈ 150 Hz and ~0 variation (YIN and pYIN);
- a glide over exactly 12 semitones must give a range of ≈ 9.6 st (p90 − p10 of a linear ramp);
- silence must have no pitch;
- hand-made word timings with known pauses must give exact pause counts and rates;
- manifest: bad quarter formats, bad dates and duplicate IDs are rejected.

### Step 1.8: First end-to-end run, and three problems fixed
Test input: a 35 s fake "bank earnings call" made with Windows text-to-speech
(`data/samples/demo_call.wav`, manifest row `DEMO_FY25Q2`). This tests the full pipeline before we have real concall audio.

**Problem 1: background noise was counted as voice (caught by a test).**
YIN reported a pitch for 65 % of frames of pure noise. The loudness gate compared each frame with the
segment's loudest frame, and steady hiss is evenly loud, so it passed.
*Fix:* a **periodicity** check (`voice.periodicity`). Voiced speech repeats every pitch period, so the
normalised autocorrelation at a lag of one period is ~0.7–1.0 for voice and ~0 for noise. Frames below
`min_periodicity: 0.5` are dropped. The test now covers near-silence and loud hiss.
- **Learn:** this is why tests use signals with a *known* answer. On a real call the bug would
  have been invisible, and every pitch number slightly wrong.

**Problem 2: faster-whisper 1.2.1 vs PyAV 19 version clash.**
`open() got an unexpected keyword argument 'metadata_errors'`: faster-whisper decodes audio with PyAV
and passes an argument newer PyAV removed.
*Fix:* our FLAC is already 16 kHz mono, so `transcribe._load_samples` reads it with soundfile and
passes the raw numpy array to Whisper, skipping PyAV entirely.
- **Learn:** upstream breakages like this are common. Owning the input format (Decision 3) gave a clean workaround.

**Problem 3: `mkl_malloc: failed to allocate memory`.**
- The GPU attempt failed as predicted (`cublas64_12.dll` missing) and fell back to CPU, but the CUDA
  attempt had already reserved memory in the process, and the laptop had only ~2.9 GB of
  *commit memory* left (RAM + page file). The page file couldn't grow because C: was full.
- *Fix (now):* `transcription.device: cpu` in settings, so no CUDA attempt.
- **Learn:** *commit memory*. Windows only hands out memory it can back with RAM **or** page file.
  A full system drive shrinks that limit, so a full disk can cause "out of memory" errors.

**Result** (transcript, CPU, `small` model):
```
[ 7.76-14.68] Our net interest margin improved to 3.6%, and gross NPA declined to 1.3%.
[15.58-20.78] Deposits grew 15% year-on-year, with a cost ratio of 38%.     <- "CASA ratio" misheard
```
- Numbers, "NPA" and "net interest margin" are correct; **"CASA" → "cost"** is wrong. Finance jargon
  is Whisper's weak spot. *To do:* pass an `initial_prompt` listing domain terms (CASA, NIM, GNPA,
  PAT, EBITDA, bps…), which biases Whisper toward them.
- Voice summary: 74 words, 2.3 words/s, 15 pauses/min, median pitch ~90 Hz (TTS male voice),
  pitch variation 2.7 semitones. Plausible for a calm, scripted read.
- Speed: 35 s of audio in 31 s with healthy memory (≈ real time), but **317 s** when RAM was nearly
  full (Windows swapping to disk). Memory, not CPU, is the bottleneck on this laptop.

### Step 1.9: Moved the project from C: to D:  (2026-11-06)
C: had 231 MB free. Moved everything to **`D:\clduse\bankingpr\bnk`**:
| What | How |
|---|---|
| Code, config, data, docs | copied with `robocopy`, verified all 24 files, then deleted on C: |
| `.venv` | **recreated** on D:, not copied. A venv hard-codes its own absolute path, so a copied venv breaks |
| Whisper model (483 MB) | moved from `C:\Users\TANMAY\.cache\huggingface` to `bnk\models\hub\` |
| Future model downloads | `config.py` sets `HF_HOME=bnk\models`, so FinBERT/pyannote will also land on D: |
| pip installs | run with `TMP/TEMP=D:\tmp` and `--no-cache-dir`, since pip unpacks in %TEMP% (on C:) by default |

Verified: 10/10 tests pass on D:, and transcription runs with `HF_HUB_OFFLINE=1` (proves the model
is read from D:, not re-downloaded). C: free space: 231 MB → 1.5 GB.

### Phase 1 status
- [x] Manifest, audio normalisation, Whisper transcription with word timestamps, voice features, CLI, tests
- [x] Real concall audio in the manifest (Step 2.1)
- [x] Whisper `initial_prompt` with finance vocabulary (Step 2.0)
- [x] GPU (Step 2.0)

---

## Phases 2–6: the rest of the pipeline  (2026-11-06)

### Step 2.0: Heavy dependencies, and the GPU for free
Installed PyTorch **2.14 + CUDA 12.6** (~2.5 GB download, all on D:), then pyannote.audio 4.0,
transformers 5.22, anthropic 2.6, yfinance, statsmodels, FastAPI, yt-dlp. `pip check`: no conflicts.

**Decision 4: reuse PyTorch's CUDA libraries for Whisper.** In Phase 1, Whisper couldn't use the GPU
because `cublas64_12.dll` / `cudnn64_9.dll` were missing (NVIDIA's toolkit is ~3 GB). The PyTorch
CUDA wheel *already contains those DLLs* in `torch/lib`. `config.py` adds that folder to the DLL
search path before CTranslate2 loads. Result:

| | CPU (Phase 1) | GPU (now) |
|---|---|---|
| 35 s demo clip | 31 s (317 s with low memory) | **5.7 s** incl. model load |
| 56 min real ICICI call | ~1 hour (estimated) | **148 s** (~23× real time) |

- **Learn:** *DLL search path.* Windows looks for DLLs in the program folder, system folders and
  `PATH`. A library can be "installed" and still not found if it lives elsewhere.

**Whisper vocabulary prompt.** `initial_prompt` lists finance terms (CASA, NIM, GNPA, bps, crore...).
Whisper treats it as text that "came before", which biases spelling. The demo now reads
"**CASA ratio** of 38%" (it was "cost ratio").

### Step 2.1: Real data, from the companies themselves
- SEBI's listing rules (since 2021) require listed companies to publish earnings-call audio. ICICI Bank
  and HDFC Bank host them on their official YouTube channels and link them from their investor pages.
- Tried NSE's filing archive first: it blocks automated downloads. Spoofing a browser to get around
  that was refused (rightly; we don't bypass bot protection). Used the companies' own pages instead.
- Manifest: **13 ICICI Bank calls (FY24Q1–FY27Q1) + 2 HDFC Bank calls**. `audio.py` downloads
  YouTube audio with yt-dlp and saves the video's **title and duration** to `data/audio/meta/`.
- **Data-quality catch:** ICICI's website lists the *same* video for Q4 FY25 and Q1 FY26. A new manifest
  check rejects one recording used for two calls; the downloaded title ("Replay of Q4-2025 earnings
  call... April 19, 2025") proved the link belongs to Q4 FY25. FY26Q1 stays without audio until the
  correct link is found.
  - **Learn:** never trust a source's labels blindly; cheap automatic cross-checks catch real errors.

### Step 2.2: Phase 2: who is speaking (`diarize.py`, `roles.py`)
- **Diarization** (pyannote) = "who spoke when": anonymous SPEAKER_00, 01... from voiceprints
  (segmentation model → speaker embeddings → clustering). Needs a free Hugging Face token: **not
  set yet**, so the pipeline currently falls back to text rules (below). Audio is passed in memory
  to avoid pyannote's FFmpeg-DLL decoder on Windows.
- **Roles** come from the *structure* of an Indian concall: the operator says "the next question is from
  the line of X from Y"; whoever speaks next is an analyst; speakers in the prepared remarks are
  management. Each speaker's votes are saved to `<id>.roles.json` so any label can be audited.
- **Text-only fallback**, first real run: analysts got only **5%** of Q&A words. Fixed with
  1. operator boilerplate patterns from the real call ("as a reminder", "question queue", and
     "Southern 0", Whisper's version of "star 0");
  2. a small **state machine**: announcement → analyst; once the questions ("?") stop → management;
     a look-ahead of 2 segments keeps context sentences inside the question.
  Analyst share went to **24%** of Q&A words, which is plausible. Known limit: when Whisper merges the
  end of a question and the start of an answer into one segment, text rules can't split it. Only real
  diarization can.
- Voice features are recomputed **per utterance** (single speaker) so management's voice isn't mixed
  with analysts'.
- **Bug fixed:** Whisper tokenises "11.9%" as `11`, `.9`, `%`; joining with spaces gave "11 .9 %".
  `join_words()` re-glues punctuation and hyphens (and has a test).

### Step 2.3: Phase 3: text signals (`text.py`, `llm.py`)
- **FinBERT** (ProsusAI/finbert) per sentence → tone = P(pos) − P(neg). Runs on the GPU.
- **Hedging lexicon** (`config/lexicons/hedging.txt`), Loughran-McDonald style weak modals and
  uncertainty words plus concall phrases ("too early to say", "you know"), whole-word matching.
- **LLM (Claude Opus 5.5)**: one request per call with **structured output**. The response must
  validate against a Pydantic schema: guidance items (raised/maintained/lowered vs the *previous
  quarter's* extracted guidance, which is passed in), evasive answers (partial / non-answer),
  sector KPIs, YoY revenue and profit growth. Every item carries a **verbatim quote**.
  Server-side fallback is enabled, so a wrongly declined request retries on another model.
  Skipped until `ANTHROPIC_API_KEY` is in `.env`.
  - **Learn:** *structured outputs* turn an LLM from "text generator" into a typed function:
    the pipeline never parses free text.

### Step 2.4: Phase 4: one row per call (`tone.py`)
Feature names follow `<who>_<signal>_<section>` (e.g. `mgmt_hedges_per_1k_qa`), plus
`*_qa_minus_prep` contrasts and `d_*` quarter-over-quarter changes, **only between consecutive
quarters**, so a missing quarter never produces a fake two-quarter "change" (there's a test for it).

### Step 2.5: Phase 5: event study (`event.py`)
- **Day 0** rule implemented and tested: during-hours call → same day; after 15:30, weekend or
  unknown time → next trading day. ICICI calls are on Saturdays → day 0 = Monday.
- AR = stock − Nifty 50 (market-adjusted, as in the README); a **market model** (α + β·market,
  fitted on days −250..−30) is a config switch. A test shows why it matters: a high-beta stock looks
  "abnormal" under market-adjusted returns even with no news.
- Regressions M1 (YoY growth controls only) → M4 (+text, +voice, +LLM), HC3 robust errors,
  standardised regressors; Spearman correlations as the small-sample-friendly view.
- **Bug caught by a test, in the test itself:** I expected CAR = 2.000% for a 2% jump, but the stock
  also had the market's 0.1% daily drift on that day: 1.02 × 1.001 − 1 − 0.001 = **2.002%**. The
  code was right. Writing the expected value out as a formula makes the test explain itself.
- First real result: ICICI Q4 FY25 (call Sat 19 Apr 2025): day 0 = Mon 21 Apr; **CAR[0,0] = −0.93%**
  vs Nifty, CAR[0,4] = −0.92%, volume 1.34× normal.

### Step 2.6: Phase 6: API + Docker (`api.py`, `Dockerfile`)
`/health`, `/calls`, `/calls/{id}/report`, `POST /calls/{id}/process` (background task),
`/results/regressions`. The API reads the pipeline's files, so it holds no state. CPU Docker image;
`data/` and `models/` are mounted volumes.

### Step 2.7: Pipeline engineering: stale outputs
Re-running `roles` didn't refresh `text` and `voice_utt`, which still used the old roles.
`pipeline.is_stale()` now compares file timestamps like `make`: if any earlier stage's output is
newer than this stage's, it rebuilds automatically.

### First real call: ICICI Bank Q4 FY25 (text-only roles)
| Management signal | Prepared remarks | Q&A |
|---|---|---|
| Hedging words per 1,000 | **1.0** | **46.3** |
| Articulation rate (words/s) | 2.51 | 2.78 |
| Pauses per minute | 4.4 | 5.8 |
| Pitch variation (semitones) | 4.19 | 3.95 |

Management hedges ~46× more when answering live questions than when reading the script, which is
the scripted-vs-unscripted contrast the project is built to measure. FinBERT tone is mildly positive
(+0.11) in both sections. Its most "negative" sentences are factual declines ("declined by 0.4%
sequentially"), a known FinBERT trait: it scores words like "declined" even in neutral reporting.

### Step 2.8: Finishing the batch, and a RAM fix in `transcribe.py`  (2026-11-06 evening)
- The overnight batch died silently at 03:05 partway through a call (the laptop slept). There was
  no traceback; the log's file timestamp just stopped changing. Rerunning `concall run` picked up
  where it left off because finished stages are skipped.
- The three last ICICI calls then failed with `Unable to allocate 445 MiB / 895 MiB`. faster-whisper
  builds the log-mel spectrogram for the **whole** input before decoding, so a 50-minute call needs
  about 900 MB in one block. With only ~3 GB free, the 43-minute call succeeded once it ran alone,
  but the two 49-minute calls still failed.
- Fix: `chunk_bounds()` cuts the audio about every 10 min (`transcription.chunk_s: 600`), moving each
  cut to the quietest 0.5 s within ±15 s so no word is split. Each chunk is transcribed separately
  and timestamps are shifted by the chunk's start. Peak memory drops ~5×, speed is unchanged
  (49-min call: 135 s), and the segment count matches an unchunked call of the same length (566 vs 543).
  Tests: `tests/test_transcribe.py` (31 pass).

**Learn:** "works on one file" is not "works on the batch". Memory that is fine for one call can
fail on the longest one, or after several calls in one process. Cap peak memory by design (chunking)
rather than hoping the free RAM is enough. Also, a long run that stops with no error usually means
sleep/shutdown: check the log's last-write time.

### Batch results: 14 calls (12 ICICI Bank FY24Q1–FY27Q1, 2 HDFC Bank), text-only roles, no LLM
ICICIBANK_FY26Q1 has no audio (the bank's page links the wrong video). The market reaction is the
cumulative abnormal return (CAR, %) vs the benchmark index, starting on the first trading day after the call.

**Spearman correlations of call features with CAR** (`data/results/correlations.csv`):

| Feature | CAR day 0 | CAR days 0–4 | CAR days 0–29 |
|---|---|---|---|
| Management tone (FinBERT, whole call) | −0.16 | −0.16 | **−0.64** |
| Pauses per minute (mgmt, Q&A) | −0.35 | −0.36 | −0.25 |
| Pitch variation, Q&A minus prepared | −0.37 | −0.44 | −0.30 |
| Hedges per 1k words | +0.15 | +0.16 | +0.29 |

**Regressions** (OLS, HC3 SEs, standardized regressors, `data/results/regressions.txt`):
- Only one coefficient has p < 0.05: tone → 30-day CAR, −2.4 pp per 1 SD (p = 0.043, N = 13,
  adj. R² 0.38). After the voice features are added it is no longer significant (p = 0.16).
- Short-window models (day 0, days 0–4) explain nothing reliable (adj. R² ≤ 0.01).
- M4 (LLM) is identical to M3 because LLM features are absent until `ANTHROPIC_API_KEY` is set.

**How to read this honestly:** N = 13–14, almost all one bank, and 12 models × several features were
tested. A single p = 0.04 is about what chance alone produces. The *sign* is interesting: more positive
tone is followed by worse 30-day returns. That could fit the "cheap talk / overly upbeat management"
story, but it is not evidence of it. The pipeline works end to end. Getting a real signal needs
(1) many more calls (more banks + IT firms; aim for 100+), (2) diarization for clean management-vs-
analyst roles, and (3) the LLM features (guidance changes, evasiveness).

### Step 2.9: Diarization switched on, sample widened to 31 calls  (2026-11-07)
- `HF_TOKEN` added (free). The first diarization run failed with `WinError 1455: the paging file is too
  small`. Windows' *commit* memory (RAM + page file) was nearly exhausted by other apps (an Android
  emulator alone held 5.9 GB). After closing them, pyannote ran on the GPU: ~155 s per call, ~9
  speakers found on a typical ICICI call. Claude Code also stopped a background batch once because
  the system ran low on memory. Diarization now runs one call per process.
- `ANTHROPIC_API_KEY`: skipped (paid). The LLM features stay optional; a free local model is a possible
  later swap in `llm.py`.
- **New sources**, all official company pages:
  - **Axis Bank:** 14 calls, FY24Q1–FY27Q2. The shareholders-corner page shows one year at a time; other years come
    from the same `GetFilteredData` request the page makes when you change the year dropdown.
  - **HDFC Bank:** +2 (FY24Q1, FY25Q1). The IR page only carries some quarters' replays.
  - **Not added:** Infosys (its site refused the request; not worked around), and Kotak (only FY27
    MP3s follow a findable URL pattern, and an official call date wasn't confirmed). Both are easy
    to add later if links are found by hand.
- Manifest: **31 calls** (13 ICICI, 4 HDFC, 14 Axis), `concall validate` OK.

**Learn:** "Out of memory" on Windows can mean *commit limit*, not RAM. Programs reserve memory that
must be backed by RAM + page file even if it's never used (CUDA DLLs reserve a lot). Free disk space
doesn't help unless the page file is allowed to grow.

### Results with diarization: same 14 calls, roles from pyannote speakers
| Spearman ρ with CAR | text-only roles (0 / 0–4 / 0–29) | **diarized roles** (0 / 0–4 / 0–29) |
|---|---|---|
| Mgmt tone (whole call) | −0.16 / −0.16 / −0.64 | −0.05 / −0.03 / **−0.61** |
| Pauses per min (mgmt Q&A) | −0.35 / −0.36 / −0.25 | −0.08 / −0.01 / **+0.34** |
| Pitch var., Q&A − prepared | −0.37 / −0.44 / −0.30 | −0.20 / −0.31 / −0.40 |
| Hedges per 1k | +0.15 / +0.16 / +0.29 | +0.28 / +0.33 / +0.29 |

- The tone → 30-day CAR regression is **no longer significant** (p = 0.043 → 0.615). No coefficient in
  any model has p < 0.05 now. The rank correlation of tone with 30-day CAR stays around −0.6, but the
  regression's robust SEs are wide at N = 13.
- The pause feature's sign **flipped**. This is what happens when "who is management" changes: text cues
  mislabelled some analyst speech as management (and vice versa), and voice features are very
  sensitive to that.

**Learn:** a result that changes sign when you fix a data-quality step was never a result. Better
speaker labels made the voice features *different*, not *stronger*. That's the strongest argument
yet for a bigger sample.

### Step 2.10: 6 more calls (20 total), a GPU-memory fix, and a company-effect trap  (2026-11-07)
- **Scope cut:** processing all 16 new calls (~10 min each, plus stalls) was too slow, so only 6 were
  run: HDFC FY24Q1, FY25Q1 and Axis **FY24Q1, FY25Q1, FY26Q1, FY27Q1**. The Axis calls were picked
  as one per year, same quarter, **chosen by date, not by how the stock moved**. Picking the "big move"
  calls would bias the event study. The other 10 Axis calls stay in the manifest (`--call` to add later).
- **Bug: `CUDA error: out of memory` after diarization.** On the 6 GB RTX 3050, PyTorch's caching allocator
  kept pyannote's GPU memory reserved after `diarize()` returned, so FinBERT, loading next in the
  same process, ran out. Fix in `diarize.py`: free the pipeline, `gc.collect()`,
  `torch.cuda.empty_cache()` in a `finally`. Rerunning the failed call resumed at the failed stage
  (transcript and diarization were already saved) and succeeded.
- **Stalls:** the laptop went into Modern Standby twice (Kernel-Power events 506 → 507), freezing
  the run for 35 and 84 min; Claude Code also once stopped the batch while the system was low on
  memory. Finished stages are never redone, so nothing was lost.
- Axis calls get 14–19 pyannote speakers (ICICI ~9–11): one person is often split into several
  labels. The role split still looks sane (Axis FY24Q1: mgmt 24 min prepared / 19 min Q&A, analysts 8).

**Spearman ρ with CAR, 14 calls → 20 calls** (`data/results_14calls_diarized/` vs `data/results/`):

| Feature | 14 calls (0 / 0–4 / 0–29) | 20 calls (0 / 0–4 / 0–29) |
|---|---|---|
| Mgmt tone (whole call) | −0.05 / −0.03 / −0.61 | **−0.46** / −0.35 / **−0.59** |
| Hedges per 1k | +0.28 / +0.33 / +0.29 | **+0.55** / **+0.54** / +0.47 |
| Pauses per min (mgmt Q&A) | −0.08 / −0.01 / +0.34 | −0.35 / −0.42 / −0.30 |
| Pitch var., Q&A − prepared | −0.20 / −0.31 / −0.40 | −0.32 / −0.40 / −0.58 |

Regressions (N = 20, HC3): hedges → day-0 CAR is +1.43 pp per SD, **p = 0.043** (M2); once pauses are
added (M3) nothing has p < 0.05. The quarter-over-quarter (`d_*`) features didn't change, because
the new calls are a year apart, so they add no consecutive-quarter pairs. M1's R² = 0: the YoY
growth controls are still empty for these calls.

**But this is mostly a company effect.** Per-bank means:

| Bank (N) | Hedges / 1k | Tone | CAR day 0 | CAR 0–29 |
|---|---|---|---|---|
| ICICI (12) | 21.8 | 0.12 | +0.88% | +0.11% |
| HDFC (4) | 14.5 | 0.22 | +0.05% | −2.66% |
| Axis (4) | 9.3 | 0.25 | **−3.98%** | −4.45% |

Axis management hedges least and sounds most positive, and these 4 Axis calls happened to be
followed by bad days. After **demeaning within each bank**, the correlations collapse: hedges vs day-0 CAR
+0.55 → −0.07, tone vs day-0 CAR −0.46 → +0.11, tone vs 30-day CAR −0.59 → −0.21.

**Learn:** adding a new company adds a new *level*, not just more data points. If one company has
both a different speaking style and a different set of returns, a pooled correlation measures the
company, not the call. The study needs company fixed effects (or within-company demeaning)
before any pooled ρ or p-value means anything. With 3 banks and 4–12 calls each, that leaves very
little power, so more calls *per company* matter more than more companies.

### Step 2.11: Company fixed effects  (2026-11-07)
`event.py` now fits every model twice, **pooled** and with **company fixed effects** (one intercept
per bank), and writes `correlations_within.csv` (Spearman after subtracting each bank's mean) next
to the pooled `correlations.csv`. `regressions.csv` has a `company_fe` column. Details:
- `fit_model()` drops a company with fewer than 3 calls from FE fits. A 1-call company is fitted
  exactly by its own dummy (leverage 1), and HC3 divides by (1 − leverage), so it would blow up.
- Signals stay z-scored on the pooled sample, so coefficients keep the same units (%-points of CAR per SD).
- Tests (`tests/test_event.py`, 35 pass in total) build a synthetic "company trap": company B hedges
  more *and* has higher returns, but within each company more hedging means lower returns. The
  pooled fit gets the sign wrong and the FE fit recovers the true −0.5 slope.

**Results (N = 20, 3 banks):**

| Model, CAR day 0 | adj. R² pooled | adj. R² with company FE |
|---|---|---|
| Company dummies only | n/a | **0.41** |
| + tone + hedges (M2) | 0.21 | 0.34 |
| + pitch + pauses (M3) | 0.21 | 0.27 |

- **Which bank it is explains more than anything the call says.** Adding the text and voice features
  to the bank dummies *lowers* adjusted R² (0.41 → 0.34 → 0.27, likewise for days 0–4).
- The one "significant" pooled result, hedges → day-0 CAR (+1.43 pp/SD, p = 0.043), drops to
  +0.23 pp/SD, **p = 0.79** with FE. No FE coefficient in any model has p < 0.2; the
  lowest is pitch variation → 30-day CAR (−1.1 pp/SD, p = 0.22).
- Within-bank Spearman ρ are all |ρ| ≤ 0.30. The largest are pauses (−0.23 / −0.28 for day 0 / days 0–4)
  and pitch-contrast (−0.22 / −0.26 / −0.30), which have the same sign as before but are weak.

**Honest conclusion so far:** with 20 calls from 3 banks, there is **no evidence** that tone, hedging
or voice features predict the stock reaction once you compare a bank only with itself. That is a
real (null) finding, not a failure. The design can now tell a company effect from a call effect,
and the next gains come from more calls *per bank*: within-bank comparisons on 4 Axis calls have
almost no power.

<!-- NEXT: YoY growth controls (LLM is off: needs another source, e.g. a hand-filled CSV from the
     banks' results press releases); process remaining 10 Axis calls for within-bank power -->





