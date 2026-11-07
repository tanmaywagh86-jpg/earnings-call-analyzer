"""Command-line entry point:  python -m concall <command> [--call CALL_ID ...] [--force]

Per-call commands (each skips calls whose output exists, unless --force):
  status       table of which stages are finished for each call
  validate     check the manifest for mistakes
  run          every stage below, call by call
  prepare | transcribe | voice | diarize | roles | voice_utt | text | llm    one stage only

Cross-call commands:
  summary      Phase 1 voice summary, one row per call -> data/features/voice_calls.csv
  study        Phases 4-5: feature table, QoQ changes, event study, regressions -> data/results/
  serve        Phase 6: start the API at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

import pandas as pd

from . import manifest, pipeline, voice
from .config import load_settings

log = logging.getLogger("concall")


def _calls(args) -> pd.DataFrame:
    df = manifest.load_manifest()
    if args.call:
        missing = set(args.call) - set(df.index)
        if missing:
            sys.exit(f"not in manifest: {sorted(missing)}")
        df = df.loc[args.call]
    return pipeline.ordered(df)


def cmd_status(args):
    df = _calls(args)
    if df.empty:
        print("manifest is empty: add rows to data/manifest/calls.csv")
        return []
    table = pd.DataFrame([{"call_id": c, **pipeline.status(c)} for c in df.index])
    print(table.replace({True: "yes", False: "-"}).to_string(index=False))
    return []


def cmd_validate(args):
    print(f"manifest OK: {len(_calls(args))} calls")
    return []


def _per_call(stages: list[str] | None):
    def run(args) -> list[str]:
        failed = []
        for cid in _calls(args).index:
            try:
                pipeline.run_call(cid, force=args.force, stages=stages)
            except Exception as e:  # noqa: BLE001 - one failing call shouldn't stop the batch
                log.error("%s failed: %s", cid, e)
                log.debug(traceback.format_exc())
                failed.append(cid)
        return failed
    return run


def cmd_summary(args):
    rows = []
    for cid, row in _calls(args).iterrows():
        p = voice.segment_features_path(cid)
        if p.exists():
            rows.append({"call_id": cid, "nse_symbol": row["nse_symbol"], "fiscal_quarter": row["fiscal_quarter"],
                         **voice.summarise_call(pd.read_parquet(p))})
    if not rows:
        print("no voice features yet: run `python -m concall run` first")
        return []
    out = load_settings().path("features") / "voice_calls.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nwritten: {out}")
    return []


def cmd_study(args):
    info = pipeline.run_study(_calls(args))
    print(info)
    print(f"results in {load_settings().path('results')}")
    return []


def cmd_serve(args):
    import uvicorn
    uvicorn.run("concall.api:app", host="127.0.0.1", port=8000)
    return []


COMMANDS = {
    "status": cmd_status, "validate": cmd_validate, "run": _per_call(None),
    **{s: _per_call([s]) for s in pipeline.STAGES},
    "summary": cmd_summary, "study": cmd_study, "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="concall", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--call", nargs="+", help="only these call_ids (default: all in manifest)")
    p.add_argument("--force", action="store_true", help="redo work even if outputs exist")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    failed = COMMANDS[args.command](args)
    if failed:
        log.error("%d call(s) failed: %s", len(failed), failed)
        return 1
    return 0
