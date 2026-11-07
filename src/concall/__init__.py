"""Earnings Call Voice & Sentiment Analyzer.

Package layout (one module per pipeline stage):
    config      - loads config/settings.yaml, resolves paths
    manifest    - the registry of calls (data/manifest/calls.csv)
    audio       - fetch + normalise audio to 16 kHz mono FLAC
    transcribe  - Whisper (faster-whisper) -> transcript JSON with word timestamps
    voice       - pitch / speech-rate / pause features per transcript segment
    cli         - `python -m concall <command>` entry point
"""

__version__ = "0.1.0"
