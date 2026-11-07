"""Get each call's audio onto disk in one standard format.

Why normalise? Recordings arrive as mp3/m4a/wav at 44.1/48 kHz, stereo or mono. Whisper,
pyannote and our pitch code all want 16 kHz mono, so we convert once with ffmpeg and every
later stage reads the same file: data/audio/clean/<call_id>.flac.

FLAC rather than WAV: lossless (no artefacts that could distort pitch), about half the size.
One hour of 16 kHz mono is ~115 MB as WAV and ~50 MB as FLAC.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

import logging

from .config import PROJECT_ROOT, load_settings
from .manifest import youtube_id

log = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (concall-research-pipeline)"


def clean_audio_path(call_id: str) -> Path:
    return load_settings().path("audio") / f"{call_id}.flac"


def fetch_youtube(video_id: str, call_id: str) -> Path:
    """Download the audio track of a company-published YouTube recording with yt-dlp.

    Many Indian companies (ICICI Bank, HDFC Bank, ...) host the SEBI-mandated call recordings on
    their YouTube channels and link them from their investor pages. We also save the video's title
    and duration to data/audio/meta/<call_id>.json so a wrong link is easy to spot.
    """
    import json

    import yt_dlp

    raw_dir = load_settings().path("raw_audio")
    existing = list(raw_dir.glob(f"{call_id}.*"))
    if existing:
        return existing[0]
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / f"{call_id}.%(ext)s"),
        "quiet": True,
        "noprogress": True,  # the per-chunk progress bar floods batch logs
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    meta = {k: info.get(k) for k in ("id", "title", "duration", "upload_date", "channel", "webpage_url")}
    (load_settings().path("audio_meta") / f"{call_id}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log.info("%s: downloaded '%s' (%s s)", call_id, meta["title"], meta["duration"])
    return next(raw_dir.glob(f"{call_id}.*"))


def fetch(source: str, call_id: str) -> Path:
    """Download a URL (or locate a local file) and return the path of the raw file."""
    if not source:
        raise ValueError(f"{call_id}: audio_source is empty in the manifest")

    vid = youtube_id(source)
    if vid:
        return fetch_youtube(vid, call_id)

    if urlparse(source).scheme in ("http", "https"):
        suffix = Path(urlparse(source).path).suffix or ".audio"
        dest = load_settings().path("raw_audio") / f"{call_id}{suffix}"
        if dest.exists():
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(source, stream=True, timeout=60, headers={"User-Agent": _USER_AGENT}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=call_id) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
        return dest

    local = Path(source).expanduser()
    if not local.is_absolute():
        local = PROJECT_ROOT / local  # relative paths in the manifest are relative to bnk/
    if not local.exists():
        raise FileNotFoundError(f"{call_id}: audio file not found: {local}")
    return local


def normalise(src: Path, dest: Path, sample_rate: int) -> Path:
    """ffmpeg: decode anything -> 16 kHz, mono, FLAC."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vn",                      # drop any video stream (some calls are webcasts)
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # resample
        "-c:a", "flac", str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest


def prepare(call_id: str, source: str, force: bool = False) -> Path:
    """Fetch + normalise one call. Skips work if the clean file already exists."""
    cfg = load_settings().section("audio")
    dest = clean_audio_path(call_id)
    if dest.exists() and not force:
        return dest
    raw = fetch(source, call_id)
    normalise(raw, dest, cfg["sample_rate"])
    # Only delete files *we* downloaded, never a user's local original.
    if not cfg.get("keep_raw", False) and raw.parent == load_settings().path("raw_audio"):
        raw.unlink(missing_ok=True)
    return dest


def duration_seconds(path: Path) -> float:
    import soundfile as sf
    return sf.info(str(path)).duration
