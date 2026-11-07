"""Load settings.yaml once and give every module the same view of paths, parameters and secrets."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# src/concall/config.py -> parents[2] is the project root (bnk/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"

# Keep every downloaded model (Whisper, FinBERT, pyannote) inside bnk/models/ instead of the default
# C:\Users\<you>\.cache\huggingface. Must be set before huggingface_hub is imported, which is why it
# runs when this module is imported (every other module imports config first).
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))

# Secrets (HF_TOKEN, ANTHROPIC_API_KEY) live in bnk/.env, which is git-ignored.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional
    pass


def _expose_torch_cuda_dlls() -> None:
    """Let CTranslate2 (Whisper) use the CUDA libraries that ship inside the PyTorch wheel.

    CTranslate2 needs cublas64_12.dll and cudnn64_9.dll. Installing NVIDIA's CUDA toolkit just for
    those is ~3 GB; the PyTorch CUDA wheel already contains them in torch/lib. Windows only searches
    PATH for DLLs, so we prepend that folder. We locate it without importing torch (slow, heavy).
    """
    if sys.platform != "win32":
        return
    for site in sys.path:
        lib = Path(site) / "torch" / "lib"
        if (lib / "cublas64_12.dll").exists():
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(lib))
            except (AttributeError, OSError):
                pass
            return


_expose_torch_cuda_dlls()


def secret(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    def path(self, key: str) -> Path:
        """Absolute path for an entry under `paths:`; the directory is created if missing."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
        return p

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))


@lru_cache(maxsize=None)
def load_settings(path: str | Path = DEFAULT_SETTINGS) -> Settings:
    with open(path, encoding="utf-8") as f:
        return Settings(yaml.safe_load(f))


def torch_device(requested: str = "auto") -> str:
    """'cuda' if requested/available, else 'cpu'. Imports torch lazily."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
