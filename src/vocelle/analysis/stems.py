"""Source separation with Demucs (`pip install vocelle[stems]`)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def separate(track: Path, out_root: Path, model: str = "htdemucs") -> Path | None:
    """Return a directory of stem WAVs (drums, bass, vocals, other[, guitar, piano]).

    Cached: if the folder already exists, Demucs isn't re-run. Returns None if Demucs
    isn't installed.
    """
    stem_dir = out_root / model / track.stem
    if stem_dir.exists() and any(stem_dir.glob("*.wav")):
        return stem_dir
    try:
        import demucs  # noqa: F401
    except ImportError:
        log.warning("demucs not installed (uv sync --extra stems); analysing the mix only")
        return None
    log.info("separating stems with %s ...", model)
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", str(out_root), str(track)], check=True)
    return stem_dir
