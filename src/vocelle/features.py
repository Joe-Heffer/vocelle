"""Feature-file schema helpers (pure stdlib, so Blender's Python can import it too).

See docs/FEATURES.md for the full description.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    v = data.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {v}, expected {SCHEMA_VERSION}; re-run `vocelle analyze`")
    return data


def summary(data: dict) -> str:
    r = data["rhythm"]
    key = data["key"]
    lines = [
        f"{data['track']}: {data['duration']:.1f}s @ {data['fps']} fps ({data['n_frames']} frames)",
        f"tempo {r['tempo']} BPM, {r['beats_per_bar']}/4, {len(r['beats'])} beats [{r['source']}]",
        f"key {key['tonic']} {key['mode']} (r={key['confidence']})",
        f"chords: {' '.join(c['label'] for c in data['chords'][:24])}{' ...' if len(data['chords']) > 24 else ''}",
        f"sections: {' '.join(s['label'] for s in data['sections'])}",
        f"melody from {data['melody']['source']}: {len(data['melody']['notes'])} notes",
        f"stems: {', '.join(data['stems']) or 'none'}; {len(data['curves'])} curves",
    ]
    return "\n".join(lines)
