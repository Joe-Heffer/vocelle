"""Command-line interface: `vocelle analyze` and `vocelle render`."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def cmd_analyze(a: argparse.Namespace) -> None:
    from .analysis.pipeline import analyze, write
    from .features import summary

    track = Path(a.track)
    feats = analyze(track, fps=a.fps, stems_dir=Path(a.stems) if a.stems else None,
                    separate_stems=not a.no_separate, model=a.model, use_beat_this=not a.no_beat_this,
                    n_sections=a.sections)
    out = Path(a.out) if a.out else Path("analysis") / f"{track.stem}.json"
    write(feats, out)
    print(summary(feats))
    print(f"wrote {out}")


def cmd_render(a: argparse.Namespace) -> None:
    blender = a.blender or os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        sys.exit("blender not found: install Blender 5.x or pass --blender / set $BLENDER")
    features = Path(a.features).resolve()
    out = Path(a.out or f"renders/{features.stem}-{a.scene}{'-still' if a.still is not None else ''}").resolve()
    args = [blender, "-b", "--factory-startup", "--enable-autoexec", "--python", str(PKG / "blender" / "run.py"), "--",
            "--features", str(features), "--scene", a.scene, "--out", str(out), "--res", a.res,
            "--start", str(a.start), "--quality", a.quality]
    if a.audio:
        args += ["--audio", str(Path(a.audio).resolve())]
    if a.duration is not None:
        args += ["--duration", str(a.duration)]
    if a.still is not None:
        args += ["--still", *[str(s) for s in a.still]]
    if a.save_blend:
        args += ["--save-blend"]
    print(" ".join(args))
    sys.exit(subprocess.run(args).returncode)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="vocelle", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="audio -> analysis/<track>.json feature file")
    an.add_argument("track")
    an.add_argument("--fps", type=float, default=30.0)
    an.add_argument("--out")
    an.add_argument("--stems", help="existing folder of stem WAVs (drums/bass/vocals/other...)")
    an.add_argument("--no-separate", action="store_true", help="don't run Demucs; analyse the mix only")
    an.add_argument("--model", default="htdemucs", help="Demucs model: htdemucs | htdemucs_ft | htdemucs_6s")
    an.add_argument("--no-beat-this", action="store_true", help="use librosa beat tracking even if Beat This! is installed")
    an.add_argument("--sections", type=int, help="force number of sections")
    an.set_defaults(func=cmd_analyze)

    rn = sub.add_parser("render", help="feature file -> Blender scene -> MP4 (or stills)")
    rn.add_argument("features")
    rn.add_argument("--scene", default="harmonic_field")
    rn.add_argument("--audio", help="audio to mux (defaults to audio/<track>)")
    rn.add_argument("--out", help="output path without extension")
    rn.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    rn.add_argument("--duration", type=float, help="seconds to render (default: to the end)")
    rn.add_argument("--res", default="1920x1080")
    rn.add_argument("--quality", choices=["preview", "final"], default="final")
    rn.add_argument("--still", type=float, nargs="+", help="render PNG stills at these times (s) instead of video")
    rn.add_argument("--save-blend", action="store_true", help="also save the built .blend for inspection")
    rn.add_argument("--blender")
    rn.set_defaults(func=cmd_render)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
