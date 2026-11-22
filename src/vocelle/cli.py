"""Command-line interface: `vocelle analyze` and `vocelle render`."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
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


FRAME_OFFSET = 1  # keep in sync with blender/drive.py (curve index i <-> Blender frame i + 1)


def split_frames(first: int, last: int, n: int) -> list[tuple[int, int]]:
    """Split inclusive frame range [first, last] into <= n contiguous, non-overlapping chunks."""
    total = last - first + 1
    n = max(1, min(n, total))
    base, extra = divmod(total, n)
    chunks, f = [], first
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append((f, f + size - 1))
        f += size
    return chunks


def _find_audio(features: Path, track: str) -> Path | None:
    for cand in (features.parent.parent / "audio" / track, Path("audio") / track):
        if cand.exists():
            return cand.resolve()
    return None


def _render_parallel(a: argparse.Namespace, base: list[str], features: Path, out: Path) -> int:
    from .features import load

    data = load(features)
    fps = data["fps"]
    first = int(round(a.start * fps)) + FRAME_OFFSET
    last = FRAME_OFFSET + data["n_frames"] - 1
    if a.duration is not None:
        last = min(last, first + int(round(a.duration * fps)) - 1)
    chunks = split_frames(first, last, a.workers)
    threads = max(1, (os.cpu_count() or 1) // len(chunks))
    tmp = out.parent / f".{out.name}-chunks"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"[vocelle] frames {first}-{last} in {len(chunks)} chunks, {threads} threads each")

    procs, files = [], []
    for i, (f0, f1) in enumerate(chunks):
        stem = tmp / f"chunk_{f0:06d}_{f1:06d}"
        files.append(stem.with_suffix(".mp4"))
        if files[-1].exists() and (tmp / f"{stem.name}.done").exists():
            print(f"[vocelle] chunk {i}: reusing {files[-1].name}")
            continue
        cmd = base + ["--out", str(stem), "--frame-start", str(f0), "--frame-end", str(f1),
                      "--no-audio", "--threads", str(threads)]
        procs.append((i, stem, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

    def pump(i: int, p: subprocess.Popen) -> None:
        for line in p.stdout:
            print(f"[w{i}] {line}", end="")

    pumps = [threading.Thread(target=pump, args=(i, p), daemon=True) for i, _, p in procs]
    for t in pumps:
        t.start()
    failed = False
    for i, stem, p in procs:
        if p.wait() != 0:
            print(f"[vocelle] chunk {i} failed (exit {p.returncode})")
            failed = True
        else:
            (tmp / f"{stem.name}.done").touch()
    for t in pumps:
        t.join()
    if failed:
        print(f"[vocelle] kept chunks in {tmp}; re-run the same command to resume")
        return 1

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"[vocelle] ffmpeg not found; chunks left in {tmp}")
        return 1
    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    audio = Path(a.audio).resolve() if a.audio else _find_audio(features, data["track"])
    dest = out.with_suffix(".mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio:
        cmd += ["-ss", f"{a.start:.6f}", "-i", str(audio), "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-shortest"]
    else:
        print("[vocelle] WARNING: audio not found, rendering silent video (pass --audio)")
        cmd += ["-c", "copy"]
    rc = subprocess.run(cmd + [str(dest)]).returncode
    if rc == 0:
        print(f"[vocelle] wrote {dest}")
        if not a.keep_chunks:
            shutil.rmtree(tmp, ignore_errors=True)
    return rc


def cmd_render(a: argparse.Namespace) -> None:
    blender = a.blender or os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        sys.exit("blender not found: install Blender 5.x or pass --blender / set $BLENDER")
    features = Path(a.features).resolve()
    out = Path(a.out or f"renders/{features.stem}-{a.scene}{'-still' if a.still is not None else ''}").resolve()
    args = [blender, "-b", "--factory-startup", "--enable-autoexec", "--python", str(PKG / "blender" / "run.py"), "--",
            "--features", str(features), "--scene", a.scene, "--res", a.res,
            "--start", str(a.start), "--quality", a.quality]
    if a.audio:
        args += ["--audio", str(Path(a.audio).resolve())]
    if a.duration is not None:
        args += ["--duration", str(a.duration)]
    if a.samples:
        args += ["--samples", str(a.samples)]
    if a.still is not None:
        args += ["--still", *[str(s) for s in a.still]]
    if a.save_blend:
        args += ["--save-blend"]
    if a.workers > 1 and a.still is None:
        sys.exit(_render_parallel(a, args, features, out))
    args += ["--out", str(out)]
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
    rn.add_argument("--workers", type=int, default=1,
                    help="parallel Blender processes, each rendering a slice of the frames (try 3; costs VRAM)")
    rn.add_argument("--samples", type=int, help="override EEVEE samples (default 64 final / 16 preview)")
    rn.add_argument("--keep-chunks", action="store_true", help="keep per-worker chunk files after stitching")
    rn.add_argument("--blender")
    rn.set_defaults(func=cmd_render)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
