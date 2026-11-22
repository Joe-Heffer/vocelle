"""Entry point executed by Blender:  blender -b --python run.py -- --features F --scene S ...

Builds the scene from scratch (factory startup), keyframes the music, then renders a
video or a set of stills.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import bpy  # noqa: E402

from vocelle import features as feat  # noqa: E402
from vocelle.blender import drive, render  # noqa: E402


def parse() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scene", default="harmonic_field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--quality", default="final")
    ap.add_argument("--frame-start", type=int, help="absolute first Blender frame (overrides --start)")
    ap.add_argument("--frame-end", type=int, help="absolute last Blender frame, inclusive (overrides --duration)")
    ap.add_argument("--no-audio", action="store_true", help="video only (parallel chunk workers)")
    ap.add_argument("--samples", type=int, help="override EEVEE render samples")
    ap.add_argument("--threads", type=int, help="CPU threads for this process (0 = auto)")
    ap.add_argument("--still", type=float, nargs="+")
    ap.add_argument("--save-blend", action="store_true")
    return ap.parse_args(argv)


def find_audio(features_path: Path, track: str) -> Path | None:
    for cand in (features_path.parent.parent / "audio" / track, Path("audio") / track):
        if cand.exists():
            return cand.resolve()
    return None


def main() -> None:
    a = parse()
    t0 = time.time()
    features_path = Path(a.features)
    data = feat.load(features_path)
    fps = data["fps"]

    # Start from an empty scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render.setup_render(scene, fps, a.res, a.quality, samples=a.samples, threads=a.threads)

    ctl = drive.build_control(data)
    hist, chroma = drive.build_history(data)
    scene_mod = importlib.import_module(f"vocelle.scenes.{a.scene}")
    scene_mod.build(scene, data, ctl=ctl, history=hist, chroma=chroma, quality=a.quality)

    first = drive.time_to_frame(a.start, fps)
    last = drive.FRAME_OFFSET + data["n_frames"] - 1
    if a.duration is not None:
        last = min(last, first + int(round(a.duration * fps)) - 1)
    if a.frame_start is not None:
        first = a.frame_start
    if a.frame_end is not None:
        last = min(a.frame_end, drive.FRAME_OFFSET + data["n_frames"] - 1)
    scene.frame_start, scene.frame_end = first, last

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))
    print(f"[vocelle] scene built in {time.time() - t0:.1f}s; frames {first}-{last}")

    if a.still:
        for t in a.still:
            scene.frame_set(drive.time_to_frame(t, fps))
            render.output_still(scene, Path(f"{out}-{t:07.2f}s.png"))
            bpy.ops.render.render(write_still=True)
            print(f"[vocelle] wrote {scene.render.filepath}")
        return

    audio = None if a.no_audio else (Path(a.audio) if a.audio else find_audio(features_path.resolve(), data["track"]))
    if a.no_audio:
        pass
    elif audio:
        render.add_audio(scene, audio, drive.FRAME_OFFSET)
    else:
        print("[vocelle] WARNING: audio not found, rendering silent video (pass --audio)")
    path = render.output_video(scene, out)
    bpy.ops.render.render(animation=True)
    print(f"[vocelle] wrote {path} in {time.time() - t0:.1f}s")


main()
