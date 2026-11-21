"""Render settings: EEVEE, colour management, bloom compositor, video/still output."""

from __future__ import annotations

from pathlib import Path

import bpy


def setup_compositor(scene, bloom_strength: float = 0.5, threshold: float = 1.0) -> None:
    g = bpy.data.node_groups.new("vocelle_comp", "CompositorNodeTree")
    g.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = g.nodes.new("CompositorNodeRLayers")
    glare = g.nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Quality"].default_value = "High"
    glare.inputs["Threshold"].default_value = threshold
    glare.inputs["Strength"].default_value = bloom_strength
    out = g.nodes.new("NodeGroupOutput")
    g.links.new(rl.outputs["Image"], glare.inputs["Image"])
    g.links.new(glare.outputs["Image"], out.inputs[0])
    scene.compositing_node_group = g


def setup_render(scene, fps: float, res: str, quality: str) -> None:
    w, h = (int(v) for v in res.lower().split("x"))
    r = scene.render
    r.engine = "BLENDER_EEVEE"
    r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
    r.fps = int(round(fps))
    r.fps_base = r.fps / fps
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5
    ee = scene.eevee
    ee.taa_render_samples = 16 if quality == "preview" else 64
    ee.use_raytracing = quality == "final"
    ee.use_shadows = True
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    setup_compositor(scene)


def add_audio(scene, audio: Path, frame_start: int = 1) -> None:
    ed = scene.sequence_editor or scene.sequence_editor_create()
    ed.strips.new_sound("music", str(audio), 1, frame_start)


def output_video(scene, out: Path) -> Path:
    s = scene.render.image_settings
    s.media_type = "VIDEO"
    s.file_format = "FFMPEG"
    f = scene.render.ffmpeg
    f.format = "MPEG4"
    f.codec = "H264"
    f.constant_rate_factor = "HIGH"
    f.ffmpeg_preset = "GOOD"
    f.audio_codec = "AAC"
    f.audio_bitrate = 256
    f.audio_mixrate = 48000
    f.audio_channels = "STEREO"
    path = out.with_suffix(".mp4")
    scene.render.filepath = str(path)
    return path


def output_still(scene, out: Path) -> None:
    s = scene.render.image_settings
    s.media_type = "IMAGE"
    s.file_format = "PNG"
    scene.render.filepath = str(out)
