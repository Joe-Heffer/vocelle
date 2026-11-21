"""Bridge feature curves into Blender.

Two complementary mechanisms:

1. **Control object** (`VOCELLE`): every curve becomes a keyframed custom property
   (`VOCELLE["mix_rms"]`). Use these from drivers, shader Attribute nodes or Python.
2. **History meshes** (`VOCELLE_history`, `VOCELLE_chroma`): a vertex per video frame
   with one float attribute per curve. Geometry Nodes can `Sample Index` these at
   *any* frame — so an instance can read the music from `d` frames ago, which turns
   1-D audio into travelling waves across space.

Frame convention: curve index i  <->  Blender frame i + 1  <->  time i / fps.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy_extras import anim_utils

CONTROL = "VOCELLE"
FRAME_OFFSET = 1
_INTERP = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def time_to_frame(t: float, fps: float) -> int:
    return int(round(t * fps)) + FRAME_OFFSET


def _fcurve(idblock, data_path: str, index: int = 0):
    ad = idblock.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return bag.fcurves.find(data_path, index=index)


def keyframe_array(idblock, data_path: str, values, index: int = -1, interpolation: str = "LINEAR",
                   frame_offset: int = FRAME_OFFSET) -> None:
    """Bulk-write one keyframe per frame (fast: foreach_set instead of keyframe_insert)."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    multi = values.shape[1] > 1
    for ch in range(values.shape[1]):
        idx = ch if multi else max(index, 0)
        idblock.keyframe_insert(data_path, index=idx if (multi or index >= 0) else -1, frame=frame_offset)
        fc = _fcurve(idblock, data_path, idx)
        fc.keyframe_points.clear()
        n = len(values)
        fc.keyframe_points.add(n)
        co = np.empty(n * 2, dtype=np.float32)
        co[0::2] = np.arange(n) + frame_offset
        co[1::2] = values[:, ch]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", np.full(n, _INTERP[interpolation], dtype=np.int32))
        fc.update()


def build_control(features: dict, names: list[str] | None = None) -> bpy.types.Object:
    obj = bpy.data.objects.new(CONTROL, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_render = True
    for name in names or features["curves"]:
        obj[name] = 0.0
        keyframe_array(obj, f'["{name}"]', features["curves"][name])
    return obj


def _vertex_object(name: str, n: int) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_render = True
    obj.hide_viewport = True
    return obj


def build_history(features: dict) -> tuple[bpy.types.Object, bpy.types.Object]:
    """History meshes for Geometry Nodes `Sample Index` lookups."""
    curves = features["curves"]
    n = features["n_frames"]
    hist = _vertex_object("VOCELLE_history", n)
    for name, vals in curves.items():
        attr = hist.data.attributes.new(name, "FLOAT", "POINT")
        attr.data.foreach_set("value", np.asarray(vals, dtype=np.float32)[:n])
    # Chroma packed as frame*12 + pitch_class, normalised per frame.
    chroma = np.stack([np.asarray(curves[f"chroma_{p:02d}"], np.float32)[:n] for p in range(12)], axis=1)
    chroma /= chroma.max(axis=1, keepdims=True) + 1e-6
    chist = _vertex_object("VOCELLE_chroma", n * 12)
    chist.data.attributes.new("chroma", "FLOAT", "POINT").data.foreach_set("value", chroma.ravel())
    return hist, chist


def add_driver(idblock, data_path: str, expression: str, index: int = -1, **vars_: str):
    """Driver reading control-object curves, e.g. add_driver(light, "energy", "100*k", k="drums_onset").

    Keep expressions to simple arithmetic/min/max/sin so Blender's fast "simple
    expression" evaluator handles them (no Python auto-exec needed).
    """
    ctl = bpy.data.objects[CONTROL]
    fc = idblock.driver_add(data_path, index) if index >= 0 else idblock.driver_add(data_path)
    drv = fc.driver
    drv.type = "SCRIPTED"
    for var_name, curve in vars_.items():
        v = drv.variables.new()
        v.name = var_name
        v.type = "SINGLE_PROP"
        v.targets[0].id = ctl
        v.targets[0].data_path = f'["{curve}"]'
    drv.expression = expression
    return fc
