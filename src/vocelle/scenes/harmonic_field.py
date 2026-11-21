"""Harmonic Field — demo scene.

A disc of hexagonal glass-metal columns on a reflective floor.

* **Travelling waves:** each column reads the music from `distance * DELAY` frames ago
  (Sample Index into the history mesh), so every kick and bass swell radiates out from
  the centre like a ripple, and chord colours spread outward on every change.
* **Circle of fifths:** the disc is split into 12 angular sectors ordered by fifths;
  each sector swells and glows with the strength of its pitch class (chroma), so you
  can *see* the harmony — a C major chord lights C, E and G sectors.
* **Colour = harmony:** chord root on the circle of fifths sets the hue.
* **Melody ribbon:** a glowing tube trails the camera-orbit path; its height follows the
  melody pitch, its thickness the voicing, so the melody draws a line through time.
* **Editing:** camera framing is chosen per *section label*, so repeated sections (a
  returning chorus) get the same shot — a visual leitmotif. Cuts land on downbeats,
  the camera dollies in through each section and pulls back as energy builds before a
  drop; lens punches on downbeats.
"""

from __future__ import annotations

import math

import bpy
import numpy as np

from vocelle.blender import drive
from vocelle.blender.nodes import Tree, out

CELL = 0.55          # hex radius (m)
FIELD_RADIUS = 20.0  # m
DELAY = 1.1          # frames of delay per metre from the centre
RIBBON_POINTS = 400
RIBBON_STEP = 1.0    # frames between ribbon points
RIBBON_RADIUS = 11.0

CAMERA_SHOTS = [  # distance, height, lens(mm), look-at height, orbit offset (rad)
    (30.0, 11.0, 30.0, 0.5, 0.0),    # wide establishing orbit
    (25.0, 4.5, 30.0, 2.0, 1.9),     # low, grazing across the column tops
    (36.0, 30.0, 40.0, 0.0, 3.6),    # high, looking down on the circle of fifths
    (9.0, 17.0, 20.0, 1.0, 5.0),     # steep overhead close-up
    (27.0, 2.6, 45.0, 3.0, 0.8),     # long lens, near floor level
]


def pick(curves: dict, *names: str) -> str:
    return next(n for n in names if n in curves)


# --------------------------------------------------------------------------- materials
def column_material() -> bpy.types.Material:
    m = bpy.data.materials.new("column")
    t = Tree(m.node_tree)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.015, 0.015, 0.02, 1)
    bsdf.inputs["Metallic"].default_value = 0.35
    bsdf.inputs["Roughness"].default_value = 0.35
    col = t.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="vcol")
    glow = t.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="glow")
    top = t.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="top")
    mask = t.madd(out(top, "Fac"), 0.97, 0.03)
    t.link(out(col, "Color"), bsdf.inputs["Emission Color"])
    t.link(t.mul(t.mul(out(glow, "Fac"), mask), 3.0), bsdf.inputs["Emission Strength"])
    return m


def ribbon_material() -> bpy.types.Material:
    m = bpy.data.materials.new("ribbon")
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    t = Tree(nt)
    col = t.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="vcol")
    light = t.vmath("ADD", out(col, "Vector"), (0.25, 0.25, 0.25))
    em = t.node("ShaderNodeEmission", {"Color": light, "Strength": 4.0})
    t.link(em, nt.nodes["Material Output"].inputs["Surface"])
    return m


def floor_material() -> bpy.types.Material:
    m = bpy.data.materials.new("floor")
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.005, 0.005, 0.006, 1)
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.12
    return m


# --------------------------------------------------------------------------- geometry
def hex_points(radius: float, cell: float) -> np.ndarray:
    a1 = np.array([1.5 * cell, math.sqrt(3) / 2 * cell])
    a2 = np.array([0.0, math.sqrt(3) * cell])
    k = int(radius / cell) + 2
    pts = [i * a1 + j * a2 for i in range(-k, k + 1) for j in range(-2 * k, 2 * k + 1)]
    pts = np.array([p for p in pts if np.hypot(*p) <= radius])
    return np.c_[pts, np.zeros(len(pts))]


def point_object(name: str, pts: np.ndarray) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(pts))
    mesh.vertices.foreach_set("co", pts.astype(np.float32).ravel())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def gn_modifier(obj, name: str) -> Tree:
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    obj.modifiers.new(name, "NODES").node_group = ng
    t = Tree(ng)
    t.gin = t.node("NodeGroupInput")
    t.gout = t.node("NodeGroupOutput")
    return t


def sampler(t: Tree, source_obj):
    geo = out(t.node("GeometryNodeObjectInfo", {"Object": source_obj}, transform_space="ORIGINAL"), "Geometry")

    def sample(attr: str, index, dtype: str = "FLOAT"):
        na = t.node("GeometryNodeInputNamedAttribute", {"Name": attr}, data_type=dtype)
        si = t.node("GeometryNodeSampleIndex", data_type=dtype, domain="POINT", clamp=True)
        t.link(geo, si.inputs["Geometry"])
        t.link(out(na, "Attribute"), si.inputs["Value"])
        t.set(si.inputs["Index"], index)
        return out(si, "Value")

    return sample


def build_field(curves: dict, hist, chroma) -> bpy.types.Object:
    obj = point_object("field", hex_points(FIELD_RADIUS, CELL))
    t = gn_modifier(obj, "harmonic_field")
    sample = sampler(t, hist)
    sample_chroma = sampler(t, chroma)

    pos = t.node("GeometryNodeInputPosition")
    dist = out(t.vmath("LENGTH", pos), "Value")
    frame = out(t.node("GeometryNodeInputSceneTime"), "Frame")
    idx = t.math("FLOOR", t.math("SUBTRACT", t.add(frame, -drive.FRAME_OFFSET), t.mul(dist, DELAY)))

    low = sample(pick(curves, "bass_rms", "mix_low"), idx)
    hit = sample(pick(curves, "drums_onset", "mix_onset"), idx)
    downbeat = sample("downbeat_pulse", idx)
    energy = sample("energy", idx)
    r, g, b = (sample(f"chord_{c}", idx) for c in "rgb")

    # Sector on the circle of fifths -> pitch class -> chroma strength.
    sp = t.sep(pos)
    ang = t.math("ARCTAN2", out(sp, "Y"), out(sp, "X"))
    sector = t.math("FLOOR", t.madd(ang, 12 / (2 * math.pi), 6.0))
    pc = t.math("FLOORED_MODULO", t.mul(sector, 7.0), 12.0)
    cval = sample_chroma("chroma", t.madd(t.math("MAXIMUM", idx, 0.0), 12.0, pc))
    c3 = t.math("POWER", cval, 3.0)

    rnd = out(t.node("FunctionNodeRandomValue", {"Min": 0.0, "Max": 1.0, "ID": t.node("GeometryNodeInputIndex")},
                     data_type="FLOAT"), "Value")
    falloff = t.math("EXPONENT", t.mul(dist, -0.08))

    h = t.add(0.2, t.mul(t.math("POWER", low, 2.0), 2.4))
    h = t.add(h, t.mul(c3, t.madd(energy, 1.6, 0.6)))
    h = t.add(h, t.mul(t.mul(hit, falloff), 1.4))
    h = t.add(h, t.mul(t.mul(rnd, energy), 0.5))

    glow = t.add(0.08, t.mul(hit, 2.0))
    glow = t.add(glow, t.mul(t.math("POWER", cval, 2.0), 1.5))
    glow = t.add(glow, t.mul(downbeat, 0.8))

    # Column: hex prism with base on z=0 and a "top" face attribute for the glowing cap.
    cyl = t.node("GeometryNodeMeshCylinder", {"Vertices": 6, "Radius": CELL * 0.9, "Depth": 1.0})
    moved = t.node("GeometryNodeTransform", {"Geometry": out(cyl, "Mesh"), "Translation": (0, 0, 0.5)})
    top = t.node("GeometryNodeStoreNamedAttribute", {"Geometry": moved, "Selection": out(cyl, "Top"),
                                                     "Name": "top", "Value": 1.0}, data_type="FLOAT", domain="FACE")
    mat = t.node("GeometryNodeSetMaterial", {"Geometry": top, "Material": column_material()})

    inst = t.node("GeometryNodeInstanceOnPoints", {"Points": out(t.gin, 0), "Instance": mat,
                                                   "Scale": t.xyz(1.0, 1.0, h)})
    s1 = t.node("GeometryNodeStoreNamedAttribute", {"Geometry": inst, "Name": "vcol", "Value": t.xyz(r, g, b)},
                data_type="FLOAT_VECTOR", domain="INSTANCE")
    s2 = t.node("GeometryNodeStoreNamedAttribute", {"Geometry": s1, "Name": "glow", "Value": glow},
                data_type="FLOAT", domain="INSTANCE")
    t.link(s2, t.gout.inputs[0])
    return obj


def build_ribbon(curves: dict, hist) -> bpy.types.Object:
    obj = point_object("melody_ribbon", np.zeros((0, 3)))
    t = gn_modifier(obj, "melody_ribbon")
    sample = sampler(t, hist)
    line = t.node("GeometryNodeCurvePrimitiveLine", {"Start": (0, 0, 0), "End": (1, 0, 0)})
    res = t.node("GeometryNodeResampleCurve", {"Curve": line, "Count": RIBBON_POINTS})
    i = t.node("GeometryNodeInputIndex")
    frame = out(t.node("GeometryNodeInputSceneTime"), "Frame")
    idx = t.math("FLOOR", t.math("SUBTRACT", t.add(frame, -drive.FRAME_OFFSET), t.mul(i, RIBBON_STEP)))
    ang = sample("ribbon_angle", idx)
    pitch = sample("melody_pitch", idx)
    voiced = sample("melody_voiced", idx)
    note = sample("melody_note_on", idx)
    age = t.math("DIVIDE", i, float(RIBBON_POINTS))
    rad = t.madd(t.math("SINE", t.mul(ang, 0.7)), 1.5, RIBBON_RADIUS)
    pos = t.xyz(t.mul(t.math("COSINE", ang), rad), t.mul(t.math("SINE", ang), rad), t.madd(pitch, 6.0, 2.0))
    setp = t.node("GeometryNodeSetPosition", {"Geometry": res, "Position": pos})
    taper = t.math("POWER", t.math("SUBTRACT", 1.0, age), 0.6)
    thick = t.mul(t.madd(voiced, 0.11, 0.015), t.mul(taper, t.madd(note, 0.8, 1.0)))
    col = t.node("GeometryNodeStoreNamedAttribute",
                 {"Geometry": setp, "Name": "vcol",
                  "Value": t.xyz(*(sample(f"chord_{c}", idx) for c in "rgb"))},
                 data_type="FLOAT_VECTOR", domain="POINT")
    prof = t.node("GeometryNodeCurvePrimitiveCircle", {"Resolution": 10, "Radius": 1.0})
    mesh = t.node("GeometryNodeCurveToMesh", {"Curve": col, "Profile Curve": prof, "Scale": thick, "Fill Caps": True})
    mat = t.node("GeometryNodeSetMaterial", {"Geometry": mesh, "Material": ribbon_material()})
    t.link(mat, t.gout.inputs[0])
    return obj


# --------------------------------------------------------------------------- camera
def build_camera(scene, data: dict) -> None:
    c = data["curves"]
    fps, n = data["fps"], data["n_frames"]
    energy = np.asarray(c["energy"])
    build = np.asarray(c["energy_build"])
    sec_phase = np.asarray(c["section_phase"])
    label = np.asarray(c["section_label"]).astype(int)
    downbeat = np.asarray(c["downbeat_pulse"])
    hit = np.asarray(c[pick(c, "drums_onset", "mix_onset")])

    shots = np.array([CAMERA_SHOTS[k % len(CAMERA_SHOTS)] for k in label])
    dist = shots[:, 0] * (1 - 0.18 * sec_phase) * (1 + 0.35 * build)
    height = shots[:, 1] * (1 + 0.25 * build)
    orbit = np.cumsum((0.06 + 0.45 * energy) / fps) + shots[:, 4]
    rng = np.random.default_rng(7)
    shake = rng.normal(0, 1, (n, 3)) * (0.12 * hit)[:, None]
    loc = np.c_[dist * np.cos(orbit), dist * np.sin(orbit), height] + shake

    target = bpy.data.objects.new("look_at", None)
    scene.collection.objects.link(target)
    drive.keyframe_array(target, "location", np.c_[np.zeros(n), np.zeros(n), shots[:, 3]])

    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    drive.keyframe_array(cam, "location", loc)
    drive.keyframe_array(cam_data, "lens", shots[:, 2] * (1 - 0.07 * downbeat))
    con = cam.constraints.new("TRACK_TO")
    con.target, con.track_axis, con.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 2.8
    cam_data.clip_end = 500


# --------------------------------------------------------------------------- scene
def build(scene, data: dict, *, ctl, history, chroma, quality: str = "final") -> None:
    c = data["curves"]
    fps = data["fps"]
    energy = np.asarray(c["energy"])
    ribbon_angle = np.cumsum((0.25 + 0.9 * energy) / fps).astype(np.float32)
    history.data.attributes.new("ribbon_angle", "FLOAT", "POINT").data.foreach_set("value", ribbon_angle)

    build_field(c, history, chroma)
    build_ribbon(c, history)

    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, -0.01))
    bpy.context.object.data.materials.append(floor_material())

    # Overhead light that flashes on downbeats (driver on the control object).
    light_data = bpy.data.lights.new("key", "AREA")
    light_data.size = 30
    light_data.color = (0.85, 0.9, 1.0)
    light = bpy.data.objects.new("key", light_data)
    light.location = (0, 0, 25)
    scene.collection.objects.link(light)
    drive.add_driver(light_data, "energy", "150 + 1500*d + 400*e", d="downbeat_pulse", e="energy")

    # World: near black, tinted by the current chord, brighter with energy.
    world = bpy.data.worlds.new("world")
    scene.world = world
    bg = world.node_tree.nodes["Background"]
    for i, ch in enumerate("rgb"):
        drive.add_driver(world.node_tree, 'nodes["Background"].inputs[0].default_value',
                         "0.002 + 0.03*c*(0.3+e)", index=i, c=f"chord_{ch}", e="energy")
    bg.inputs["Strength"].default_value = 1.0

    build_camera(scene, data)
