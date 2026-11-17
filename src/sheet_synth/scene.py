"""Builds the room in Blender and photographs the sheet: surface, lights, what casts shadows, and the render.

The sheet's shape and the camera come from geometry.py, so the truth is placed by the same arithmetic the mesh is
built from. Everything here is made in code: no texture, model or lighting file is loaded. README.md, "The scene".
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .geometry import FRAME_MM, Camera, Sheet, stacked

# What the sheet lies on. Planks and the music stand are what my real test photos show most; a line finder trained without them drew staves in a floor's seams.
SURFACES = ["planks", "planks", "planks", "wood", "wood", "stand", "stand", "plain", "plain", "cloth", "dark", "stone", "paper"]
PLANKS = [(0.42, 0.36, 0.30), (0.55, 0.40, 0.24), (0.62, 0.50, 0.36), (0.30, 0.20, 0.13), (0.70, 0.66, 0.60), (0.48, 0.30, 0.18)]   # greyed oak, honey, pale, walnut, whitewashed, red
LIGHTINGS = ["window", "window", "ceiling", "ceiling", "lamp", "soft", "mixed"]
WARM, COOL = (1.0, 0.78, 0.55), (0.82, 0.9, 1.0)


def draw_room(rng: random.Random) -> dict:
    """The room at random: what the sheet lies on, how it is lit, how glossy the paper is, and what stands in the light."""
    lighting = rng.choice(LIGHTINGS)
    warmth = rng.random()
    return {
        "surface": rng.choice(SURFACES),
        "surface_seed": rng.getrandbits(16),
        "surface_colour": [round(rng.random(), 3) for _ in range(3)],
        "lighting": lighting,
        "light_from_deg": round(rng.uniform(0, 360), 1),
        "light_height_deg": round(rng.uniform(20, 75), 1),
        "light_colour": [round(a + (b - a) * warmth, 3) for a, b in zip(COOL, WARM)],
        "light_soft": round(rng.uniform(0.0, 1.0), 3),
        "fill": round(rng.choice([0.05, 0.15, 0.3, 0.6]), 2),          # how much light comes from everywhere
        "gloss": round(rng.choice([0.0, 0.0, 0.0, 0.3, 0.7]), 2),      # a coated page shines back at a lamp
        "holder": rng.random() < 0.75,                                 # the phone and the person holding it stand in the light
        "window_bars": lighting == "window" and rng.random() < 0.5,
        "second_light_deg": round(rng.uniform(0, 360), 1),
        # The sheets under this one: where each lies from the one above it, in metres along and across the page, and
        # how far it is turned, in degrees. A part is one of a stack more often than it lies alone.
        "stack": [[round(rng.gauss(0, 0.0025), 4), round(rng.gauss(0, 0.0025), 4), round(rng.gauss(0, 0.25), 2)] for _ in range(rng.choice([0, 0, 0, 0, 1, 2, 3, 4, 6, 9]))],
        "saturation": round(rng.random() ** 2, 3),                     # of a plain or cloth surface: most are near grey
    }


def reset() -> None:
    """An empty scene. Blender keeps every block until told otherwise, so a long run must clear them."""
    for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for block in list(collection):
            collection.remove(block)


def material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree, bpy.types.Node]:
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    tree = made.node_tree
    return made, tree, tree.nodes["Principled BSDF"]


def add_object(name: str, mesh: bpy.types.Mesh, look: bpy.types.Material | None = None) -> bpy.types.Object:
    made = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(made)
    if look:
        mesh.materials.append(look)
    return made


def box(name: str, centre, size, look=None, turn=None, seen: bool = True) -> bpy.types.Object:
    """A box, its axes turned to the columns of `turn`."""
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * np.array(size)
    corners = corners @ np.asarray(turn).T if turn is not None else corners
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata((corners + np.array(centre)).tolist(), [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    made = add_object(name, mesh, look)
    made.visible_camera = seen
    return made


def sheet_object(sheet: Sheet, texture: np.ndarray, gloss: float, columns: int = 60, rows: int = 80) -> bpy.types.Object:
    """The sheet as a mesh, with the printed page on it. `texture` is height by width by 3, 0 to 1, as the eye sees it."""
    u, v = np.meshgrid(np.linspace(0, 1, columns + 1), np.linspace(0, 1, rows + 1))
    points = sheet.points(u, v).reshape(-1, 3)
    faces = [(r * (columns + 1) + c, r * (columns + 1) + c + 1, (r + 1) * (columns + 1) + c + 1, (r + 1) * (columns + 1) + c) for r in range(rows) for c in range(columns)]
    mesh = bpy.data.meshes.new("sheet")
    mesh.from_pydata(points.tolist(), [], faces)
    layer = mesh.uv_layers.new(name="page")
    flat_u, flat_v = u.reshape(-1), 1 - v.reshape(-1)
    for loop in mesh.loops:
        layer.data[loop.index].uv = (flat_u[loop.vertex_index], flat_v[loop.vertex_index])
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    height, width = texture.shape[:2]
    image = bpy.data.images.new("page", width, height, alpha=False, float_buffer=True, is_data=True)
    linear = np.where(texture <= 0.04045, texture / 12.92, ((texture + 0.055) / 1.055) ** 2.4).astype(np.float32)   # the renderer works in light, not in screen values
    image.pixels.foreach_set(np.concatenate([linear[::-1], np.ones((height, width, 1), np.float32)], axis=2).reshape(-1))
    look, tree, shader = material("paper")
    picture = tree.nodes.new("ShaderNodeTexImage")
    picture.image = image
    picture.extension = "EXTEND"
    tree.links.new(picture.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.85 - 0.6 * gloss
    shader.inputs["Specular IOR Level"].default_value = 0.2 + 0.5 * gloss
    return add_object("sheet", mesh, look)


def under_sheets(sheet: Sheet, room: dict) -> float:
    """The stack under the sheet: each sheet bent as the top one is, a little out of place, so its edges show.
    Returns how far below the top sheet the stack reaches, in metres, so the surface can be put under it."""
    layers, depth = stacked(sheet, room.get("stack", []))
    rng = random.Random(room["surface_seed"] + 1)
    for number, points in enumerate(layers):
        rows, columns = points.shape[0] - 1, points.shape[1] - 1
        faces = [(r * (columns + 1) + c, r * (columns + 1) + c + 1, (r + 1) * (columns + 1) + c + 1, (r + 1) * (columns + 1) + c) for r in range(rows) for c in range(columns)]
        look, _, shader = material(f"under{number}")
        shade = rng.uniform(0.82, 0.97)
        shader.inputs["Base Color"].default_value = (shade, shade, shade * rng.uniform(0.94, 1.0), 1)
        shader.inputs["Roughness"].default_value = 0.85
        mesh = bpy.data.meshes.new(f"under{number}")
        mesh.from_pydata(points.reshape(-1, 3).tolist(), [], faces)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        add_object(f"under{number}", mesh, look)
    return depth


def stand_object(sheet: Sheet, room: dict, below: float) -> None:
    """A music stand's desk behind the sheet and the ledge the sheet stands on: dark, matt, a little wider than a
    page. The camera looks at it as it looks down at a table, so nothing else in the scene changes."""
    rng = random.Random(room["surface_seed"] + 2)
    look, _, shader = material("stand")
    shade = rng.uniform(0.01, 0.06)
    shader.inputs["Base Color"].default_value = (shade, shade, shade * rng.uniform(0.9, 1.2), 1)
    shader.inputs["Roughness"].default_value = rng.uniform(0.35, 0.8)
    wide, tall = rng.uniform(0.44, 0.56), rng.uniform(0.3, 0.36)
    foot = -sheet.height / 2 - rng.uniform(0.0, 0.004)                 # the sheet's lower edge rests on the ledge
    box("desk", (rng.gauss(0, 0.02), foot + tall / 2, -below - 0.004), (wide, tall, 0.006), look)
    box("ledge", (0, foot - 0.006, 0.012 - below), (wide, 0.012, 0.036), look)
    if rng.random() < 0.5:   # a lip in front of the page's foot, which hides a few millimetres of its lower margin
        box("lip", (0, foot + rng.uniform(0.002, 0.005), 0.032 - below), (wide, 0.003, 0.012), look)


def surface_object(room: dict, size: float = 3.0, below: float = 0.0) -> bpy.types.Object:
    """What the sheet lies on: a big slab with a pattern made from noise, its top `below` the sheet's lowest point."""
    rng = random.Random(room["surface_seed"])
    look, tree, shader = material("surface")
    nodes, links = tree.nodes, tree.links
    where = nodes.new("ShaderNodeTexCoord")
    placed = nodes.new("ShaderNodeMapping")
    placed.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 3.14))
    links.new(where.outputs["Object"], placed.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    kind = room["surface"]
    r, g, b = room["surface_colour"]
    if kind == "stand":
        kind = "plain"   # the room behind the stand; the stand itself is added apart
    grey = (r + g + b) / 3
    r, g, b = (grey + (c - grey) * room.get("saturation", 1.0) for c in (r, g, b)) if kind in ("plain", "cloth") else (r, g, b)
    if kind == "planks":
        # A floor or a table of boards: each a shade of its own, grain along it, a dark seam between.
        boards = nodes.new("ShaderNodeTexBrick")
        base = np.array(rng.choice(PLANKS)) ** 2.2 * rng.uniform(0.7, 1.25)   # the list is as the eye sees them
        boards.inputs["Color1"].default_value = (*base, 1)
        boards.inputs["Color2"].default_value = (*(base * rng.uniform(0.72, 0.95)), 1)
        boards.inputs["Mortar"].default_value = (*(base * rng.uniform(0.08, 0.4)), 1)
        boards.inputs["Scale"].default_value = 1.0
        boards.inputs["Mortar Size"].default_value = rng.uniform(0.0004, 0.002)
        boards.inputs["Mortar Smooth"].default_value = rng.uniform(0.0, 0.3)
        boards.inputs["Bias"].default_value = rng.uniform(-0.3, 0.3)
        boards.inputs["Brick Width"].default_value = rng.uniform(0.5, 1.8)
        boards.inputs["Row Height"].default_value = rng.uniform(0.06, 0.2)
        boards.offset, boards.offset_frequency = rng.uniform(0.2, 0.8), rng.choice([2, 3, 5])
        links.new(placed.outputs["Vector"], boards.inputs["Vector"])
        stretched = nodes.new("ShaderNodeMapping")
        stretched.inputs["Scale"].default_value = (rng.uniform(0.8, 2.5), rng.uniform(25, 70), 1)
        links.new(placed.outputs["Vector"], stretched.inputs["Vector"])
        grain = nodes.new("ShaderNodeTexNoise")
        grain.inputs["Scale"].default_value = rng.uniform(2, 6)
        grain.inputs["Detail"].default_value = 9
        grain.inputs["Distortion"].default_value = rng.uniform(0.2, 1.5)
        links.new(stretched.outputs["Vector"], grain.inputs["Vector"])
        links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.3, 0.72
        dark = rng.uniform(0.55, 0.85)
        ramp.color_ramp.elements[0].color, ramp.color_ramp.elements[1].color = (dark, dark, dark, 1), (1, 1, 1, 1)
        both = nodes.new("ShaderNodeMix")
        both.data_type, both.blend_type = "RGBA", "MULTIPLY"
        both.inputs["Factor"].default_value = 1.0
        links.new(boards.outputs["Color"], both.inputs["A"])
        links.new(ramp.outputs["Color"], both.inputs["B"])
        links.new(both.outputs["Result"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = rng.uniform(0.3, 0.7)
        return box("surface", (0, 0, -0.01 - below), (size, size, 0.02), look)
    if kind == "wood":
        # Wood grain: noise stretched along the board.
        grain = nodes.new("ShaderNodeTexNoise")
        grain.inputs["Scale"].default_value = rng.uniform(2.5, 7)
        grain.inputs["Detail"].default_value = 7
        grain.inputs["Distortion"].default_value = rng.uniform(0.3, 1.8)
        placed.inputs["Scale"].default_value = (1, rng.uniform(14, 40), 1)
        links.new(placed.outputs["Vector"], grain.inputs["Vector"])
        links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.3, 0.7
        base = np.array([0.18 + 0.4 * r, 0.09 + 0.22 * r, 0.03 + 0.1 * r])
        colours = (base * rng.uniform(0.45, 0.8), base)
        roughness = rng.uniform(0.25, 0.6)
    else:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = {"plain": rng.uniform(2, 20), "cloth": rng.uniform(300, 900), "dark": rng.uniform(5, 60), "stone": rng.uniform(8, 40), "paper": rng.uniform(100, 400)}[kind]
        noise.inputs["Detail"].default_value = 8
        links.new(placed.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        if kind == "dark":
            base, contrast, roughness = np.array([0.02 + 0.05 * r, 0.02 + 0.05 * g, 0.02 + 0.06 * b]), 0.7, rng.uniform(0.2, 0.6)
        elif kind == "paper":
            base, contrast, roughness = np.array([0.8, 0.8, 0.78]) + 0.1 * r, 0.96, 0.85
        elif kind == "stone":
            base, contrast, roughness = np.array([0.3 + 0.5 * r] * 3) * np.array([1, 0.97, 0.92]), 0.55, rng.uniform(0.1, 0.5)
        elif kind == "cloth":
            base, contrast, roughness = np.array([r, g, b]) * 0.8 + 0.05, 0.75, 1.0
        else:
            base, contrast, roughness = np.array([r, g, b]) * 0.7 + 0.2 * rng.random(), 0.9, rng.uniform(0.3, 0.8)
        colours = (base * contrast, base)
    ramp.color_ramp.elements[0].color = (*colours[0], 1)
    ramp.color_ramp.elements[1].color = (*colours[1], 1)
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = roughness
    return box("surface", (0, 0, -0.01 - below), (size, size, 0.02), look)


def add_light(kind: str, location, colour, energy: float, size: float = 0.1, aim=(0, 0, 0)) -> None:
    light = bpy.data.lights.new(kind.lower(), kind)
    light.color, light.energy = colour, energy
    if kind == "SUN":
        light.angle = size
    elif kind == "AREA":
        light.size = size
    else:
        light.shadow_soft_size = size
    made = bpy.data.objects.new(kind.lower(), light)
    made.location = location
    made.rotation_euler = Vector(np.subtract(aim, location).tolist()).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(made)


def light_room(room: dict, camera: Camera) -> None:
    """The lights, and the things that stand in them."""
    around, height = math.radians(room["light_from_deg"]), math.radians(room["light_height_deg"])
    toward = np.array([math.cos(height) * math.cos(around), math.cos(height) * math.sin(around), math.sin(height)])
    colour, soft, kind = tuple(room["light_colour"]), room["light_soft"], room["lighting"]
    world = bpy.context.scene.world or bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    sky = world.node_tree.nodes["Background"]
    sky.inputs["Color"].default_value = (*(COOL if kind == "window" else colour), 1)
    sky.inputs["Strength"].default_value = room["fill"] * (1.5 if kind == "soft" else 1.0)
    if kind == "window":
        add_light("SUN", toward * 3, (1.0, 0.96, 0.9), 4.0, size=math.radians(0.5 + 6 * soft))
        # The wall the window is in: light comes only through the opening, and the bars cross it.
        middle = toward * 1.6
        side = np.cross(toward, [0, 0, 1.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, toward)
        turn = np.stack([side, up, toward], axis=1)
        wide, tall = 0.25 + 0.5 * soft, 0.3 + 0.6 * room["fill"]
        shift = (room["second_light_deg"] / 360 - 0.5) * 0.5
        for dx, dy, w, h in ((-1.5 - wide / 2 + shift, 0, 3, 6), (1.5 + wide / 2 + shift, 0, 3, 6), (shift, 1.5 + tall / 2, wide, 3), (shift, -1.5 - tall / 2, wide, 3)):
            box("wall", middle + side * dx + up * dy, (w, h, 0.01), turn=turn, seen=False)
        if room["window_bars"]:
            for step in (-0.25, 0.0, 0.25):
                box("bar", middle + side * (shift + step * wide), (0.012, tall, 0.012), turn=turn, seen=False)
            box("bar", middle + side * shift, (wide, 0.012, 0.012), turn=turn, seen=False)
    elif kind == "ceiling":
        add_light("AREA", (toward[0] * 0.6, toward[1] * 0.6, 1.6), colour, 220, size=0.2 + 1.0 * soft)
    elif kind == "lamp":
        low = np.array([math.cos(around), math.sin(around), 0]) * 0.45 + np.array([0, 0, 0.18 + 0.3 * soft])
        add_light("POINT", low, colour, 14, size=0.02 + 0.05 * soft)
    elif kind == "soft":
        add_light("AREA", toward * 1.5, colour, 150, size=1.5)
    else:   # daylight from one side and a warm lamp from another
        add_light("AREA", toward * 1.8, COOL, 260, size=0.8)
        other = math.radians(room["second_light_deg"])
        add_light("POINT", (math.cos(other) * 0.8, math.sin(other) * 0.8, 0.9), WARM, 40, size=0.05)
    if room["holder"]:
        # The phone, the hands under it and the person behind it: unseen by the camera, but they cast shadows.
        axes = camera.axes()
        right, up, back = axes[:, 0], axes[:, 1], axes[:, 2]
        at = np.array(camera.location)
        box("phone", at - up * 0.055 + back * 0.006, (0.072, 0.148, 0.008), turn=axes, seen=False)
        box("hands", at - up * 0.15 + back * 0.03, (0.16, 0.1, 0.05), turn=axes, seen=False)
        box("person", at - up * 0.45 + back * 0.25, (0.42, 0.5, 0.25), turn=axes, seen=False)


def camera_object(camera: Camera, focus_error: float = 0.0, blur: bool = True) -> None:
    data = bpy.data.cameras.new("camera")
    data.sensor_fit, data.sensor_width, data.lens = "AUTO", FRAME_MM, camera.lens_mm
    data.clip_start, data.clip_end = 0.01, 20
    if blur:
        # f/1.78 on the phone's small sensor blurs like about f/6.5 on this 36 mm frame.
        data.dof.use_dof = True
        data.dof.aperture_fstop = 6.5
        data.dof.focus_distance = float(np.linalg.norm(np.subtract(camera.target, camera.location))) * (1 + focus_error)
    made = bpy.data.objects.new("camera", data)
    made.matrix_world = Matrix(np.block([[camera.axes(), np.array(camera.location)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist())
    scene = bpy.context.scene
    scene.collection.objects.link(made)
    scene.camera = made
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = camera.width, camera.height, 100


_configured = False


def configure_once(samples: int) -> None:
    """Sets the renderer up the first time it is needed. SHEET_SYNTH_DEVICE picks the device: CPU, OPTIX or CUDA."""
    global _configured
    if not _configured:
        configure(samples, os.environ.get("SHEET_SYNTH_DEVICE", "CPU"))
        _configured = True


def configure(samples: int = 32, device: str = "CPU") -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    if device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = device   # "OPTIX" or "CUDA"
        preferences.get_devices()
        for found in preferences.devices:
            found.use = found.type == device
        scene.cycles.device = "GPU"
        # Denoising on the CPU took 5 of a render's 6.4 seconds on an RTX 4090; on the GPU the whole render takes 0.9.
        scene.cycles.denoising_use_gpu = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    # Off. Kept loaded between examples, the GPU renderer appears to reuse an earlier sheet's picture: in a check of 36
    # examples, homr read the same 124 notes from seven different photos. Not yet confirmed by looking at those photos.
    scene.render.use_persistent_data = False


def photograph(sheet: Sheet, camera: Camera, room: dict, texture: np.ndarray, work: Path, focus_error: float = 0.0, blur: bool = True) -> np.ndarray:
    """Renders the scene. Returns the light that reached the sensor: height by width by 3, linear, not yet a photo."""
    reset()
    sheet_object(sheet, texture, room["gloss"])
    below = under_sheets(sheet, room)
    if room["surface"] == "stand":
        stand_object(sheet, room, below)
        surface_object(room, below=below + 0.4)   # the room behind the stand, out of focus
    else:
        surface_object(room, below=below)
    light_room(room, camera)
    camera_object(camera, focus_error, blur)
    scene = bpy.context.scene
    scene.render.filepath = str(work / "render.exr")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath)
    width, height = image.size
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1, :, :3].copy()
