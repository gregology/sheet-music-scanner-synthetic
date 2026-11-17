import random
from pathlib import Path

import numpy as np

from sheet_synth.geometry import Camera, Sheet, page_grid, to_picture


# Paper that stretched as it bent would put notes where no real page could, and teach a line straightener nonsense.
def test_a_bent_sheet_keeps_its_lengths():
    for seed in range(30):
        sheet = Sheet.draw(random.Random(seed), 0.2159, 0.2794)
        sheet.ripple = 0.0
        u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 80))
        points = sheet.points(u, v)
        across = np.linalg.norm(np.diff(points, axis=1), axis=2)
        down = np.linalg.norm(np.diff(points, axis=0), axis=2)
        assert np.allclose(across, 0.2159 / 59, rtol=0.02), (seed, sheet.kind)
        assert np.allclose(down, 0.2794 / 79, rtol=0.02), (seed, sheet.kind)


# The truth is placed by our arithmetic and the picture is drawn by Blender. If the two disagree about the camera
# or the sheet's shape, every position in the dataset is wrong. This renders dots and looks for them.
def test_dots_on_the_page_appear_in_the_photo_where_the_truth_says(tmp_path: Path):
    from sheet_synth import scene

    width, height = 300, 400
    scene.configure_once(8)
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        # Different dots each time: a renderer that kept the last sheet's picture would show them in the old places.
        places = [(rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)) for _ in range(5)]
        texture = np.full((880, 680, 3), 0.02, np.float32)
        for u, v in places:
            texture[int(v * 880) - 9:int(v * 880) + 9, int(u * 680) - 9:int(u * 680) + 9] = 1.0
        sheet = Sheet.draw(rng, 0.2159, 0.2794)
        view, _ = Camera.draw(rng, sheet, width, height)
        room = scene.draw_room(rng) | {"lighting": "soft", "holder": False, "gloss": 0.0, "fill": 0.6, "surface": "dark", "surface_colour": [0, 0, 0]}
        light = scene.photograph(sheet, view, room, texture, tmp_path, blur=False).mean(axis=2)
        expected = to_picture(page_grid(sheet, view), np.array([p[0] for p in places]), np.array([p[1] for p in places]))
        for x, y in expected:
            if not (12 <= x < width - 12 and 12 <= y < height - 12):
                continue
            window = light[int(y) - 10:int(y) + 11, int(x) - 10:int(x) + 11]
            ys, xs = np.mgrid[int(y) - 10:int(y) + 11, int(x) - 10:int(x) + 11]
            weight = np.clip(window - window.max() * 0.5, 0, None)
            assert window.max() > 4 * np.median(light), (seed, sheet.kind, (x, y))   # a dot is there at all
            found = np.array([(xs * weight).sum(), (ys * weight).sum()]) / weight.sum()
            assert np.linalg.norm(found - np.array([x, y])) < 2.0, (seed, sheet.kind, (x, y), found)


# A sheet of the stack that cuts through the top sheet covers its music with blank paper, and the truth still lists
# the notes under it: a trial render lost a third of a page that way. Every sheet must lie under the one above it,
# wherever both are.
def test_no_sheet_of_a_stack_rises_through_the_sheet_above():
    from sheet_synth.geometry import stacked

    def height_at(above: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The height of the lattice `above` over the room points (x, y), and whether it reaches each."""
        u, v = 0.5 + x / 0.2159, 0.5 - y / 0.2794                       # where the point would be on a flat sheet, to start from
        for _ in range(40):
            at = to_picture(above, u, v)
            u, v = u + 0.8 * (x - at[..., 0]) / 0.2159, v - 0.8 * (y - at[..., 1]) / 0.2794
        at = to_picture(above, u, v)
        reached = (u > 0) & (u < 1) & (v > 0) & (v < 1) & (np.hypot(x - at[..., 0], y - at[..., 1]) < 1e-5)
        return at[..., 2], reached

    kinds = set()
    for seed in range(40):
        rng = random.Random(seed)
        sheet = Sheet.draw(rng, 0.2159, 0.2794, [{"across": seed % 2 == 0, "at": 0.5}] if seed % 3 == 0 else None)
        kinds.add(sheet.kind)
        stack = [[rng.gauss(0, 0.0025), rng.gauss(0, 0.0025), rng.gauss(0, 0.25)] for _ in range(9)]
        u, v = np.meshgrid(np.linspace(0, 1, 61), np.linspace(0, 1, 81))
        above = sheet.points(u, v)
        layers, depth = stacked(sheet, stack)
        assert depth < 0.0007 * 9 + 1e-9                     # and the stack stays a stack: no gap wider than a few sheets
        for layer in layers:
            over, there = height_at(above, layer[..., 0], layer[..., 1])
            assert there.mean() > 0.9 and np.all(layer[..., 2][there] < over[there]), (seed, sheet.kind, float(np.max((layer[..., 2] - over)[there])))
            above = layer
    assert kinds == {"flat", "bow", "book", "wave", "corner"}
