"""The sheet's shape and the camera's view of it, as arithmetic.

A point on the page is (u, v): fractions of its width and height from its top left corner. `Sheet.points` says where
that point is in the room, in metres, once the sheet is bent. `Camera.project` says which pixel it lands on. The
scene builds its mesh from the same `Sheet.points`, so the picture and the truth cannot disagree about the shape.

The sheet bends without stretching, as paper does: it curves in one direction only, and every straight line across
that direction stays straight (README.md, "The scene").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

# iPhone 16 main camera: 26 mm as a 35 mm equivalent. The frame's long side is 36 mm on that scale.
LENS_MM = 26.0
FRAME_MM = 36.0


@dataclass
class Sheet:
    """A bent sheet lying near the plane z = 0, its centre near the origin, its top edge toward +y."""

    width: float                  # metres
    height: float
    kind: str = "flat"
    axis: float = 0.0             # the direction that stays straight, radians from the page's width
    curvature: list[float] | None = None   # 1/metres, at evenly spaced stations across the bending direction
    ripple: float = 0.0           # metres: small bumps a flat sheet still has
    ripple_seed: int = 0

    def points(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Where page points are in the room: an array with the shape of `u` and a last axis of x, y, z."""
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        x, y = (u - 0.5) * self.width, (0.5 - v) * self.height
        straight = np.array([math.cos(self.axis), math.sin(self.axis)])
        across = np.array([-straight[1], straight[0]])
        t, s = x * straight[0] + y * straight[1], x * across[0] + y * across[1]
        reach = 0.5 * (abs(self.width * across[0]) + abs(self.height * across[1]))   # the page runs from -reach to +reach across the bend
        stations = np.linspace(-reach, reach, 801)
        step = stations[1] - stations[0]
        curvature = np.interp(stations, np.linspace(-reach, reach, len(self.curvature)), self.curvature) if self.curvature else np.zeros_like(stations)
        angle = np.cumsum(curvature) * step
        angle -= angle[len(angle) // 2]            # the middle of the page lies level
        along, up = np.cumsum(np.cos(angle)) * step, np.cumsum(np.sin(angle)) * step
        along, up = along - along[len(along) // 2], up - up.min()
        bent_s, z = np.interp(s, stations, along), np.interp(s, stations, up)
        if self.ripple:
            rng = np.random.default_rng(self.ripple_seed)
            for _ in range(3):
                fx, fy, phase = rng.uniform(2, 9, 2).tolist() + [rng.uniform(0, 6.28)]
                z = z + self.ripple / 3 * (1 + np.sin(fx * u * 2 + fy * v * 2 + phase))
        return np.stack([t * straight[0] + bent_s * across[0], t * straight[1] + bent_s * across[1], z + 0.0004], axis=-1)

    @classmethod
    def draw(cls, rng: random.Random, width: float, height: float, folds: list[dict] | None = None) -> "Sheet":
        """A shape at random: flat, bowed, a book's page, wavy, or folded where the paper stage drew a crease."""
        kind = rng.choices(["flat", "bow", "book", "wave", "corner"], weights=[24, 38, 10, 16, 12])[0]
        sheet = cls(width, height, kind, ripple=rng.choice([0.0, 0.0003, 0.0008, 0.002]), ripple_seed=rng.getrandbits(32))
        upright = rng.random() < 0.6     # the straight direction runs up the page: the sheet bows across its width
        sheet.axis = (math.pi / 2 if upright else 0.0) + rng.gauss(0, 0.12)
        reach = (width if upright else height) / 2
        stations = np.linspace(-reach, reach, 41)
        if kind == "flat":
            profile = np.zeros_like(stations)
        elif kind == "bow":
            # How far the sheet's two edges turn from each other. A sheet on a table bows a little; a stack that has been
            # rolled or carried bows 40 to 55 degrees, as my real test photos of bent pages do (the `curve`
            # photos), and those are the photos the first readers read worst.
            total = math.radians(rng.uniform(4, 22) if rng.random() < 0.55 else rng.uniform(22, 60)) * rng.choice([-1, 1])
            profile = np.full_like(stations, total / (2 * reach)) * (1 + 0.4 * np.sin(stations / reach * rng.uniform(1, 3) + rng.uniform(0, 6)))
        elif kind == "book":
            sheet.axis = math.pi / 2 + rng.gauss(0, 0.03)                      # the spine runs up the page
            reach = width / 2
            stations = np.linspace(-reach, reach, 41)
            spine = rng.choice([-1, 1])                                        # which side the spine is on
            fall = rng.uniform(0.012, 0.04)
            profile = -math.radians(rng.uniform(12, 40)) / fall * np.exp(-(reach + spine * stations) / fall)   # the page drops into the gutter
            profile += math.radians(rng.uniform(-8, 12)) / (2 * reach)
        elif kind == "wave":
            profile = math.radians(rng.uniform(4, 18)) / reach * np.sin(stations / reach * math.pi * rng.uniform(0.8, 2.2) + rng.uniform(0, 6.28))
        else:   # a corner or an edge that lifts
            sheet.axis += rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
            side = rng.choice([-1, 1])
            fall = rng.uniform(0.02, 0.06)
            profile = math.radians(rng.uniform(10, 40)) / fall * np.exp(-(reach - side * stations) / fall)
        for fold in folds or []:
            if kind != "book" and fold["across"] != upright:   # a crease across the page bends it along its height
                sharp = np.exp(-((stations - (0.5 - fold["at"]) * 2 * reach) / 0.004) ** 2)
                profile = profile + math.radians(rng.uniform(4, 30)) * rng.choice([-1, 1]) * sharp / (0.004 * math.sqrt(math.pi))
        sheet.curvature = [round(float(value), 4) for value in profile]
        return sheet


def stacked(sheet: Sheet, stack: list, columns: int = 60, rows: int = 80) -> tuple[list[np.ndarray], float]:
    """The sheets under the top one: each a lattice of room points, `rows + 1` by `columns + 1` by (x, y, z), and how
    far below the top sheet the last one reaches. `stack` gives each sheet's place from the one above it: metres
    along and across the page, and a turn in degrees.

    Each is the sheet above it, moved. Moved sideways, a bent sheet would cut through the one above and cover its
    music: this happened in a trial render, where a third of a page lay under blank paper its truth knew nothing of.
    So each is dropped until it clears, by the most the shape's height changes over the distance moved, and a
    sheet's thickness more. Where that would open a gap no stack has, the sheet is moved less."""
    u, v = np.meshgrid(np.linspace(0, 1, columns + 1), np.linspace(0, 1, rows + 1))
    points = sheet.points(u, v)
    layers, depth = [], 0.0
    for along, across, turn in stack:
        # The slope of the sheet above at each lattice point: how its height changes with x and with y.
        by_column, by_row = np.gradient(points, axis=1), np.gradient(points, axis=0)
        plane = np.stack([by_column[..., :2], by_row[..., :2]], axis=-2)                       # d(x, y) over one column, over one row
        slope = np.linalg.solve(plane, np.stack([by_column[..., 2], by_row[..., 2]], axis=-1)[..., None])[..., 0]
        for _ in range(6):
            angle = math.radians(turn)
            # Where each point moves to: turned about the page's middle, then shifted.
            moved_x = points[..., 0] * math.cos(angle) - points[..., 1] * math.sin(angle) + along
            moved_y = points[..., 0] * math.sin(angle) + points[..., 1] * math.cos(angle) + across
            rise = slope[..., 0] * (moved_x - points[..., 0]) + slope[..., 1] * (moved_y - points[..., 1])   # how much higher the sheet above is where the point came from
            drop = 1.5 * float(np.max(np.abs(rise))) + 0.00012
            if drop < 0.0007:
                break
            along, across, turn = along / 2, across / 2, turn / 2
        moved = np.stack([moved_x, moved_y, points[..., 2] - drop], axis=-1)
        layers.append(moved)
        depth += drop
        points = moved
    return layers, depth


@dataclass
class Camera:
    """A pinhole camera: where it is, where it looks, how it is rolled, and its picture's size in pixels."""

    location: tuple[float, float, float]
    target: tuple[float, float, float]
    roll: float                   # radians, about the line of sight; 0 keeps the page's top toward the picture's top
    width: int
    height: int
    lens_mm: float = LENS_MM

    @property
    def focal_px(self) -> float:
        return self.lens_mm / FRAME_MM * max(self.width, self.height)

    def axes(self) -> np.ndarray:
        """The camera's right, up and backward directions in the room, as the columns of a matrix."""
        forward = np.subtract(self.target, self.location)
        forward = forward / np.linalg.norm(forward)
        hint = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.999 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        cos, sin = math.cos(self.roll), math.sin(self.roll)
        right, up = cos * right + sin * up, -sin * right + cos * up
        return np.stack([right, up, -forward], axis=1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """The pixel each room point lands on: x to the right, y downward, from the picture's top left corner."""
        seen = (np.asarray(points, np.float64) - np.array(self.location)) @ self.axes()
        depth = -seen[..., 2]
        return np.stack([self.width / 2 + seen[..., 0] / depth * self.focal_px, self.height / 2 - seen[..., 1] / depth * self.focal_px], axis=-1)

    @classmethod
    def draw(cls, rng: random.Random, sheet: Sheet, width: int, height: int) -> tuple["Camera", dict]:
        """A hand-held shot of the sheet: mostly from above and from the foot of the page, sometimes steep, close or
        off-centre."""
        # Most people hold the phone nearly over the page. Steep shots are the tail, not the middle.
        tilt = math.radians(min(abs(rng.gauss(0, 11)), 45) if rng.random() < 0.88 else rng.uniform(20, 48))
        around = math.radians(rng.gauss(270, 35) if rng.random() < 0.8 else rng.uniform(0, 360))   # 270: from the foot of the page
        fill = rng.choices([rng.uniform(0.8, 1.0), rng.uniform(0.55, 0.8), rng.uniform(1.0, 1.15)], weights=[84, 10, 6])[0]   # over 1 cuts the page off
        lens = rng.choices([LENS_MM, 13.0, 48.0], weights=[90, 4, 6])[0]
        aim = sheet.points(0.5 + rng.gauss(0, 0.04), 0.5 + rng.gauss(0, 0.04))
        # Far enough that the page's height takes `fill` of the picture's long side, and its width fits the short side;
        # farther when tilted, since the near edge then looms larger.
        short = min(width, height) / max(width, height)
        distance = max(sheet.height, sheet.width / short) / fill / (FRAME_MM / lens) * (1 + 0.3 * math.sin(tilt))
        direction = np.array([math.sin(tilt) * math.cos(around), math.sin(tilt) * math.sin(around), math.cos(tilt)])
        roll = rng.gauss(0, math.radians(4)) if rng.random() < 0.85 else rng.uniform(-0.4, 0.4)
        camera = cls(tuple(float(c) for c in aim + distance * direction), tuple(float(c) for c in aim), roll, width, height, lens)
        return camera, {"tilt_deg": round(math.degrees(tilt), 1), "from_deg": round(math.degrees(around) % 360, 1), "roll_deg": round(math.degrees(roll), 1),
                        "distance_m": round(distance, 3), "lens_mm": lens, "fill": round(fill, 3)}


def page_grid(sheet: Sheet, camera: Camera, columns: int = 49, rows: int = 65) -> np.ndarray:
    """Where a lattice of page points lands in the picture: `rows` by `columns` by (x, y). Point [r, c] is the page
    point u = c / (columns - 1), v = r / (rows - 1)."""
    u, v = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
    return camera.project(sheet.points(u, v))


def to_picture(grid: np.ndarray, u, v) -> np.ndarray:
    """Page points to pixels, by interpolating the lattice."""
    rows, columns = grid.shape[:2]
    u, v = np.clip(np.asarray(u, np.float64), 0, 1) * (columns - 1), np.clip(np.asarray(v, np.float64), 0, 1) * (rows - 1)
    c0, r0 = np.minimum(u.astype(int), columns - 2), np.minimum(v.astype(int), rows - 2)
    fu, fv = (u - c0)[..., None], (v - r0)[..., None]
    return (grid[r0, c0] * (1 - fu) * (1 - fv) + grid[r0, c0 + 1] * fu * (1 - fv) + grid[r0 + 1, c0] * (1 - fu) * fv + grid[r0 + 1, c0 + 1] * fu * fv)
