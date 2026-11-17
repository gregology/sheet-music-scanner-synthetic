"""Marks a page by hand: finger numbers over notes, bowings, circles, ticks, words, highlighter.

Every mark is drawn as pen strokes of our own, so no handwriting font or sample is needed. A sign is a few strokes
through control points in a unit square; each time it is written the points are jittered, slanted and scaled, and a
smooth curve is drawn through them, so no two are alike. Each mark's truth says what was written, with what, where,
and for a finger number on which note. README.md, "Handwriting".
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

# Each sign: one or more ways of writing it; each way a list of strokes; each stroke a list of (x, y), y downward.
SIGNS: dict[str, list[list[list[tuple[float, float]]]]] = {
    "0": [[[(0.5, 0.05), (0.22, 0.3), (0.22, 0.7), (0.5, 0.95), (0.78, 0.7), (0.78, 0.3), (0.5, 0.05)]],
          [[(0.62, 0.1), (0.35, 0.12), (0.25, 0.5), (0.4, 0.92), (0.68, 0.8), (0.72, 0.35), (0.5, 0.05)]]],
    "1": [[[(0.5, 0.05), (0.5, 0.95)]],
          [[(0.28, 0.32), (0.55, 0.05), (0.55, 0.95)]],
          [[(0.3, 0.3), (0.55, 0.05), (0.55, 0.95)], [(0.3, 0.95), (0.8, 0.95)]]],
    "2": [[[(0.2, 0.3), (0.35, 0.08), (0.65, 0.08), (0.78, 0.3), (0.6, 0.55), (0.2, 0.95), (0.82, 0.95)]],
          [[(0.2, 0.3), (0.4, 0.07), (0.7, 0.12), (0.72, 0.38), (0.3, 0.85), (0.18, 0.9), (0.3, 0.78), (0.55, 0.9), (0.85, 0.95)]]],
    "3": [[[(0.2, 0.15), (0.5, 0.05), (0.78, 0.2), (0.6, 0.45), (0.42, 0.5), (0.65, 0.55), (0.8, 0.75), (0.5, 0.95), (0.18, 0.85)]],
          [[(0.2, 0.08), (0.78, 0.08), (0.45, 0.45), (0.75, 0.6), (0.7, 0.85), (0.4, 0.95), (0.18, 0.82)]]],
    "4": [[[(0.62, 0.05), (0.15, 0.65), (0.88, 0.65)], [(0.62, 0.3), (0.62, 0.95)]],
          [[(0.25, 0.05), (0.2, 0.55), (0.8, 0.55)], [(0.7, 0.1), (0.68, 0.95)]]],
    "5": [[[(0.78, 0.07), (0.3, 0.07), (0.25, 0.45), (0.55, 0.4), (0.8, 0.6), (0.7, 0.88), (0.4, 0.96), (0.18, 0.85)]],
          [[(0.3, 0.08), (0.26, 0.45), (0.6, 0.42), (0.8, 0.65), (0.6, 0.93), (0.2, 0.88)], [(0.3, 0.08), (0.8, 0.05)]]],
    "up-bow": [[[(0.15, 0.05), (0.5, 0.95), (0.85, 0.05)]]],
    "down-bow": [[[(0.15, 0.95), (0.15, 0.1), (0.85, 0.1), (0.85, 0.95)]]],
    "tick": [[[(0.1, 0.55), (0.35, 0.9), (0.9, 0.1)]]],
    "breath": [[[(0.55, 0.1), (0.62, 0.35), (0.4, 0.62)]]],
    "sharp": [[[(0.35, 0.1), (0.3, 0.9)], [(0.68, 0.05), (0.63, 0.85)], [(0.1, 0.42), (0.9, 0.3)], [(0.1, 0.7), (0.9, 0.58)]]],
    "flat": [[[(0.3, 0.0), (0.3, 0.95), (0.72, 0.7), (0.62, 0.5), (0.3, 0.66)]]],
    "natural": [[[(0.3, 0.05), (0.3, 0.72), (0.7, 0.62)], [(0.7, 0.95), (0.7, 0.3), (0.3, 0.4)]]],
    "plus": [[[(0.5, 0.15), (0.5, 0.85)], [(0.15, 0.5), (0.85, 0.5)]]],
    "cross": [[[(0.15, 0.15), (0.85, 0.85)], [(0.85, 0.15), (0.15, 0.85)]]],
    "arrow-up": [[[(0.5, 0.95), (0.5, 0.08)], [(0.2, 0.38), (0.5, 0.05), (0.8, 0.38)]]],
    "arrow-down": [[[(0.5, 0.05), (0.5, 0.92)], [(0.2, 0.62), (0.5, 0.95), (0.8, 0.62)]]],
    "exclaim": [[[(0.5, 0.05), (0.5, 0.68)], [(0.5, 0.88), (0.52, 0.95)]]],
    "star": [[[(0.5, 0.05), (0.7, 0.9), (0.1, 0.38), (0.9, 0.38), (0.3, 0.9), (0.5, 0.05)]]],
    "slashes": [[[(0.2, 0.95), (0.45, 0.05)], [(0.55, 0.95), (0.8, 0.05)]]],
    "glasses": [[[(0.25, 0.5), (0.12, 0.35), (0.25, 0.25), (0.4, 0.4), (0.25, 0.55)], [(0.4, 0.4), (0.6, 0.4)], [(0.75, 0.5), (0.62, 0.35), (0.75, 0.25), (0.9, 0.4), (0.75, 0.55)]]],
}
WORDS = ["slow", "watch", "count", "soft", "loud", "tune", "breathe", "light", "sing", "listen", "again", "ok", "rit", "cresc", "dim", "bow", "fast", "relax", "high", "low", "time", "easy"]
# The few lower-case letters the words need, each one stroke.
LETTERS = {
    "a": [(0.75, 0.45), (0.45, 0.35), (0.22, 0.62), (0.42, 0.92), (0.75, 0.6), (0.75, 0.4), (0.8, 0.95)],
    "b": [(0.25, 0.0), (0.25, 0.95), (0.6, 0.9), (0.78, 0.62), (0.55, 0.4), (0.25, 0.55)],
    "c": [(0.78, 0.45), (0.5, 0.35), (0.25, 0.62), (0.5, 0.93), (0.8, 0.82)],
    "d": [(0.75, 0.5), (0.45, 0.38), (0.22, 0.65), (0.45, 0.93), (0.75, 0.7), (0.75, 0.0), (0.78, 0.95)],
    "e": [(0.25, 0.65), (0.75, 0.6), (0.6, 0.38), (0.3, 0.5), (0.3, 0.82), (0.55, 0.94), (0.8, 0.82)],
    "f": [(0.75, 0.1), (0.5, 0.02), (0.4, 0.3), (0.4, 0.95), (0.4, 0.45), (0.2, 0.45), (0.7, 0.45)],
    "g": [(0.75, 0.45), (0.45, 0.36), (0.22, 0.62), (0.45, 0.9), (0.75, 0.6), (0.75, 0.4), (0.75, 1.2), (0.45, 1.35), (0.25, 1.2)],
    "h": [(0.25, 0.0), (0.25, 0.95), (0.25, 0.55), (0.55, 0.38), (0.75, 0.55), (0.75, 0.95)],
    "i": [(0.5, 0.4), (0.5, 0.95)],
    "k": [(0.25, 0.0), (0.25, 0.95), (0.25, 0.65), (0.7, 0.38), (0.4, 0.6), (0.75, 0.95)],
    "l": [(0.5, 0.0), (0.5, 0.95)],
    "m": [(0.1, 0.95), (0.1, 0.42), (0.3, 0.38), (0.45, 0.55), (0.45, 0.95), (0.45, 0.5), (0.68, 0.38), (0.85, 0.55), (0.85, 0.95)],
    "n": [(0.22, 0.95), (0.22, 0.42), (0.5, 0.38), (0.75, 0.55), (0.75, 0.95)],
    "o": [(0.5, 0.38), (0.25, 0.62), (0.5, 0.93), (0.75, 0.66), (0.5, 0.38)],
    "r": [(0.3, 0.95), (0.3, 0.42), (0.35, 0.55), (0.55, 0.38), (0.75, 0.45)],
    "s": [(0.75, 0.45), (0.5, 0.36), (0.3, 0.5), (0.5, 0.65), (0.72, 0.78), (0.5, 0.94), (0.25, 0.85)],
    "t": [(0.45, 0.1), (0.45, 0.85), (0.6, 0.95), (0.45, 0.75), (0.45, 0.42), (0.2, 0.42), (0.75, 0.42)],
    "u": [(0.22, 0.4), (0.22, 0.78), (0.45, 0.93), (0.75, 0.75), (0.75, 0.4), (0.78, 0.95)],
    "w": [(0.1, 0.4), (0.25, 0.93), (0.48, 0.5), (0.68, 0.93), (0.88, 0.4)],
    "x": [(0.2, 0.4), (0.8, 0.95), (0.5, 0.68), (0.8, 0.4), (0.2, 0.95)],
    "y": [(0.22, 0.4), (0.3, 0.8), (0.55, 0.9), (0.75, 0.4), (0.75, 1.2), (0.5, 1.35), (0.3, 1.2)],
}
TOOLS = {   # colour as paper sees it (0 dark, 1 none), how wide in mm, how solid, how grainy
    "pencil": ((0.32, 0.32, 0.34), (0.25, 0.55), (0.5, 0.85), 0.45),
    "black pen": ((0.08, 0.08, 0.1), (0.25, 0.45), (0.85, 1.0), 0.05),
    "blue pen": ((0.1, 0.18, 0.55), (0.25, 0.45), (0.8, 1.0), 0.05),
    "red pen": ((0.7, 0.1, 0.12), (0.3, 0.5), (0.8, 1.0), 0.08),
    "highlighter": ((0.98, 0.93, 0.25), (3.0, 5.0), (0.45, 0.7), 0.1),
}


def smooth(points: list[tuple[float, float]], steps: int = 10) -> np.ndarray:
    """A curve through the points (Catmull-Rom), as closely spaced points."""
    p = np.array([points[0], *points, points[-1]], dtype=np.float64)
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    out = [0.5 * ((2 * p[i + 1]) + (-p[i] + p[i + 2]) * t + (2 * p[i] - 5 * p[i + 1] + 4 * p[i + 2] - p[i + 3]) * t**2 + (-p[i] + 3 * p[i + 1] - 3 * p[i + 2] + p[i + 3]) * t**3) for i in range(len(p) - 3)]
    return np.vstack([*out, p[-2][None]])


class Hand:
    """One person marking one page with one tool, onto the page's picture."""

    def __init__(self, rng: random.Random, height: int, width: int, px_per_mm: float):
        self.rng = rng
        self.height, self.width, self.px_per_mm = height, width, px_per_mm
        self.cover = np.zeros((height, width), np.float32)        # how much of each pixel the tool covers
        self.colour = np.zeros((height, width, 3), np.float32)    # and in what colour
        self.slant = rng.uniform(-0.25, 0.35)
        self.wobble = rng.uniform(0.02, 0.07)
        self.pick_tool(rng.choices(["pencil", "black pen", "blue pen", "red pen"], weights=[70, 12, 12, 6])[0])

    def pick_tool(self, name: str) -> None:
        colour, (thin, thick), (faint, solid), self.grain = TOOLS[name]
        self.tool, self.ink = name, colour
        self.line = max(1, round(self.rng.uniform(thin, thick) * self.px_per_mm))
        self.solid = self.rng.uniform(faint, solid)

    def strokes(self, strokes: list[list[tuple[float, float]]], x: float, y: float, w: float, h: float, turn: float = 0.0) -> list[float]:
        """Writes strokes given in a unit square into the box at (x, y) of size w by h, in px. Returns the box drawn."""
        rng = self.rng
        cos, sin = math.cos(turn), math.sin(turn)
        drawn = []
        for stroke in strokes:
            points = [(px + rng.gauss(0, self.wobble), py + rng.gauss(0, self.wobble)) for px, py in stroke]
            curve = smooth(points) if len(points) > 2 else np.array(points, dtype=np.float64)
            cx, cy = (curve[:, 0] - 0.5) * w + (0.5 - curve[:, 1]) * self.slant * h, (curve[:, 1] - 0.5) * h
            drawn.append(np.stack([x + w / 2 + cx * cos - cy * sin, y + h / 2 + cx * sin + cy * cos], axis=1))
        # Drawn on a patch just big enough, not on a layer the size of the page: a page has a hundred strokes.
        every = np.vstack(drawn)
        left, top = max(0, int(every[:, 0].min()) - self.line - 2), max(0, int(every[:, 1].min()) - self.line - 2)
        right, bottom = min(self.width, int(every[:, 0].max()) + self.line + 3), min(self.height, int(every[:, 1].max()) + self.line + 3)
        if right <= left or bottom <= top:
            return [x, y, x + w, y + h]
        layer = np.zeros((bottom - top, right - left), np.uint8)
        for placed in drawn:
            cv2.polylines(layer, [np.round((placed - [left, top]) * 8).astype(np.int32)], False, 255, self.line, cv2.LINE_AA, shift=3)
        mask = layer.astype(np.float32) / 255
        if not mask.any():
            return [x, y, x + w, y + h]
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = top + ys.min(), top + ys.max() + 1, left + xs.min(), left + xs.max() + 1
        patch = mask[y0 - top:y1 - top, x0 - left:x1 - left] * self.solid
        if self.grain:   # a pencil skips over the paper's tooth
            noise = np.random.default_rng(rng.getrandbits(32)).random(patch.shape, dtype=np.float32)
            patch *= 1 - self.grain * noise
        region = self.cover[y0:y1, x0:x1]
        mixed = np.maximum(region, patch)
        self.colour[y0:y1, x0:x1][patch > region] = self.ink
        self.cover[y0:y1, x0:x1] = mixed
        return [float(x0), float(y0), float(x1), float(y1)]

    def sign(self, name: str, x: float, y: float, size: float, turn: float | None = None) -> list[float]:
        way = self.rng.choice(SIGNS[name])
        width = size * self.rng.uniform(0.55, 0.75)
        return self.strokes(way, x - width / 2, y - size / 2, width, size, self.rng.gauss(0, 0.08) if turn is None else turn)

    def word(self, text: str, x: float, y: float, size: float) -> list[float]:
        """Writes a word starting at x, its letters sitting on a line through y."""
        boxes, turn, advance = [], self.rng.gauss(0, 0.06), size * self.rng.uniform(0.5, 0.62)
        for position, letter in enumerate(text):
            boxes.append(self.strokes([LETTERS[letter]], x + position * advance * math.cos(turn), y - size + position * advance * math.sin(turn), advance, size, turn))
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    def ring(self, box: list[float]) -> list[float]:
        """Circles a box, as a hand does: an oval that overshoots where it closes."""
        x0, y0, x1, y1 = box
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * self.rng.uniform(1.25, 1.6), (y1 - y0) / 2 * self.rng.uniform(1.25, 1.6)
        start, sweep = self.rng.uniform(0, 2 * math.pi), 2 * math.pi * self.rng.uniform(1.02, 1.15)
        points = [(0.5 + 0.5 * math.cos(start + sweep * i / 14) * (1 + self.rng.gauss(0, 0.04)), 0.5 + 0.5 * math.sin(start + sweep * i / 14) * (1 + self.rng.gauss(0, 0.04))) for i in range(15)]
        slant, self.slant = self.slant, 0.0
        drawn = self.strokes([points], cx - rx, cy - ry, 2 * rx, 2 * ry)
        self.slant = slant
        return drawn

    def apply(self, picture: np.ndarray) -> np.ndarray:
        """The picture with the marks on it. A highlighter tints what is beneath; a pencil or pen covers it."""
        cover = self.cover[..., None]
        return picture * (1 - cover) + picture * self.colour * cover if self.tool == "highlighter" else picture * (1 - cover) + self.colour * cover


def annotate(picture: np.ndarray, truth: dict, rng: random.Random, px_per_mm: float) -> tuple[np.ndarray, list[dict]]:
    """Marks the page (floats, 0 dark to 1 light) and returns it with the truth of each mark. Positions in the truth
    are fractions of the page."""
    height, width = picture.shape[:2]
    systems = truth["systems"]
    if not systems or rng.random() < 0.45:
        return picture, []
    hand = Hand(rng, height, width, px_per_mm)
    written: list[dict] = []

    def record(kind: str, box: list[float], system: int, **more) -> None:
        written.append({"type": kind, "tool": hand.tool, "system": system, "box": [round(box[0] / width, 5), round(box[1] / height, 5), round(box[2] / width, 5), round(box[3] / height, 5)], **more})

    def pixels(box: list[float]) -> list[float]:
        return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]

    # Finger numbers, in runs of neighbouring notes, above the staff or under it.
    top_finger = rng.choice([4, 5])
    density = rng.choice([0.0, 0.05, 0.1, 0.2, 0.4])
    for number, system in enumerate(systems):
        lines = system["staff"]["lines"]
        space = (lines[4] - lines[0]) / 4 * height
        notes = [symbol for symbol in system["symbols"] if symbol["type"] == "note"]
        size = space * rng.uniform(1.5, 2.4)
        index = 0
        while index < len(notes):
            if rng.random() < density:
                above = rng.random() < 0.8
                for note in notes[index:index + rng.choice([1, 1, 2, 3, 4])]:
                    x0, y0, x1, y1 = pixels(note["box"])
                    y = min(lines[0] * height, y0) - size * rng.uniform(0.8, 1.5) if above else max(lines[4] * height, y1) + size * rng.uniform(0.8, 1.5)
                    digit = str(rng.randint(0 if top_finger == 4 else 1, top_finger))
                    record("fingering", hand.sign(digit, (x0 + x1) / 2 + rng.gauss(0, space * 0.25), y, size), number, text=digit, of=note["id"])
                    index += 1
            index += 1
        # Other marks a player leaves.
        for _ in range(rng.choice([0, 0, 0, 1, 1, 2])):
            kind = rng.choice(["ring", "ring", "bowing", "tick", "word", "accidental", "arrow", "breath", "sign", "slashes"])
            if not notes:
                break
            note = rng.choice(notes)
            x0, y0, x1, y1 = pixels(note["box"])
            above_y = min(lines[0] * height, y0) - space * rng.uniform(1.5, 3.0)
            if kind == "ring":
                target = pixels(rng.choice([s for s in system["symbols"] + system["marks"] if s["type"] in ("note", "key", "time", "dynamic", "rest", "accid")])["box"])
                record("circle", hand.ring(target), number)
            elif kind == "bowing":
                name = rng.choice(["up-bow", "down-bow"])
                record(name, hand.sign(name, (x0 + x1) / 2, above_y, space * rng.uniform(1.4, 2.0)), number, of=note["id"])
            elif kind == "word":
                text = rng.choice(WORDS)
                below = rng.random() < 0.5
                y = max(lines[4] * height, y1) + space * rng.uniform(3.5, 5.0) if below else above_y
                record("word", hand.word(text, x0, y, space * rng.uniform(1.8, 2.8)), number, text=text)
            elif kind == "accidental":
                name = rng.choice(["sharp", "flat", "natural"])
                record(name, hand.sign(name, x0 - space * rng.uniform(1.0, 1.6), (y0 + y1) / 2, space * rng.uniform(2.0, 2.8)), number, of=note["id"])
            elif kind == "arrow":
                name = rng.choice(["arrow-up", "arrow-down"])
                record(name, hand.sign(name, (x0 + x1) / 2, above_y, space * rng.uniform(2.0, 3.0)), number, of=note["id"])
            else:
                name = {"tick": "tick", "breath": "breath", "slashes": "slashes"}.get(kind) or rng.choice(["star", "exclaim", "glasses", "plus", "cross"])
                record(name, hand.sign(name, x1 + space * rng.uniform(0.5, 2.0), above_y, space * rng.uniform(1.8, 3.0)), number)
    picture = hand.apply(picture)
    # A highlighter over a loudness letter or a word, by a second pass with a different tool.
    targets = [(number, mark) for number, system in enumerate(systems) for mark in system["marks"] if mark["type"] in ("dynamic", "tempo", "direction", "rehearsal")]
    if targets and rng.random() < 0.15:
        marker = Hand(rng, height, width, px_per_mm)
        marker.pick_tool("highlighter")
        marker.ink = rng.choice([(0.98, 0.93, 0.25), (1.0, 0.6, 0.8), (0.55, 0.95, 0.5), (0.5, 0.85, 1.0)])
        marker.wobble = 0.01
        hand = marker
        for number, mark in rng.sample(targets, k=min(len(targets), rng.randint(1, 4))):
            x0, y0, x1, y1 = pixels(mark["box"])
            middle = 0.5 + rng.gauss(0, 0.05)
            record("highlight", marker.strokes([[(0.0, middle), (1.0, middle + rng.gauss(0, 0.05))]], x0 - 4, y0, x1 - x0 + 8, y1 - y0), number, of=mark.get("id"))
        picture = marker.apply(picture)
    return picture, written
