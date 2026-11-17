"""Turns the engraver's black-on-white page into a printed sheet: paper colour and grain, and how the ink went on.

Nothing here moves anything: the page's symbols stay where the truth says they are. Bending, tilting and lighting
belong to the scene. README.md, "Paper and print".
"""

from __future__ import annotations

import random

import cv2
import numpy as np

# Base colour (R, G, B), how far it varies, grain, and how often.
PAPERS = {
    "white": ((0.965, 0.965, 0.965), 0.025, 0.008, 45),
    "bright": ((0.985, 0.985, 0.995), 0.01, 0.005, 15),
    "cream": ((0.955, 0.93, 0.85), 0.03, 0.01, 18),
    "aged": ((0.9, 0.84, 0.69), 0.05, 0.016, 12),
    "grey": ((0.86, 0.86, 0.83), 0.03, 0.02, 10),
}
PRINTS = ["laser", "laser", "inkjet", "offset", "photocopy", "faded"]


def cloud(rng: np.random.Generator, height: int, width: int, cells: int) -> np.ndarray:
    """Smooth noise from 0 to 1: a coarse grid of random values, stretched to the picture's size."""
    grid = rng.random((max(2, cells * height // max(height, width) + 2), max(2, cells * width // max(height, width) + 2)), dtype=np.float32)
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def print_page(page: np.ndarray, seed: random.Random, dpi: int = 400) -> tuple[np.ndarray, dict]:
    """`page` is the engraver's picture as greys, 0 black to 1 white, at `dpi` dots to the inch. Returns the sheet in
    colour (floats, 0 to 1) and what was done to it."""
    rng = np.random.default_rng(seed.getrandbits(63))
    height, width = page.shape
    dot = dpi / 200   # the sizes below were set at 200 dots to the inch
    ink = 1 - page.astype(np.float32)
    kind = seed.choices(list(PAPERS), weights=[weight for *_, weight in PAPERS.values()])[0]
    printing = seed.choice(PRINTS)
    base, spread, grain, _ = PAPERS[kind]
    made = {"paper": kind, "print": printing}

    # How the ink went on.
    density = {"laser": (0.88, 1.0), "inkjet": (0.75, 0.95), "offset": (0.92, 1.0), "photocopy": (0.85, 1.0), "faded": (0.4, 0.7)}[printing]
    density = seed.uniform(*density)
    if printing == "inkjet":
        ink = cv2.GaussianBlur(ink, (0, 0), seed.uniform(0.5, 1.1) * dot)
        ink *= 1 - 0.06 * (np.sin(np.arange(height, dtype=np.float32) * seed.uniform(0.2, 0.6) / dot) > 0.9)[:, None]   # faint bands across the page
    elif printing == "photocopy":
        # A copy of a copy: edges harden and roughen, and specks appear. The blur is kept well under a staff line's
        # width: a wider one thinned the lines below the cut and erased every staff on the page.
        soft = cv2.GaussianBlur(ink, (0, 0), seed.uniform(0.25, 0.45) * dot) + (rng.random(ink.shape, dtype=np.float32) - 0.5) * seed.uniform(0.05, 0.2)
        ink = cv2.GaussianBlur((soft > seed.uniform(0.3, 0.45)).astype(np.float32), (0, 0), 0.4 * dot)
        specks = rng.random(ink.shape, dtype=np.float32) < seed.uniform(0.0, 0.0006) / dot**2
        ink = np.maximum(ink, cv2.dilate(specks.astype(np.float32), np.ones((round(2 * dot), round(2 * dot)), np.uint8)))
    elif printing == "faded":
        streak = cloud(rng, 1, width, 12)   # lighter stripes down the page, as from a tired cartridge
        ink *= 0.6 + 0.4 * streak
    ink *= density * (0.9 + 0.1 * cloud(rng, height, width, 6))
    ink_colour = np.array(seed.choice([(0.05, 0.05, 0.05), (0.08, 0.07, 0.07), (0.1, 0.09, 0.12), (0.12, 0.1, 0.08), (0.03, 0.03, 0.04)]), np.float32)
    made |= {"ink_density": round(density, 3)}

    # The paper itself.
    tint = np.array(base, np.float32) + np.array([seed.uniform(-spread, spread) for _ in range(3)], np.float32) * np.array([1, 1, 1.5], np.float32)
    sheet = np.ones((height, width, 3), np.float32) * tint.clip(0, 1)
    sheet *= (1 - seed.uniform(0.0, 0.05) * cloud(rng, height, width, 5))[..., None]            # uneven tone
    sheet += ((rng.random((height, width), dtype=np.float32) - 0.5) * 2 * grain)[..., None]        # grain
    if kind in ("aged", "cream") and seed.random() < 0.7:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        edge = np.minimum(np.minimum(xs, width - xs) / width, np.minimum(ys, height - ys) / height)
        browning = np.exp(-edge / seed.uniform(0.02, 0.08)) * seed.uniform(0.05, 0.25) * (0.6 + 0.4 * cloud(rng, height, width, 9))
        sheet *= 1 - browning[..., None] * np.array([0.5, 0.8, 1.3], np.float32)
        made["browned_edges"] = True
    if kind == "grey" or seed.random() < 0.1:
        flecks = (rng.random((height, width), dtype=np.float32) < 0.002 / dot**2).astype(np.float32)
        sheet *= 1 - cv2.GaussianBlur(cv2.dilate(flecks, np.ones((round(dot), round(dot)), np.uint8)), (0, 0), 0.7 * dot)[..., None] * seed.uniform(0.3, 0.8)
    if seed.random() < 0.15:
        # The other side of the sheet showing through, mirrored and faint.
        through = cv2.GaussianBlur(np.roll(ink[:, ::-1], round(seed.randint(-40, 40) * dot), axis=0), (0, 0), 2.0 * dot)
        strength = seed.uniform(0.03, 0.1)
        sheet *= 1 - strength * through[..., None]
        made["show_through"] = round(strength, 3)
    if seed.random() < 0.15:
        # The sheet underneath showing through thin paper: another page's print, the right way round, faint and out
        # of place. This page's own print, turned upside down and moved, stands in for the other page.
        below = np.roll(ink[::-1, ::-1], (round(seed.randint(-120, 120) * dot), round(seed.randint(-60, 60) * dot)), axis=(0, 1))
        strength = seed.uniform(0.02, 0.07)
        sheet *= 1 - strength * cv2.GaussianBlur(below, (0, 0), 1.5 * dot)[..., None]
        made["sheet_below_shows"] = round(strength, 3)
    if seed.random() < 0.08:
        # A stain: a blotch with a darker rim, as a spilt drink leaves.
        blotch = cloud(rng, height, width, seed.randint(3, 6))
        level = seed.uniform(0.6, 0.8)
        inside = np.clip((blotch - level) * 12, 0, 1)
        rim = np.exp(-((blotch - level) * 40) ** 2)
        stain = (0.5 * inside + 0.8 * rim) * seed.uniform(0.08, 0.25)
        sheet *= 1 - stain[..., None] * np.array([0.5, 0.8, 1.2], np.float32)
        made["stain"] = True
    folds = []
    if seed.random() < 0.2:
        # A sheet that was folded keeps a line: darker on one side of the crease, lighter on the other.
        for across in seed.sample([True, False], k=seed.choice([1, 1, 2])):
            where = seed.choice([0.5, 0.5, 1 / 3, 2 / 3]) + seed.uniform(-0.01, 0.01)
            strength = seed.uniform(0.03, 0.1)
            axis = np.arange(height if across else width, dtype=np.float32) - where * (height if across else width)
            line = strength * (np.exp(-((axis / dot - 2) / 3) ** 2) - 0.7 * np.exp(-((axis / dot + 2) / 3) ** 2))
            sheet *= 1 - (line[:, None, None] if across else line[None, :, None])
            folds.append({"across": across, "at": round(where, 3)})
        made["folds"] = folds

    sheet = sheet.clip(0, 1)
    printed = sheet * (1 - ink[..., None] * (1 - ink_colour))
    return printed.clip(0, 1), made
