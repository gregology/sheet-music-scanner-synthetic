"""Makes one example: a photo and its truth, from the dataset's seed and the example's number.

Each stage draws from a generator of its own, named for the stage, so a change to one stage's random choices does
not change the others'. The truth's positions are given twice: on the flat page, and in the photo.
README.md, "An example".
"""

from __future__ import annotations

import functools
import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import camera as phone
from .compose import Composer
from .engrave import Style, engrave
from .generate import FORMAT
from .geometry import Camera, Sheet, page_grid, to_picture
from .handwriting import annotate
from .paper import print_page
from .scores import Corpus
from .things import page_things

PAGE_DPI = 400          # the printed page's resolution on the sheet: finer than the photo sees it
STAFF_POINTS = 17       # points along each staff line in the truth
TRIES = 200             # how many pieces are tried for one that holds what was asked for; a try takes a tenth of a second
REAL_LEVEL = 2          # a real score is engraved as the composer's third level is: a staff near 7 mm


def stage(seed: int, index: int, name: str) -> random.Random:
    return random.Random(f"{seed}/{index}/{name}")


@dataclass(frozen=True)
class Music:
    """Where a dataset's music comes from. The default is the composer alone, as the first datasets were made."""

    real: float = 0.0               # the share of examples whose music is a real score
    scores: str | tuple[str, ...] | None = None   # the folder of the corpus of real scores, or several folders
    holds: tuple[str, ...] = ()     # names from `things.py`: every page holds all of them


@functools.lru_cache(maxsize=4)
def corpus(folder: str) -> Corpus:
    return Corpus(Path(folder))


def corpus_of(music: Music, seed: int, index: int) -> Corpus:
    """The corpus an example's real score is from. Of several, one is drawn by how many measures each has to draw, or
    how many of them hold what is asked for: every such measure of real music is as likely as any other. The draw
    has a stage of its own, and with one corpus there is none, so that one corpus's examples stay the ones they were."""
    if isinstance(music.scores, str):
        return corpus(music.scores)
    all_of = [corpus(folder) for folder in music.scores]
    weights = [one.holding(music.holds) for one in all_of]
    return stage(seed, index, "corpus").choices(all_of, weights=weights)[0] if any(weights) else all_of[0]   # the first then says that no line holds it


def engraved_page(seed: int, index: int, music: Music = Music()):
    """The example's music, engraved: the page, the composer's level (None for a real score) and where the music is
    from. With names to hold, the music is drawn again, up to `TRIES` times, until page one holds them all."""
    real = music.real > 0 and stage(seed, index, "source").random() < music.real
    for attempt in range(TRIES if music.holds else 1):
        name = f"{seed}/{index}/music" + (f"/{attempt}" if attempt else "")
        if real:
            rng = random.Random(name)
            piece, source = corpus_of(music, seed, index).piece(rng, list(music.holds))
            level = None
        else:
            composer = Composer(name)
            piece, level, source = composer.piece(), composer.level, {"kind": "composer"}
        style = Style.draw(stage(seed, index, "style"), REAL_LEVEL if real else level)
        if real and source["from_measure"] and rng.random() < 0.7:
            style.header = False   # a page from the middle of a piece has no title over it
        page = engrave(piece, style)
        if set(music.holds) <= page_things(page.truth):
            return page, level, source
    raise RuntimeError(f"no page in {TRIES} tries holds {' and '.join(music.holds)}")


def printed_page(seed: int, index: int, music: Music = Music()) -> tuple[np.ndarray, dict]:
    """The sheet before the scene: composed or cut from a real score, engraved, printed and marked. Returns its
    picture and its truth."""
    page, level, source = engraved_page(seed, index, music)
    grey = np.asarray(Image.open(io.BytesIO(page.raster(PAGE_DPI))).convert("L"), np.float32) / 255
    sheet, paper = print_page(grey, stage(seed, index, "paper"), PAGE_DPI)
    sheet, written = annotate(sheet, page.truth, stage(seed, index, "hand"), PAGE_DPI / 25.4)
    truth = page.truth | {"level": level, "source": source, "paper": paper, "handwriting": written}
    return sheet, truth


def place(truth: dict, grid: np.ndarray, width: int, height: int) -> None:
    """Adds to the truth where everything is in the photo, in pixels from its top left corner."""

    def quad(box: list[float]) -> list[list[float]]:
        x0, y0, x1, y1 = box
        return np.round(to_picture(grid, np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1])), 1).tolist()

    def inside(points) -> float:
        points = np.asarray(points).reshape(-1, 2)
        return float(np.mean((points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)))

    for system in truth["systems"]:
        staff = system["staff"]
        along = np.linspace(staff["left"], staff["right"], STAFF_POINTS)
        staff["lines_px"] = [np.round(to_picture(grid, along, np.full_like(along, line)), 1).tolist() for line in staff["lines"]]
        staff["in_photo"] = round(inside(staff["lines_px"]), 3)
        for item in system["symbols"] + system["marks"]:
            item["quad"] = quad(item["box"])
            if inside(item["quad"]) < 1:
                item["cut_off"] = True
    for mark in truth["handwriting"]:
        mark["quad"] = quad(mark["box"])
    edge = np.linspace(0, 1, 9)
    outline = np.concatenate([to_picture(grid, edge, np.zeros(9)), to_picture(grid, np.ones(9), edge)[1:], to_picture(grid, edge[::-1], np.ones(9))[1:], to_picture(grid, np.zeros(9), edge[::-1])[1:-1]])
    truth["page"]["outline_px"] = np.round(outline, 1).tolist()
    truth["page"]["in_photo"] = round(inside(grid), 3)
    truth["page"]["grid_px"] = np.round(grid[::2, ::2], 1).tolist()   # every other lattice point: 33 rows by 25 columns


def shows_its_page(light: np.ndarray, truth: dict, sample: int = 60) -> float:
    """The share of the truth's notes that have ink where the truth says they are, in the clean render.

    A GPU renderer kept loaded between examples once drew an earlier example's page under a later example's truth.
    The photos looked right and every label was wrong. This asks the picture itself: a notehead fills its box with
    ink, and a different page puts at most a staff line there.

    It can also turn away a right page. Example 7062 of the first dataset was hollow whole and half notes, printed
    faint and photographed out of focus: ink on 14% of its notes. One in 10,000 examples is lost this way."""
    levels = light.mean(axis=2)
    height, width = levels.shape
    notes = [symbol for system in truth["systems"] for symbol in system["symbols"] if symbol["type"] == "note" and not symbol.get("cut_off")]
    notes = notes[::max(1, len(notes) // sample)]
    inked = 0
    for note in notes:
        quad = np.array(note["quad"])
        x0, y0, x1, y1 = int(quad[:, 0].min()), int(quad[:, 1].min()), int(quad[:, 0].max()) + 1, int(quad[:, 1].max()) + 1
        reach = max(x1 - x0, y1 - y0)
        around = levels[max(0, y0 - 2 * reach):y1 + 2 * reach, max(0, x0 - 2 * reach):x1 + 2 * reach]
        inside = levels[max(0, y0):y1, max(0, x0):x1]
        if inside.size and around.size:
            paper = float(np.percentile(around, 90))
            inked += float(np.mean(inside < 0.7 * paper)) > 0.25
    return inked / len(notes) if notes else 1.0


def make_example(seed: int, index: int, width: int, height: int, work: Path, samples: int = 32, music: Music = Music()) -> tuple[bytes, dict]:
    """The example's photo as a JPEG, and its truth. Blender is loaded here, not at import, so the stages before the
    scene can run and be tested without it."""
    from . import scene

    sheet_picture, truth = printed_page(seed, index, music)
    rng = stage(seed, index, "scene")
    sheet = Sheet.draw(rng, truth["page"]["width_mm"] / 1000, truth["page"]["height_mm"] / 1000, truth["paper"].get("folds"))
    camera, shot = Camera.draw(rng, sheet, width, height)
    room = scene.draw_room(rng)
    focus_error = rng.gauss(0, 0.03) if rng.random() < 0.9 else rng.uniform(-0.25, 0.25)
    scene.configure_once(samples)
    light = scene.photograph(sheet, camera, room, sheet_picture, work, focus_error)
    if not np.isfinite(light).all() or float(light.mean()) < 1e-5:
        raise RuntimeError("the render came back empty")   # a GPU out of memory gives a black picture, not an error
    place(truth, page_grid(sheet, camera), width, height)
    agreement = shows_its_page(light, truth)
    if agreement < 0.5:
        raise RuntimeError(f"the render does not show the page its truth describes: ink on {agreement:.0%} of its notes")
    settings = phone.draw_settings(stage(seed, index, "camera"))
    photo = phone.develop(light, settings, stage(seed, index, "noise").getrandbits(63))
    truth |= {
        "format": FORMAT, "seed": seed, "index": index, "photo": {"width": width, "height": height},
        "sheet": {"shape": sheet.kind, "straight_axis_deg": round(float(np.degrees(sheet.axis)), 1), "curvature_per_m": sheet.curvature, "ripple_mm": sheet.ripple * 1000},
        "shot": shot | {"focus_error": round(focus_error, 3)}, "room": room, "camera": settings,
    }
    return phone.jpeg(photo, settings["jpeg_quality"]), truth
