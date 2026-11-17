"""Draws an example's truth over its picture, to check by eye that the two agree."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

COLOURS = {"note": "#1f77b4", "grace": "#17becf", "rest": "#2ca02c", "measure-rest": "#2ca02c", "multirest": "#2ca02c", "clef": "#d62728", "key": "#ff7f0e", "time": "#9467bd", "barline": "#8c564b"}


def page_overlay(png: bytes, truth: dict) -> Image.Image:
    """The flat page with a box round every symbol and mark, and each line's tokens beneath its staff."""
    picture = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = picture.size
    draw = ImageDraw.Draw(picture)
    for system in truth["systems"]:
        staff = system["staff"]
        for line in (staff["lines"][0], staff["lines"][-1]):
            draw.line([(staff["left"] * width, line * height), (staff["right"] * width, line * height)], fill="#e377c2", width=1)
        for mark in system["marks"]:
            if mark["type"] != "beam":
                x0, y0, x1, y1 = mark["box"]
                draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline="#bcbd22")
        for symbol in system["symbols"]:
            x0, y0, x1, y1 = symbol["box"]
            draw.rectangle([x0 * width - 1, y0 * height - 1, x1 * width + 1, y1 * height + 1], outline=COLOURS[symbol["type"]], width=2)
        draw.text((staff["left"] * width, staff["lines"][-1] * height + 0.012 * height), " ".join(system["tokens"])[:230], fill="#d62728", font_size=11)
    return picture


def photo_overlay(photo: Image.Image, truth: dict) -> Image.Image:
    """The photo with the truth drawn where it says things are: the page's outline, each staff's lines, a box round
    every symbol, and every handwritten mark."""
    picture = photo.convert("RGB")
    draw = ImageDraw.Draw(picture)
    outline = [tuple(point) for point in truth["page"]["outline_px"]]
    draw.line(outline + outline[:1], fill="#ff0000", width=2)
    for system in truth["systems"]:
        for line in (system["staff"]["lines_px"][0], system["staff"]["lines_px"][-1]):
            draw.line([tuple(point) for point in line], fill="#e377c2", width=1)
        for symbol in system["symbols"]:
            corners = [tuple(point) for point in symbol["quad"]]
            draw.line(corners + corners[:1], fill=COLOURS[symbol["type"]], width=1)
    for mark in truth["handwriting"]:
        corners = [tuple(point) for point in mark["quad"]]
        draw.line(corners + corners[:1], fill="#00ff00", width=2)
    return picture
