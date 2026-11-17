"""Engraves a piece as a page, and reads from the drawing what is on the page and where.

Verovio draws the page as SVG. Each element of ours carries our id, so the drawing says which notes landed on the
page, on which line of music, and inside which box. The page's truth is the piece's own data, laid out as drawn:
line by line, left to right. Positions are fractions of the page, from its top left corner.
"""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cairosvg
import verovio

from .compose import CLEFS, Event, Measure, Piece, to_mei

SVG = "{http://www.w3.org/2000/svg}"
FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
PAGES = {"letter": (816, 1056), "a4": (794, 1123)}   # px at 96 to the inch
MM_PER_PX = 25.4 / 96
# The drawn elements kept as marks: signs that belong to a note or a measure, each with its box.
MARKS = {"accid", "artic", "dots", "fing", "dynam", "verse", "harm", "fermata", "trill", "slur", "tie", "hairpin", "tempo", "dir", "reh", "tupletNum", "tupletBracket", "beam", "mNum"}
BARLINES = {None: "barline", "dbl": "barline-double", "end": "barline-end", "rptend": "repeat-end", "rptstart": "repeat-start"}
_BOXES = re.compile(r'<g id="bbox-[^"]*" class="[^"]*bounding-box"\s*(?:/>|>\s*<rect [^>]*/>\s*</g>)')


@dataclass
class Style:
    """How the page is engraved."""

    font: str = "Leipzig"
    page: str = "letter"
    scale: int = 38                 # percent; 38 draws a staff 7.2 mm high
    margin: tuple[int, int, int, int] = (54, 54, 54, 54)   # left, right, top, bottom, px at 96 to the inch
    header: bool = True
    options: dict = field(default_factory=dict)            # Verovio options beyond the page's size

    @classmethod
    def draw(cls, rng: random.Random, level: int = 1) -> "Style":
        """An engraving style at random: font, page, staff size, spacing and line weights."""
        scale = rng.randint(40, 54) if level == 0 else round(rng.triangular(29, 46, 36))   # most music is printed with a staff near 7 mm
        side = rng.randint(38, 86)
        options = {
            "spacingLinear": round(rng.uniform(0.2, 0.33), 3),
            "spacingNonLinear": round(rng.uniform(0.52, 0.66), 3),
            "spacingStaff": rng.randint(8, 16),
            "spacingSystem": rng.randint(4, 16),
            "staffLineWidth": round(rng.uniform(0.1, 0.22), 3),
            "stemWidth": round(rng.uniform(0.14, 0.28), 3),
            "barLineWidth": round(rng.uniform(0.2, 0.42), 3),
            "ledgerLineThickness": round(rng.uniform(0.18, 0.36), 3),
            "lyricSize": round(rng.uniform(3.6, 5.0), 2),
            "justifyVertically": rng.random() < 0.3,
            "mnumInterval": rng.choice([0, 0, 1, 5, 10]),
            # A rest of several measures as most engravers draw it, a thick bar with the count above, or as older
            # editions do, in whole and double whole rests up to four measures.
            "multiRestStyle": rng.choice(["block", "block", "block", "auto"]),
            "multiRestThickness": round(rng.uniform(1.4, 2.6), 2),
            "tupletNumHead": rng.random() < 0.3,
            "graceFactor": round(rng.uniform(0.62, 0.8), 2),
            "slurMidpointThickness": round(rng.uniform(0.4, 0.8), 2),
        }
        margin = (side + rng.randint(-8, 8), side + rng.randint(-8, 8), rng.randint(36, 90), rng.randint(36, 90))
        return cls(rng.choice(FONTS), rng.choice(["letter", "letter", "a4"]), scale, margin, rng.random() < 0.85, options)


@dataclass
class Page:
    svg: str            # the drawing, without the boxes
    width: int          # px at 96 to the inch
    height: int
    truth: dict

    def raster(self, dpi: int = 200) -> bytes:
        """The page as a PNG: black ink on white."""
        return cairosvg.svg2png(bytestring=self.svg.encode(), scale=dpi / 96, background_color="white")


def engrave(piece: Piece, style: Style) -> Page:
    width, height = PAGES[style.page]
    left, right, top, bottom = style.margin
    units = 100 / style.scale   # Verovio lays out in its own units and draws `scale` percent of a pixel for each
    toolkit = verovio.toolkit()
    toolkit.setOptions({
        "scale": style.scale, "pageWidth": round(width * units), "pageHeight": round(height * units),
        "pageMarginLeft": round(left * units), "pageMarginRight": round(right * units), "pageMarginTop": round(top * units), "pageMarginBottom": round(bottom * units),
        "font": style.font, "header": "auto" if style.header else "none", "footer": "none", "breaks": "auto", "svgBoundingBoxes": True, "xmlIdSeed": 1, **style.options,
    })
    if not toolkit.loadData(to_mei(piece)):
        raise ValueError("Verovio could not read the piece")
    drawing = toolkit.renderToSVG(1)
    truth = read_truth(drawing, piece)
    truth["page"] = {"size": style.page, "width_mm": round(width * MM_PER_PX, 1), "height_mm": round(height * MM_PER_PX, 1), "pages_engraved": toolkit.getPageCount()}
    truth["style"] = {"font": style.font, "scale": style.scale, "margin_px": list(style.margin), "header": style.header, **style.options}
    return Page(_BOXES.sub("", drawing), width, height, truth)


def classes(element: ET.Element) -> list[str]:
    return element.get("class", "").split()


class Index:
    """What each of our ids stands for."""

    def __init__(self, piece: Piece):
        self.events: dict[str, Event] = {}
        self.under: dict[str, tuple[str, int]] = {}     # a chord's lower notes: the note each is under, and how far down
        self.measures: dict[str, tuple[int, Measure]] = {}
        self.marks: dict[str, dict] = {}
        self.state: list[tuple[str, int, tuple]] = []   # clef, key and time in force in each measure
        clef, key, time = piece.clef, piece.key, piece.time
        for number, measure in enumerate(piece.measures):
            clef, key, time = measure.clef or clef, measure.key if measure.key is not None else key, measure.time or time
            self.state.append((clef, key, time))
            self.measures[measure.id] = (number, measure)
            for event in measure.events:
                for place, note in enumerate(event.under, 1):
                    self.events[note.id] = note
                    self.under[note.id] = (event.id, place)
            for name, kind, text in ((f"{measure.id}t", "tempo", measure.tempo), (f"{measure.id}h", "rehearsal", measure.rehearsal), (f"{measure.id}d", "direction", measure.direction)):
                if text:
                    self.marks[name] = {"type": kind, "text": text, "measure": measure.id}
            for event in measure.events:
                self.events[event.id] = event
                for name, kind, value in ((f"{event.id}y", "dynamic", event.dynamic), (f"{event.id}u", "fermata", event.fermata), (f"{event.id}w", "trill", event.trill),
                                          (f"{event.id}f", "fingering", event.fingering), (f"{event.id}c", "chord", event.chord), (f"{event.id}v", "lyric", event.lyric and event.lyric[0])):
                    if value:
                        self.marks[name] = {"type": kind, "of": event.id} | ({"text": value} if isinstance(value, str) else {})
                for number_, artic in enumerate(event.artics):
                    self.marks[f"{event.id}a{number_}"] = {"type": "articulation", "of": event.id, "text": artic}
        for number, (start, end) in enumerate(piece.slurs):
            self.marks[f"s{number}"] = {"type": "slur", "from": start, "to": end}
        for number, (form, start, end) in enumerate(piece.hairpins):
            self.marks[f"h{number}"] = {"type": "hairpin", "text": form, "from": start, "to": end}


def clef_token(name: str) -> str:
    shape, line, displaced, *_ = CLEFS[name]
    return f"clef-{shape}{line}" + ("_8vb" if displaced else "")


def key_token(sharps: int) -> str:
    return "key-0" if sharps == 0 else f"key-{abs(sharps)}{'#' if sharps > 0 else 'b'}"


def time_token(time: tuple) -> str:
    count, unit, symbol = time
    return {"common": "time-C", "cut": "time-C|"}.get(symbol, f"time-{count}/{unit}")


def tuplet_token(tuplet: tuple[int, int]) -> str:
    """`:3` for three in the time of two, `:6` for six in the time of four: a count alone stands for that many in the
    time of the power of two below it. Any other tuplet names both numbers: `:2/3` is two in the time of three."""
    count, time_of = tuplet
    return f":{count}" if time_of == 2 ** (count.bit_length() - 1) and count != time_of else f":{count}/{time_of}"


def length_token(event: Event, tuplet: tuple[int, int] | None) -> str:
    return f"{event.dur}{'.' * event.dots}" + (tuplet_token(tuplet) if tuplet else "")


def event_symbol(event: Event, tuplet: tuple[int, int] | None, under: str | None = None, rocked: int = 0) -> dict:
    """A note or rest as the truth records it, and its token. `under` names the note a chord's lower note is under.
    `rocked` is the strokes of a tremolo between the note before and this one."""
    if event.rest:
        return {"type": "rest", "id": event.id, "duration": event.dur, "dots": event.dots, "tuplet": tuplet and list(tuplet), "token": f"rest-{length_token(event, tuplet)}"}
    name = event.step.upper() + ("#" * event.alter if event.alter > 0 else "b" * -event.alter) + str(event.octave)
    symbol = {
        "type": "grace" if event.grace else "note", "id": event.id, "pitch": name, "step": event.step.upper(), "alter": event.alter, "octave": event.octave,
        "duration": event.dur, "dots": event.dots, "tuplet": tuplet and list(tuplet), "accidental_shown": event.shown, "tie": event.tie,
        "token": f"{'grace' if event.grace else 'note'}-{name}_{length_token(event, tuplet)}",
    }
    for name_, value in (("articulations", event.artics), ("fermata", event.fermata), ("trill", event.trill), ("fingering", event.fingering),
                         ("lyric", event.lyric and {"text": event.lyric[0], "place": event.lyric[1]}), ("dynamic", event.dynamic), ("chord", event.chord),
                         ("under", under), ("slash", event.slash), ("tremolo", event.tremolo), ("rocked", rocked)):
        if value:
            symbol[name_] = value
    return symbol


def read_truth(drawing: str, piece: Piece) -> dict:
    """What the drawing of page 1 holds, line by line."""
    index = Index(piece)
    tuplets = {note.id: cell.tuplet for measure in piece.measures for cell in measure.cells for event in cell.events for note in [event] + event.under}
    shows = {event.id: cell.show for measure in piece.measures for cell in measure.cells if cell.tuplet for event in cell.events}
    rocked = {cell.events[-1].id: cell.rock for measure in piece.measures for cell in measure.cells if cell.rock}   # the second note of each tremolo
    root = ET.fromstring(drawing)
    inner = next(element for element in root.iter(f"{SVG}svg") if "definition-scale" in classes(element))
    _, _, view_width, view_height = (float(value) for value in inner.get("viewBox").split())
    margin = next(element for element in inner.iter(f"{SVG}g") if "page-margin" in classes(element))
    shift_x, shift_y = (float(value) for value in re.search(r"translate\(([-\d.]+),\s*([-\d.]+)\)", margin.get("transform")).groups())

    def box_of(element: ET.Element, deep: bool = False) -> list[float] | None:
        """The element's box as fractions of the page: left, top, right, bottom. Verovio boxes words on the text
        inside the element, not on the element, so `deep` takes the box around every box within."""
        if deep:
            rects = [rect for group in element.iter(f"{SVG}g") if "bounding-box" in classes(group) for rect in group.findall(f"{SVG}rect")]
        else:
            first = element.find(f"{SVG}g")
            rects = first.findall(f"{SVG}rect") if first is not None and "bounding-box" in classes(first) else []
        if not rects:
            return None
        corners = [(float(r.get("x")), float(r.get("y")), float(r.get("x")) + float(r.get("width")), float(r.get("y")) + float(r.get("height"))) for r in rects]
        x0, y0, x1, y1 = min(c[0] for c in corners), min(c[1] for c in corners), max(c[2] for c in corners), max(c[3] for c in corners)
        return [round((x0 + shift_x) / view_width, 5), round((y0 + shift_y) / view_height, 5), round((x1 + shift_x) / view_width, 5), round((y1 + shift_y) / view_height, 5)]

    def text_box(element: ET.Element) -> list[float] | None:
        """A box for words Verovio drew without one, such as a lyric: estimated from where the text starts, its size
        and its length, since the engraver does not measure the font."""
        text = element.find(f".//{SVG}text")
        sized = None if text is None else next((span for span in text.iter(f"{SVG}tspan") if span.get("font-size")), None)
        if sized is None:
            return None
        x, y, size = float(text.get("x")), float(text.get("y")), float(sized.get("font-size").rstrip("px"))
        width = 0.5 * size * max(1, len("".join(sized.itertext()).strip()))
        x -= width / 2 if text.get("text-anchor") == "middle" else 0
        return [round((x + shift_x) / view_width, 5), round((y - 0.75 * size + shift_y) / view_height, 5), round((x + width + shift_x) / view_width, 5), round((y + 0.25 * size + shift_y) / view_height, 5)]

    systems = []
    for system in (element for element in margin.iter(f"{SVG}g") if "system" in classes(element) and "bounding-box" not in classes(element)):
        lines: set[float] = set()
        extent = [1.0, 0.0]
        anchors: list[dict] = []
        marks: list[dict] = []

        def walk(element: ET.Element, measure: str | None, owner: str | None, in_layer: bool) -> None:
            kinds = classes(element)
            kind = kinds[0] if kinds else ""
            name = element.get("id", "")
            if "bounding-box" in kinds:
                return
            if kind == "measure":
                measure = name
            elif kind == "staff":
                for path in element.findall(f"{SVG}path"):
                    x0, y0, x1, _ = (float(value) for value in re.match(r"M([-\d.]+) ([-\d.]+) L([-\d.]+) ([-\d.]+)", path.get("d")).groups())
                    lines.add(round((y0 + shift_y) / view_height, 5))
                    extent[0], extent[1] = min(extent[0], (x0 + shift_x) / view_width), max(extent[1], (x1 + shift_x) / view_width)
            elif kind == "layer":
                in_layer = True
            box = box_of(element)
            estimated = False
            if box is None and (name in index.marks or kind in MARKS):
                box = box_of(element, deep=True)
                if box is None and name in index.marks:
                    box, estimated = text_box(element), True
            if box is not None and measure in index.measures:
                if name in index.events:
                    anchors.append(event_symbol(index.events[name], tuplets[name], index.under.get(name, (None, 0))[0], rocked.get(name, 0)) | {"box": box, "measure": measure}
                                   | ({"tuplet_shown": shows[name]} if name in shows else {}))
                    owner = name
                elif kind in ("mRest", "multiRest"):
                    count = index.measures[measure][1].multirest
                    anchors.append({"type": "multirest" if count else "measure-rest", "id": name, "measures": count or 1, "token": f"multirest-{count}" if count else "rest-measure", "box": box, "measure": measure})
                elif kind in ("clef", "keySig", "meterSig", "barLine"):
                    anchors.append({"type": kind, "id": name, "box": box, "measure": measure, "in_layer": in_layer})
                elif name in index.marks:
                    marks.append(index.marks[name] | {"id": name, "box": box} | ({"box_estimated": True} if estimated else {}))
                elif kind in MARKS:
                    marks.append({"type": kind, "box": box} | ({"of": owner} if owner else {}))
            elif box is not None and name in index.marks:
                marks.append(index.marks[name] | {"id": name, "box": box})
            for child in element:
                if child.tag == f"{SVG}g":
                    walk(child, measure, owner, in_layer)

        walk(system, None, None, False)
        if len(lines) != 5:
            continue   # not a line of music: nothing was drawn on it
        # Left to right; a chord's notes together, from the top down, where its top note stands.
        centres = {anchor["id"]: (anchor["box"][0] + anchor["box"][2]) / 2 for anchor in anchors}
        anchors.sort(key=lambda anchor: (centres.get(index.under.get(anchor["id"], (anchor["id"], 0))[0], centres[anchor["id"]]), index.under.get(anchor["id"], (None, 0))[1]))
        symbols = [named for named in (name_anchor(anchor, anchors, index) for anchor in anchors) if named]
        # Verovio draws a measure's plain barline and then the repeat sign that opens the next: on the page they are one sign.
        symbols = [symbol for symbol, after in zip(symbols, symbols[1:] + [{}]) if not (symbol.get("style") == "single" and after.get("style") == "rptstart")]
        systems.append({
            "staff": {"lines": sorted(lines), "left": round(extent[0], 5), "right": round(extent[1], 5)},
            "symbols": symbols, "marks": marks, "tokens": tokens_of(symbols),
        })
    return {"systems": systems, "title": piece.title, "composer": piece.composer, "label": piece.label, "subtitle": piece.subtitle, "movement": piece.movement, "part": piece.part}


def name_anchor(anchor: dict, anchors: list[dict], index: Index) -> dict | None:
    """Says which clef, key, time signature or barline an unnamed drawn one is, from where it sits in its measure."""
    kind = anchor["type"]
    if kind not in ("clef", "keySig", "meterSig", "barLine"):
        return anchor
    number, measure = index.measures[anchor["measure"]]
    centre = (anchor["box"][0] + anchor["box"][2]) / 2
    sounding = [(other["box"][0] + other["box"][2]) / 2 for other in anchors if other["measure"] == anchor["measure"] and other["type"] in ("note", "grace", "rest", "measure-rest", "multirest")]
    after = bool(sounding) and centre > max(sounding)
    following = number + 1 if after and number + 1 < len(index.state) else number
    clef, key, time = index.state[following]
    common = {"id": anchor["id"], "box": anchor["box"], "measure": anchor["measure"]}
    if kind == "clef":
        # A clef inside the measure's notes is the next measure's, drawn early; one before them is the line's own.
        clef = index.state[number + 1][0] if anchor["in_layer"] and number + 1 < len(index.state) else index.state[number][0]
        return common | {"type": "clef", "clef": clef, "token": clef_token(clef)}
    if kind == "keySig":
        if key == 0 and anchor["box"][2] - anchor["box"][0] < 1e-4:
            return None   # C major with nothing to cancel: no sign is drawn
        return common | {"type": "key", "sharps": key, "token": key_token(key)}
    if kind == "meterSig":
        return common | {"type": "time", "count": time[0], "unit": time[1], "symbol": time[2], "token": time_token(time)}
    style = measure.right if after or not sounding else measure.left
    if not after and sounding and measure.left is None:
        return None   # the line's opening barline, which carries no meaning
    return common | {"type": "barline", "style": style or "single", "token": BARLINES[style]}


def tokens_of(symbols: list[dict]) -> list[str]:
    """The line as a flat sequence, left to right. A note that starts a tie is followed by `tie`; a chord's notes run
    from the top down, with `chord` before each one after the first; `tremolo` stands between two notes or chords
    that a tremolo joins."""
    tokens = []
    for symbol in symbols:
        if symbol.get("rocked"):
            tokens.append("tremolo")
        if symbol.get("under"):
            tokens.append("chord")
        tokens.append(symbol["token"])
        if symbol.get("tie") in ("i", "m"):
            tokens.append("tie")
    return tokens
