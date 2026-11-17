"""Real scores as music: the second source beside the composer.

A score file is read into the same plain data the composer writes (`Piece`, `Measure`, `Cell`, `Event`), one line of
music at a time: a singer's line, or one hand of the piano with its first voice alone. From there on a real line is
drawn, printed, bent and photographed as a made-up one is, and its truth is that data, never the picture.

`build` makes a corpus folder from OpenScore's Lieder Corpus (CC0): it downloads the scores at a fixed commit, has
MuseScore write each as MusicXML, reads every line, and keeps each line as a file with an `index.json` that says what
each line holds. `Corpus` draws a page's worth of measures from the folder. A measure whose notation the piece's data
cannot hold is not drawn, and no page runs across one. README.md, "Real scores".

Usage: uv run python -m sheet_synth.scores [--out scores/lieder] [--cache ~/.cache/sheet_synth/lieder]
       [--musescore <MuseScore 4's AppRun>] [--workers 8]
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import random
import subprocess
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

from .compose import CLEFS, STEPS, Cell, Event, Measure, Piece, show_accidentals
from .engrave import BARLINES, clef_token, event_symbol, key_token, time_token
from .things import bottom_line, line_things

SOURCE = "https://github.com/OpenScore/Lieder"
COMMIT = "38c5db510224d9facdc4b08d741fc788cfb58ea8"   # 2026-09-09; LICENSE.txt there is CC0 1.0 Universal
LENGTHS = {"whole": 1, "half": 2, "quarter": 4, "eighth": 8, "16th": 16, "32nd": 32, "64th": 64}
CLEF_NAMES = {(shape, line, -1 if displaced else 0): name for name, (shape, line, displaced, *_) in CLEFS.items()}
ARTICULATIONS = {"staccato": "stacc", "accent": "acc", "tenuto": "ten", "strong-accent": "marc", "staccatissimo": "stacciss", "up-bow": "upbow", "down-bow": "dnbow", "harmonic": "harm"}
PLACES = {"single": "s", "begin": "i", "middle": "m", "end": "t"}
BAR_STYLES = {"light-light": "dbl", "light-heavy": "end"}
SOURCE_CLEFS = {"G": "treble", "F": "bass", "C3": "alto", "C4": "tenor", "G8vb": "treble8"}   # MuseScore's names for the clefs we draw
MIN_RUN = 8      # a page is not cut from fewer measures in a row than this: a line or two of music
PAGE = [48, 64, 80, 96]   # how many measures a page is given, where the line has them; what does not fit on page one is not drawn
FULL = 64        # a page that starts this many measures before its stretch ends is most often full
MIN_NOTES = 24   # a page with fewer notes and rests than this in its measures is drawn again: a singer's line rests for pages
FAR = 10         # steps from the staff's middle line, past the third ledger line: a measure whose notes sit that far off on average is not drawn, where no MuseScore file vouches for the clef


def pack(value):
    """A piece's data as plain JSON, without the fields left as they are made."""
    if dataclasses.is_dataclass(value):
        out = {}
        for field in dataclasses.fields(value):
            mine = getattr(value, field.name)
            required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
            usual = None if required else field.default if field.default is not dataclasses.MISSING else field.default_factory()
            if required or mine != usual:
                out[field.name] = pack(mine)
        return out
    return [pack(item) for item in value] if isinstance(value, (list, tuple)) else value


def event_from(data: dict) -> Event:
    return Event(**data | {"under": [event_from(note) for note in data.get("under", [])], "lyric": tuple(data["lyric"]) if data.get("lyric") else None})


def measure_from(data: dict) -> Measure:
    cells = [Cell(**cell | {"events": [event_from(event) for event in cell["events"]], "tuplet": tuple(cell["tuplet"]) if cell.get("tuplet") else None}) for cell in data.get("cells", [])]
    return Measure(**data | {"cells": cells, "time": tuple(data["time"]) if data.get("time") else None})


def event_length(event: Event, cell: Cell) -> Fraction:
    """In quarter notes, as played."""
    if event.grace:
        return Fraction(0)
    length = Fraction(4, event.dur) * (2 - Fraction(1, 2 ** event.dots))
    if cell.tuplet:
        length = length * cell.tuplet[1] / cell.tuplet[0]
    return length / 2 if cell.rock else length


def source_clefs(path: Path) -> list[tuple[int, dict[int, str | None]]]:
    """From a score's own MuseScore file, for each staff in order: how many measures it has, and the clef each change
    of clef puts in force, by the place of the first measure that starts with it. A clef we do not draw is None.

    MuseScore 4.7.6 leaves some changes of clef out of the MusicXML it writes: of the corpus's 5,500, about 110, most
    of them written as the first thing in a measure. The notes after one keep their true pitches and would be drawn
    under the old clef, far off the staff. So the clef a measure starts with is taken from here."""
    staffs = []
    for staff in ET.parse(path).getroot().find("Score").findall("Staff"):
        measures = staff.findall("Measure")
        changes: dict[int, str | None] = {}
        for place, measure in enumerate(measures):
            for voice in measure.findall("voice") or [measure]:   # files written by MuseScore 2 keep a measure's signs in the measure itself
                signs = list(voice)
                music = [at for at, sign in enumerate(signs) if sign.tag in ("Chord", "Rest")]
                for at, sign in enumerate(signs):
                    if sign.tag == "Clef":   # before the measure's first note it is this measure's; anywhere after, the next one's
                        changes[place if not music or at < music[0] else place + 1] = SOURCE_CLEFS.get(sign.findtext("concertClefType"))
        staffs.append((len(measures), changes))
    return staffs


class LineReader:
    """Reads one staff's first voice out of a MusicXML part, measure by measure. `given` is the staff's changes of
    clef from the score's own file (`source_clefs`), where there is one."""

    def __init__(self, part: ET.Element, staff: int, voice: str, given: dict[int, str | None] | None = None, lost: bool = False):
        self.part, self.staff, self.voice = part, staff, voice
        self.given, self.lost = given, lost
        self.count = 0
        self.divisions = 1
        self.clef: str | None = None
        self.key = 0
        self.time: tuple | None = (4, 4, None)
        self.next_clef: str | None | bool = False   # a clef drawn at a measure's end: the next measure's
        self.unknown_clef = False
        self.open_slurs: dict[str, Event] = {}
        self.slurs: list[tuple[str, str]] = []
        self.wedge: tuple[str, Fraction] | None = None
        self.wedges: list[tuple[str, Fraction, Fraction]] = []
        self.timeline: list[tuple[Fraction, str]] = []    # when each note of the line starts, in quarter notes from its start
        self.offset = Fraction(0)
        self.ending: int | None = None
        self.ottava = False    # under an 8va or 8vb line, where the file gives the notes as they sound, not as they are printed
        self.measures: list[Measure] = []
        self.states: list[list] = []
        self.bad: list[str | None] = []
        self.cut: list[int] = []     # the measures where another voice on the staff was left out

    def id(self, kind: str) -> str:
        self.count += 1
        return f"{kind}{self.count}"

    def read(self) -> "LineReader":
        for xml in self.part.findall("measure"):
            self.measure(xml)
        notes = dict(self.timeline)   # one note for each start
        starts = sorted(notes)
        self.hairpins = []
        for form, start, stop in self.wedges:
            inside = [at for at in starts if start <= at < stop] or [at for at in starts if start <= at <= stop]
            if len(inside) > 1:
                self.hairpins.append((form, notes[inside[0]], notes[inside[-1]]))
        return self

    def measure(self, xml: ET.Element) -> None:
        measure = Measure(self.id("m"))
        bad: list[str] = []
        position = furthest = ours = 0    # in the file's divisions of a quarter note; `ours` is the voice's own length
        events: list[Event] = []
        starts: dict[str, int] = {}
        groups: dict[str, dict] = {}
        pitches: dict[str, list[tuple[int, int, str | None]]] = {}
        clefs: list[tuple[int, str | None]] = []
        directions: list[tuple[int, str, str]] = []
        tuplet: dict | None = None
        rock: dict | None = None
        beams = {False: None, True: None}   # the beam open over the notes, and the one open over grace notes
        beam_count = 0
        whole = others = False
        ottava = self.ottava
        current: Event | None = None
        if self.next_clef is not False and self.given is None:
            self.clef, measure.clef, self.next_clef = self.next_clef, self.next_clef, False

        for child in xml:
            if child.tag == "attributes":
                if child.findtext("divisions"):
                    self.divisions = int(child.findtext("divisions"))
                for clef in child.findall("clef"):
                    if int(clef.get("number") or 1) == self.staff:
                        clefs.append((position, CLEF_NAMES.get((clef.findtext("sign"), int(clef.findtext("line") or 0), int(clef.findtext("clef-octave-change") or 0)))))
                # A hidden key or time signature is left out, as the page leaves it out: the notes after a hidden key
                # show their accidentals, and a measure under a hidden time is drawn as one that does not fill its time.
                for key in child.findall("key"):
                    if int(key.get("number") or self.staff) == self.staff and key.findtext("fifths") is not None and key.get("print-object") != "no":
                        fifths = int(key.findtext("fifths"))
                        if position:
                            bad.append("a key signature inside the measure")
                        elif fifths != self.key or not self.measures:
                            measure.key = fifths
                        self.key = fifths
                for time in child.findall("time"):
                    if int(time.get("number") or self.staff) != self.staff or time.get("print-object") == "no":
                        continue
                    beats, unit = time.findtext("beats") or "", time.findtext("beat-type") or ""
                    new = (int(beats), int(unit), {"common": "common", "cut": "cut"}.get(time.get("symbol"))) if beats.isdigit() and unit.isdigit() and len(time.findall("beats")) == 1 else None
                    if position:
                        bad.append("a time signature inside the measure")
                    elif new != self.time or not self.measures:
                        measure.time = new
                    self.time = new
            elif child.tag == "backup":
                position -= int(child.findtext("duration"))
            elif child.tag == "forward":
                if child.findtext("voice") == self.voice and int(child.findtext("staff") or 1) == self.staff:
                    bad.append("a gap in the voice")
                position += int(child.findtext("duration"))
            elif child.tag == "note":
                chord, grace = child.find("chord") is not None, child.find("grace")
                start = starts[current.id] if chord and current is not None and (child.findtext("voice") or "1") == self.voice else position
                if not chord and grace is None:
                    position += int(child.findtext("duration") or 0)
                    furthest = max(furthest, position)
                    if (child.findtext("voice") or "1") == self.voice:
                        ours += int(child.findtext("duration") or 0)
                staff = int(child.findtext("staff") or 1)
                if (child.findtext("voice") or "1") != self.voice:
                    others = others or (staff == self.staff and child.get("print-object") != "no")
                    continue
                if staff != self.staff:
                    bad.append("a note on the other staff")
                    continue
                kind = child.find("type")
                if child.get("print-object") == "no":
                    bad.append("a hidden note or rest")
                if child.find("cue") is not None or (kind is not None and kind.get("size")):
                    bad.append("small notes")
                rest = child.find("rest")
                if rest is not None and (rest.get("measure") == "yes" or kind is None):
                    whole = True
                    continue
                if kind is None or kind.text not in LENGTHS:
                    bad.append(f"a note or rest of a length that is not drawn: {'none' if kind is None else kind.text}")
                    continue
                pitch = child.find("pitch")
                if rest is None and pitch is None:
                    bad.append("a note with no pitch")
                    continue
                tie = None
                if pitch is not None:
                    alter = float(pitch.findtext("alter") or 0)
                    if alter != int(alter) or abs(alter) > 2:
                        bad.append("a quarter tone")
                        continue
                    ties = {tied.get("type") for tied in child.findall("tie")}
                    tie = "m" if {"start", "stop"} <= ties else "i" if "start" in ties else "t" if "stop" in ties else None
                    sounding = (int(pitch.findtext("octave")) * 7 + STEPS.index(pitch.findtext("step").lower()), int(alter), tie)
                if chord:
                    if current is None or current.rest:
                        bad.append("a chord with no first note")
                    elif current.grace:
                        bad.append("a grace note that is a chord")
                    else:
                        pitches[current.id].append(sounding)
                    continue
                dots = len(child.findall("dot"))
                if dots > 2:
                    bad.append("three dots")
                current = event = Event(self.id("r" if rest is not None else "n"), rest=rest is not None, dur=LENGTHS[kind.text], dots=dots)
                events.append(event)
                starts[event.id] = start
                if rest is None:
                    pitches[event.id] = [sounding]
                if grace is not None:
                    event.grace, event.slash = True, grace.get("slash") == "yes"

                marks = [mark.text for mark in child.findall("beam") if mark.get("number", "1") == "1"]
                small = grace is not None
                if "begin" in marks:
                    beam_count += 1
                    beams[small] = event.beam = beam_count
                elif "continue" in marks or "end" in marks:
                    event.beam = beams[small]
                    if "end" in marks:
                        beams[small] = None
                elif event.rest and not small:
                    event.beam = beams[False]   # a rest under a beam carries no mark of it

                notations = child.find("notations")
                tremolo = None if notations is None else notations.find("ornaments/tremolo")
                modified = child.find("time-modification")
                stops = False
                for sign in ([] if notations is None else notations.findall("tuplet")):
                    if sign.get("type") == "start":
                        if tuplet is not None or modified is None:
                            bad.append("a tuplet inside a tuplet")
                        else:
                            tuplet = {"tuplet": (int(modified.findtext("actual-notes")), int(modified.findtext("normal-notes"))),
                                      "show": "none" if sign.get("show-number") == "none" else "bracket" if sign.get("bracket") == "yes" else "number",
                                      "above": {"above": True, "below": False}.get(sign.get("placement"))}
                    elif tuplet is None:
                        bad.append("a tuplet with no start")
                    else:
                        stops = True
                if tremolo is not None and tremolo.get("type") in ("start", "stop"):
                    if tuplet is not None:
                        bad.append("a tremolo between two notes inside a tuplet")
                    elif tremolo.get("type") == "start":
                        rock = {"rock": int(tremolo.text or 3)}
                        groups[event.id] = rock
                    elif rock is None:
                        bad.append("a tremolo with no start")
                    else:
                        groups[event.id], rock = rock, None
                elif tuplet is not None:
                    groups[event.id] = tuplet
                elif modified is not None and not small:
                    bad.append("a tuplet with no start")
                if tremolo is not None and tremolo.get("type") == "single" and rest is None:
                    event.tremolo = min(int(tremolo.text or 3), 4)
                if stops:
                    tuplet = None

                if notations is not None and rest is None:
                    for slur in notations.findall("slur"):
                        number = slur.get("number", "1")
                        if slur.get("type") == "start":
                            self.open_slurs[number] = event
                        elif slur.get("type") == "stop" and number in self.open_slurs:
                            first = self.open_slurs.pop(number)
                            if first is not event:
                                self.slurs.append((first.id, event.id))
                    for sign in notations.findall("articulations/*") + notations.findall("technical/*"):
                        if sign.tag in ARTICULATIONS and ARTICULATIONS[sign.tag] not in event.artics:
                            event.artics.append(ARTICULATIONS[sign.tag])
                        elif sign.tag == "fingering" and (sign.text or "").strip().isdigit() and len(sign.text.strip()) == 1:
                            event.fingering = sign.text.strip()
                    event.fermata = notations.find("fermata") is not None
                    event.trill = notations.find("ornaments/trill-mark") is not None
                lyric = child.find("lyric")
                if lyric is not None and rest is None and (lyric.findtext("text") or "").strip():
                    event.lyric = (lyric.findtext("text").strip()[:24], PLACES.get(lyric.findtext("syllabic"), "s"))
            elif child.tag == "direction":
                if int(child.findtext("staff") or 1) != self.staff:
                    continue
                for kind in child.findall("direction-type/*"):
                    if kind.tag == "dynamics" and len(kind) and kind[0].tag.isalpha() and kind[0].tag != "other-dynamics":
                        directions.append((position, "dynamic", kind[0].tag))
                    elif kind.tag == "words" and (kind.text or "").strip():
                        tempo = child.find("sound") is not None and child.find("sound").get("tempo") is not None or kind.get("font-weight") == "bold"
                        directions.append((position, "tempo" if tempo and child.get("placement") != "below" else "words", " ".join(kind.text.split())[:40]))
                    elif kind.tag == "rehearsal" and (kind.text or "").strip():
                        measure.rehearsal = kind.text.strip()[:3]
                    elif kind.tag == "wedge":
                        directions.append((position, "wedge", kind.get("type")))
                    elif kind.tag == "octave-shift":
                        self.ottava = kind.get("type") in ("up", "down")
                        ottava = True
            elif child.tag == "barline":
                style, repeat, place = child.findtext("bar-style"), child.find("repeat"), child.get("location", "right")
                for sign in child.findall("ending"):
                    if sign.get("type") == "start":
                        numbers = [int(number) for number in (sign.get("number") or "").replace(",", " ").split() if number.isdigit()]
                        self.ending = numbers[0] if numbers else 1
                        measure.ending = self.ending
                    else:
                        measure.ending, self.ending = self.ending, None
                if repeat is not None:
                    if repeat.get("direction") == "forward":
                        measure.left = "rptstart"
                    else:
                        measure.right = "rptend"
                elif place == "right" and style in BAR_STYLES:
                    measure.right = BAR_STYLES[style]
                elif place == "middle" or style in ("dashed", "dotted", "none", "heavy", "tick", "short", "heavy-heavy"):
                    bad.append("a barline that is not drawn")
        if measure.ending is None:
            measure.ending = self.ending

        if self.given is not None and self.measures:   # after the first measure, the score's own file says which clef a measure starts with
            place = len(self.measures)
            clefs = ([(0, self.given[place])] if place in self.given else []) + [(at, name) for at, name in clefs if at]
        if self.lost:
            bad.append("measures that do not line up with the score's own file")
        for at, name in clefs:
            if at == 0:
                if name != self.clef and self.measures:
                    measure.clef = name
                self.clef = name
            else:
                if at < furthest:
                    bad.append("a clef inside the measure")
                self.next_clef = name
        if ottava:
            bad.append("an 8va or 8vb line")
        if self.clef is None:
            bad.append("a clef that is not drawn")
        if self.time is None:
            bad.append("a time signature that is not drawn")
        if tuplet is not None:
            bad.append("a tuplet across the barline")
        if rock is not None:
            bad.append("a tremolo with no end")

        # A chord's notes from the top down: the event is its top note, the others go under it.
        for event in events:
            if event.rest:
                continue
            (event.di, event.alter, event.tie), *under = sorted(pitches[event.id], key=lambda pitch: pitch[:2], reverse=True)
            event.under = [Event(self.id("n"), di=di, alter=alter, tie=tie, dur=event.dur, dots=event.dots) for di, alter, tie in under]

        # A change of clef the MusicXML has lost leaves the notes after it far off the staff. With the score's own
        # file the clef is known. Without it, a measure that sits there is turned away: of the Lieder corpus's
        # measures under their true clefs 0.7% do, and of those under a lost one 42%, the worst of them.
        if self.given is None and not self.lost and self.clef is not None:
            heights = [note.di for event in events if not event.rest for note in [event, *event.under]]
            if heights and abs(sum(heights) / len(heights) - (bottom_line(clef_token(self.clef)) + 4)) >= FAR:
                bad.append("notes far off the staff for the clef")

        # Cells: a tuplet, a tremolo's two notes, or the notes under one beam.
        cells: list[tuple[list[Event], dict | None]] = []
        for place, event in enumerate(events):
            group = groups.get(event.id)
            last = cells[-1] if cells else None
            if group is not None:
                joins = last is not None and last[1] is group
            else:
                beam = next((other.beam for other in reversed(last[0]) if not other.grace), None) if last is not None and last[1] is None else None
                later = [other for other in events[place + 1:]]
                inside = event.grace and event.beam is None and any(other.beam == beam and not other.grace for other in later[:4])
                joins = beam is not None and (event.beam == beam or inside)
            if joins:
                last[0].append(event)
            else:
                cells.append(([event], group))
        for members, group in cells:
            cell = Cell(members, given=True, **(group or {}))
            if cell.rock:
                if len(members) != 2 or any(event.rest or event.grace for event in members):
                    bad.append("a tremolo that is not between two notes")
                for event in members:
                    event.beam = None
            measure.cells.append(cell)

        if whole:
            if events:
                bad.append("a whole-measure rest beside notes")
            measure.whole_rest = True
            measure.cells = []
        elif not events:
            bad.append("nothing in the voice")
        else:
            # The lengths as read must add up to what the file says the voice lasts: a tuplet or a dot misread shows here.
            length = sum((event_length(event, cell) for cell in measure.cells for event in cell.events), Fraction(0))
            if length != Fraction(ours, self.divisions):
                bad.append("lengths that do not add up to the file's")
            if self.time is not None:
                measure.pickup = xml.get("implicit") == "yes" or length != Fraction(4 * self.time[0], self.time[1])

        notes = [event for event in events if not event.rest and not event.grace]
        quarter = Fraction(1, self.divisions)
        for at, kind, value in directions:
            target = next((event for event in notes if starts[event.id] >= at), notes[-1] if notes else None)
            if kind == "wedge":
                if value in ("crescendo", "diminuendo"):
                    self.wedge = ("cres" if value == "crescendo" else "dim", self.offset + at * quarter)
                elif value == "stop" and self.wedge is not None:
                    self.wedges.append((*self.wedge, self.offset + at * quarter))
                    self.wedge = None
            elif kind == "tempo" and measure.tempo is None:
                measure.tempo = value
            elif target is not None:
                if kind == "dynamic" and target.dynamic is None:
                    target.dynamic = value
                elif kind == "words" and measure.direction is None:
                    measure.direction = value
        for event in notes:
            self.timeline.append((self.offset + starts[event.id] * quarter, event.id))
        self.offset += furthest * quarter
        if others:
            self.cut.append(len(self.measures))
        self.measures.append(measure)
        self.states.append([self.clef, self.key, list(self.time) if self.time else None])
        self.bad.append(bad[0] if bad else None)


def measure_things(measure: Measure, state: list) -> list[str]:
    """What one measure holds, by the names of `things.py`, with the clef, key and time in force."""
    clef, key, time = state
    symbols = []
    if measure.whole_rest or measure.multirest:
        symbols.append({"type": "multirest" if measure.multirest else "measure-rest"})
    for cell in measure.cells:
        for place, event in enumerate(cell.events):
            symbols.append(event_symbol(event, cell.tuplet, rocked=cell.rock if place else 0) | ({"tuplet_shown": cell.show} if cell.tuplet else {}))
            symbols += [event_symbol(note, cell.tuplet, under=event.id) for note in event.under]
    found, _ = line_things(symbols + [{"type": "barline"}], bottom_line(clef_token(clef)))
    found |= {f"clef: {clef_token(clef)[5:]}", clef_token(clef), key_token(key), time_token(tuple(time))}
    found |= {BARLINES[bar] for bar in (measure.left, measure.right) if bar}
    if abs(key) >= 4:
        found.add(f"key of {abs(key)} sharps or flats")
    return sorted(found)


def runs_of(bad: list) -> list[list[int]]:
    """The stretches of measures in a row that can be drawn, each at least `MIN_RUN` long: [first, one past the last]."""
    runs, start = [], None
    for at, reason in enumerate([*bad, "the end"]):
        if reason is None and start is None:
            start = at
        elif reason is not None and start is not None:
            if at - start >= MIN_RUN:
                runs.append([start, at])
            start = None
    return runs


def parse(path: Path) -> ET.Element:
    """A MusicXML file's score, plain (`.musicxml`) or packed (`.mxl`, a zip whose `META-INF/container.xml` names the score)."""
    if path.suffix != ".mxl":
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as packed:
        named = ET.fromstring(packed.read("META-INF/container.xml")).find(".//{*}rootfile")
        return ET.fromstring(packed.read(named.get("full-path")))


def read_score(path: Path, source: Path | None = None) -> list[dict]:
    """Every line of a MusicXML score: each staff's first voice, as a line's stored data. `source` is the MuseScore
    file the MusicXML was written from, for the changes of clef the conversion leaves out."""
    root = parse(path)
    given = source_clefs(source) if source else None
    staff_number = 0
    title = root.findtext("work/work-title") or root.findtext("movement-title") or ""
    composer = next((creator.text for creator in root.findall("identification/creator") if creator.get("type") == "composer"), None)
    names = {part.get("id"): (part.findtext("part-name") or "").strip() for part in root.findall("part-list/score-part")}
    lines = []
    for number, part in enumerate(root.findall("part")):
        staves = max([int(staves.text) for staves in part.iter("staves")] or [1])
        # Each voice belongs to the staff most of its notes are on; a staff's line is the first voice that belongs to it.
        places: dict[str, Counter] = {}
        for note in part.iter("note"):
            places.setdefault(note.findtext("voice") or "1", Counter())[int(note.findtext("staff") or 1)] += 1
        home = {voice: counts.most_common(1)[0][0] for voice, counts in places.items()}
        for staff in range(1, staves + 1):
            staff_number += 1
            mine = [voice for voice, where in home.items() if where == staff]
            if not mine:
                continue
            count, changes = given[staff_number - 1] if given and staff_number <= len(given) else (None, None)
            lost = given is not None and count != len(part.findall("measure"))
            reader = LineReader(part, staff, min(mine, key=int), None if lost else changes, lost).read()
            if not reader.measures:
                continue
            show_accidentals(Piece(title, composer, None, "treble", reader.states[0][1], (4, 4, None), reader.measures))   # for "accidental drawn" in the index
            things = [measure_things(measure, state) if reason is None else [] for measure, state, reason in zip(reader.measures, reader.states, reader.bad)]
            lines.append({
                "score": path.stem, "title": " ".join(title.split()), "composer": composer and " ".join(composer.split()), "label": " ".join(names.get(part.get("id"), "").split()) or None, "part": number, "staff": staff, "voice": reader.voice,
                "measures": [pack(measure) for measure in reader.measures], "states": reader.states, "bad": reader.bad, "things": things,
                "slurs": reader.slurs, "hairpins": reader.hairpins, "cut": reader.cut,
            })
    return lines


def window(line: dict, start: int, count: int, rng: random.Random) -> Piece:
    """A piece made of `count` measures of a stored line, from measure `start`: a page's worth of a real score."""
    measures = [measure_from(data) for data in line["measures"][start:start + count]]
    clef, key, time = line["states"][start]
    first = measures[0]
    first.clef = first.key = first.time = None   # the piece opens with them
    if rng.random() < 0.8:
        measures = join_rests(measures)
    mend_ties(measures)
    ids = {event.id for measure in measures for event in measure.events}
    whole = start == 0
    label = line["label"]
    piece = Piece(line["title"] or "Song", line["composer"] if whole or rng.random() < 0.5 else None, label, clef, key, tuple(time), measures,
                  slurs=[(a, b) for a, b in line["slurs"] if a in ids and b in ids], hairpins=[(form, a, b) for form, a, b in line["hairpins"] if a in ids and b in ids])
    if label and rng.random() < 0.5:
        piece.part, piece.airy = True, rng.random() < 0.4   # the line's name at the top left, as a player's part has it
    show_accidentals(piece)
    return piece


def mend_ties(measures: list[Measure]) -> None:
    """Takes away each tie that has only one end among these measures: one from a note before the window, one to a
    note after it. A tie joins a note to the same note in the very next chord, so every other is taken away too."""
    def sweep(events: list[Event | None], start: str, end: str) -> None:
        open_from: set[tuple[int, int]] = set()   # the pitches the event before has a tie towards this one on
        for event in events:
            if event is None or event.rest:
                open_from = set()
                continue
            mine = set()
            for note in [event] + event.under:
                if note.tie in (end, "m") and (note.di, note.alter) not in open_from:
                    note.tie = start if note.tie == "m" else None
                if note.tie in (start, "m"):
                    mine.add((note.di, note.alter))
            open_from = mine

    events = [event for measure in measures for event in (measure.events or [None])]
    sweep(events, "i", "t")
    sweep(events[::-1], "t", "i")


def join_rests(measures: list[Measure]) -> list[Measure]:
    """Whole-measure rests in a row as one rest of several measures, as a player's part prints them. A change of key,
    time or clef, a repeat, an ending or a tempo starts a new one."""
    out: list[Measure] = []
    for measure in measures:
        before = out[-1] if out else None
        plain = measure.whole_rest and not (measure.key is not None or measure.time or measure.clef or measure.left or measure.tempo or measure.rehearsal or measure.direction or measure.pickup)
        if plain and before is not None and (before.whole_rest or before.multirest) and before.right is None and before.ending == measure.ending and not before.pickup and before.multirest < 99:
            before.multirest = (before.multirest or 1) + 1
            before.whole_rest = False
            before.right = measure.right
        else:
            out.append(measure)
    return out


class Corpus:
    """A folder of real lines, and the pages cut from it."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        if not (self.folder / "index.json").exists():
            raise FileNotFoundError(f"no corpus of real scores in {self.folder}: make one with `python -m sheet_synth.scores --out {self.folder}`")
        self.index = json.loads((self.folder / "index.json").read_text())
        self.lines = self.index["lines"]
        self.weights = [line["usable"] for line in self.lines]
        self.wanted: dict[tuple[str, ...], list[int]] = {}   # for names asked for: how many measures of each line hold them

    @property
    def name(self) -> str:
        return f"{self.index['name']}@{self.index['commit'][:12]}/{self.index['sha256'][:12]}"

    def held(self, wanted: list[str] | tuple[str, ...]) -> list[int]:
        """For each line, how many of its measures hold the names asked for; kept, since a large corpus has many lines."""
        if tuple(wanted) not in self.wanted:
            self.wanted[tuple(wanted)] = [min(line["things"].get(name, 0) for name in wanted) for line in self.lines]
        return self.wanted[tuple(wanted)]

    def holding(self, wanted: list[str] | tuple[str, ...] = ()) -> int:
        """How many measures the corpus has to draw, or how many of them hold the names asked for."""
        return sum(self.held(wanted)) if wanted else sum(self.weights)

    def line(self, number: int) -> dict:
        with gzip.open(self.folder / self.lines[number]["file"], "rt") as file:
            return json.load(file)

    def piece(self, rng: random.Random, wanted: list[str] = ()) -> tuple[Piece, dict]:
        """A page's worth of a real line, drawn with `rng`. With `wanted`, measures that hold the first of those names
        are looked up and the page starts up to twelve measures before one; whether page one holds them all is for the
        caller to check on the page as engraved."""
        for _ in range(20):
            if wanted:
                weights = self.held(wanted)
                if not any(weights):
                    raise LookupError(f"no line of the corpus holds {' and '.join(wanted)}")
                number = rng.choices(range(len(self.lines)), weights=weights)[0]
                line = self.line(number)
                holding = [(at, run) for run in self.lines[number]["runs"] for at in range(*run) if wanted[0] in line["things"][at]]
                # A measure with more music after it makes a fuller page: one at the end of its stretch is chosen less often.
                at, (first, end) = rng.choices(holding, weights=[1 + min(end - at, FULL) ** 2 for at, (_, end) in holding])[0]
                start = max(first, min(at - rng.randint(0, 12), end - MIN_RUN))
            else:
                number = rng.choices(range(len(self.lines)), weights=self.weights)[0]
                line = self.line(number)
                first, end = rng.choices(self.lines[number]["runs"], weights=[b - a for a, b in self.lines[number]["runs"]])[0]
                # Three pages in ten start where the stretch does, five anywhere that leaves a full page, and two near
                # the stretch's end: those are short pages, and the only ones that show how a song ends.
                where, late = rng.random(), max(first, end - FULL)
                start = first if where < 0.3 else rng.randint(first, late) if where < 0.8 else rng.randint(late, max(late, end - MIN_RUN))
            count = min(end - start, rng.choice(PAGE))
            if sum(len(cell["events"]) for measure in line["measures"][start:start + count] for cell in measure.get("cells", [])) >= MIN_NOTES:
                break
        piece = window(line, start, count, rng)
        about = {"about": line["about"]} if line.get("about") else {}   # a corpus of many uploaders says whose score it is, and under which licence
        return piece, about | {"kind": "score", "corpus": self.name, "score": line["score"], "line": self.lines[number]["file"], "label": line["label"], "part": line["part"], "staff": line["staff"], "voice": line["voice"],
                       "from_measure": start, "measures": count, "measures_with_another_voice_left_out": sum(1 for at in line["cut"] if start <= at < start + count)}


def convert(musescore: Path, sources: list[Path], out: Path) -> None:
    """Has MuseScore write each score as MusicXML. It runs without a screen and takes its Qt platform from a variable of its own."""
    jobs = [{"in": str(source), "out": str(out / f"{source.stem}.musicxml")} for source in sources if not (out / f"{source.stem}.musicxml").exists()]
    if jobs:
        job = out / "job.json"
        job.write_text(json.dumps(jobs))
        done = subprocess.run([str(musescore), "-j", str(job)], env=os.environ | {"MU_QT_QPA_PLATFORM": "offscreen", "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"MuseScore stopped with {done.returncode}:\n{done.stderr[-2000:]}")


def read_or_say(paths: tuple[Path, Path | None]) -> list[dict] | str:
    path, source = paths
    try:
        return read_score(path, source)
    except Exception as failure:   # one score that cannot be read is a line in the index, not the end of the build
        return f"{path.stem}: {type(failure).__name__}: {failure}"


def build(out: Path, cache: Path, musescore: Path, workers: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"lieder-{COMMIT[:12]}.tar.gz"
    if not archive.exists():
        print(f"downloading {SOURCE} at {COMMIT[:12]}", flush=True)
        urllib.request.urlretrieve(f"https://codeload.github.com/OpenScore/Lieder/tar.gz/{COMMIT}", archive)
    scores, musicxml = cache / "mscx", cache / "musicxml"
    if not scores.exists():
        with tarfile.open(archive) as tar:
            members = [member for member in tar.getmembers() if member.name.endswith(".mscx") and "/scores/" in member.name]
            tar.extractall(scores, members=members, filter="data")
    sources = sorted(scores.rglob("*.mscx"))
    musicxml.mkdir(exist_ok=True)
    version = subprocess.run([str(musescore), "--version"], env=os.environ | {"MU_QT_QPA_PLATFORM": "offscreen", "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True).stdout.strip()
    print(f"{len(sources)} scores; converting with {version}", flush=True)
    convert(musescore, sources, musicxml)
    described = write_corpus(out, [musicxml / f"{source.stem}.musicxml" for source in sources], version, workers, sources)
    print(f"{len(described['lines'])} lines of {described['scores']} scores in {out}: {described['usable']} of {described['measures']} measures can be drawn; {len(described['unread'])} scores not read")
    for reason, count in described["turned_away"].items():
        print(f"  {count:7d}  {reason}")


def write_corpus(out: Path, paths: list[Path], converted_with: str, workers: int = 1, sources: list[Path] | None = None) -> dict:
    """Reads MusicXML scores into a corpus folder: a file for each line that has measures to draw, and `index.json`,
    which says where the scores are from, what each line holds and why measures were turned away. `sources` are the
    MuseScore files the MusicXML was written from, in the same order. Returns the index."""
    pairs = list(zip(paths, sources or [None] * len(paths)))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json.gz"):
        old.unlink()
    if workers > 1:
        with Pool(workers) as pool:
            read = pool.map(read_or_say, pairs, chunksize=8)
    else:
        read = [read_or_say(pair) for pair in pairs]
    index, unread, reasons, digest = [], [], Counter(), hashlib.sha256()
    measures = cut = 0
    for lines in read:
        if isinstance(lines, str):
            unread.append(lines)
            continue
        for line in lines:
            runs = runs_of(line["bad"])
            usable = [at for run in runs for at in range(*run)]
            reasons.update(reason for reason in line["bad"] if reason)
            if short := sum(1 for reason in line["bad"] if reason is None) - len(usable):
                reasons["in a stretch too short to make a page"] += short
            measures += len(line["bad"])
            cut += len(line["cut"])
            if not usable:
                continue
            name = f"{line['score']}-{len([other for other in index if other['score'] == line['score']]) + 1}.json.gz"
            data = json.dumps(line, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()
            digest.update(data)
            with gzip.GzipFile(out / name, "wb", mtime=0) as file:
                file.write(data)
            index.append({"file": name, "score": line["score"], "title": line["title"], "composer": line["composer"], "label": line["label"], "staff": line["staff"],
                          "measures": len(line["bad"]), "usable": len(usable), "runs": runs, "things": dict(Counter(name for at in usable for name in line["things"][at]))})
    described = {"name": "lieder", "source": SOURCE, "commit": COMMIT, "licence": "CC0 1.0 Universal", "converted_with": converted_with, "sha256": digest.hexdigest(), "scores": len(paths) - len(unread),
                 "measures": measures, "usable": sum(line["usable"] for line in index), "other_voices_left_out_in": cut, "turned_away": dict(reasons.most_common()), "unread": unread, "lines": index}
    (out / "index.json").write_text(json.dumps(described, ensure_ascii=False))
    return described


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("scores/lieder"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/sheet_synth/lieder")
    parser.add_argument("--musescore", type=Path, default=Path.home() / ".local/opt/musescore/squashfs-root/AppRun")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    build(arguments.out, arguments.cache, arguments.musescore, arguments.workers)


if __name__ == "__main__":
    main()
