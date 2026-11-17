"""Asks whether the pages cut from real scores say what the scores say: reads the same measures of the same files
with music21 and compares.

Usage: uv run python checks/read_scores_with_music21.py [--scores scores/lieder] [--musicxml ~/.cache/sheet_synth/lieder/musicxml]
       [--seed 11] [--count 100]
       uv run python checks/read_scores_with_music21.py --scores scores/pdmx --musicxml ~/.cache/sheet_synth/pdmx/mxl

`scores.py` reads MusicXML with a reader of its own, so that the truth is our own data. This holds that reader to a
second one that shares no code with it. For each page it compares, note by note, the pitch, the length as played and
which notes are a chord's. It is a check on the reading of files. It makes no labels and looks at no picture.
"""

from __future__ import annotations

import argparse
import re
from fractions import Fraction
from pathlib import Path

from music21 import chord, converter, harmony, note, stream

from sheet_synth.example import Music, engraved_page
from sheet_synth.scores import parse

PITCH = re.compile(r"([A-G])(#*|b*)(\d+)")


def length(text: str) -> Fraction:
    """A token's length in quarter notes: `8.`, `8:3`, `8:2/3`."""
    text, _, tuplet = text.partition(":")
    value = Fraction(4, int(text.rstrip("."))) * (2 - Fraction(1, 2 ** text.count(".")))
    if tuplet:
        count, _, time_of = tuplet.partition("/")
        value = value * (int(time_of) if time_of else 2 ** (int(count).bit_length() - 1)) / int(count)
    return value


def ours(tokens: list[str]) -> list[tuple[str, Fraction, bool]]:
    """(pitch or "rest", length as played, under the note before) for each note and rest of a page's tokens."""
    groups: list[list[tuple[str, Fraction, bool]]] = []   # a note with the chord notes under it, or a rest
    joined: list[int] = []   # a tremolo stands between the group before this place and the one at it
    measure, under = Fraction(4), False
    for token in tokens:
        kind, _, rest = token.partition("-")
        if kind == "time":
            count, _, unit = rest.partition("/")
            measure = Fraction(4 * int(count), int(unit)) if unit else Fraction(4)
        elif token == "chord":
            under = True
        elif token == "tremolo":
            joined.append(len(groups))
        elif kind in ("note", "grace"):
            pitch, _, long = rest.partition("_")
            letter, sign, octave = PITCH.fullmatch(pitch).groups()
            entry = (letter + sign.replace("b", "-") + octave, Fraction(0) if kind == "grace" else length(long), under)
            if under:
                groups[-1].append(entry)
            else:
                groups.append([entry])
            under = False
        elif kind == "rest":
            groups.append([("rest", measure if rest == "measure" else length(rest), False)])
        elif kind == "multirest":
            groups += [[("rest", measure, False)] for _ in range(int(rest))]
    for place in joined:   # the two notes or chords a tremolo joins are each played for half their printed length
        for at in (place - 1, place):
            if 0 <= at < len(groups):
                groups[at] = [(pitch, value / 2, low) for pitch, value, low in groups[at]]
    return [entry for group in groups for entry in group]


def theirs(path: Path, part: int, staff: int, voice: str, start: int, count: int) -> list[tuple[str, Fraction, bool]]:
    score = converter.parse(path)
    staves = []   # music21 lists a part's staves as parts of their own
    for number, xml in enumerate(parse(path).findall("part")):
        staves += [(number, each) for each in range(1, max([int(sign.text) for sign in xml.iter("staves")] or [1]) + 1)]
    out = []
    for measure in list(list(score.parts)[staves.index((part, staff))].getElementsByClass(stream.Measure))[start:start + count]:
        voices = list(measure.voices)
        chosen = next((each for each in voices if str(each.id) == str(voice)), None) if voices else measure
        for event in ([] if chosen is None else chosen.notesAndRests):
            if isinstance(event, harmony.Harmony):   # a chord name over the staff, which music21 counts as a chord of no length
                continue
            value = Fraction(0) if event.duration.isGrace else Fraction(event.duration.quarterLength).limit_denominator(1000)
            if isinstance(event, note.Rest):
                out.append(("rest", value, False))
            elif isinstance(event, chord.Chord):
                pitches = sorted(event.pitches, key=lambda pitch: (pitch.diatonicNoteNum, pitch.alter), reverse=True)
                out += [(pitch.nameWithOctave, value, place > 0) for place, pitch in enumerate(pitches)]
            else:
                out.append((event.pitch.nameWithOctave, value, False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=Path("scores/lieder"))
    parser.add_argument("--musicxml", type=Path, default=Path.home() / ".cache/sheet_synth/lieder/musicxml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--count", type=int, default=100)
    arguments = parser.parse_args()
    music = Music(1.0, str(arguments.scores))
    agree = differ = unread = 0
    for index in range(arguments.count):
        page, _, source = engraved_page(arguments.seed, index, music)
        tokens = [token for system in page.truth["systems"] for token in system["tokens"]]
        mine = ours(tokens)
        try:
            path = arguments.musicxml / f"{source['score']}.musicxml"
            if not path.exists():   # PDMX keeps its scores packed, in the folders its lines are in
                path = arguments.musicxml / Path(source["line"]).parent / f"{source['score']}.mxl"
            other = theirs(path, source["part"], source["staff"], source["voice"], source["from_measure"], source["measures"])
        except Exception as failure:   # music21 cannot read a few of the files; that is its fault to report, not the page's
            unread += 1
            print(f"{index}: music21 could not read {source['score']}: {type(failure).__name__}")
            continue
        # The page holds the first of the window's measures; a tie cut at the page's end changes no pitch or length.
        wrong = [(place, mine[place], other[place]) for place in range(min(len(mine), len(other))) if mine[place] != other[place]]
        if wrong or len(mine) > len(other):
            differ += 1
            place, a, b = wrong[0] if wrong else (len(other), mine[len(other)], None)
            print(f"{index}: {source['score']} {source['label']} staff {source['staff']} from measure {source['from_measure']}: note {place} is {a} here and {b} to music21")
        else:
            agree += 1
    print(f"{agree} pages agree, {differ} differ, {unread} that music21 could not read")


if __name__ == "__main__":
    main()
