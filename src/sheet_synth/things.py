"""Names for what a page of music holds, so that pages can be asked for by name.

A reader's mistakes are sorted by what is drawn where they fall (the model's `tools/sort_mistakes.py`,
https://github.com/gregology/sheet-music-scanner-model),
and the names here are the ones that sorting uses wherever both speak of the same thing: `32nd note`,
`triplet, no mark`, `tied`. `generate --holds <name>` then makes only pages that hold the thing. A name is one of the
kinds below, or any token that is not a note, grace note or rest of one length: `clef-C3`, `time-6/8`, `key-5b`,
`multirest-12`. README.md, "Pages that hold a named thing".

Usage: uv run python -m sheet_synth.things [<corpus folder>] [--composed 200]

Prints the names. With a corpus folder, how many of its measures hold each; with `--composed N`, the share of N
composed pages that hold each. A thing the composer never writes can be asked of real scores alone: `--real 1`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

STEPS = "cdefgab"
CLEF_NOTE = {"G": 32, "F": 24, "C": 28}   # the note each clef's shape names, in steps above C0: G4, F3 and C4

KINDS = """
32nd note | 64th note | double dot
triplet, no mark | triplet, number only | triplet, bracket and number
sextuplet, no mark | sextuplet, number only | sextuplet, bracket and number
tuplet <n> in <m>, no mark | ... number only | ... bracket and number     (tuplet 2 in 3 is a duplet)
tied | tied over a barline | tie on a chord
chord | chord of <n>
accidental drawn | double sharp or flat
grace note | rest of whole measures
ledger lines above: <n> | ledger lines below: <n>
tremolo strokes | tremolo between two notes
mark: <stacc, acc, ten, marc, stacciss, upbow, dnbow, harm> | fermata | trill
clef: <G2, F4, C3, C4, G2_8vb> | key of <n> sharps or flats     (n from 4)
""".strip()


NAMES = [   # every name a page can be asked for, as patterns; `KINDS` says the same for people
    r"32nd note|64th note|double dot",
    r"(triplet|sextuplet|tuplet \d+ in \d+), (no mark|number only|bracket and number)",
    r"tied|tied over a barline|tie on a chord",
    r"chord( of \d+)?",
    r"accidental drawn|double sharp or flat|grace note|rest of whole measures",
    r"ledger lines (above|below): \d+",
    r"tremolo strokes|tremolo between two notes",
    r"mark: (stacc|acc|ten|marc|stacciss|upbow|dnbow|harm)|fermata|trill",
    r"clef: (G2|F4|C3|C4|G2_8vb)|key of [4-7] sharps or flats",
    r"(clef|time|key|multirest|barline|repeat)-\S+|barline|rest-measure",
]


def is_name(name: str) -> bool:
    """Whether a page can be asked for by this name. A misspelt name would otherwise be looked for in vain on every page."""
    return any(re.fullmatch(pattern, name) for pattern in NAMES)


def bottom_line(clef_token: str) -> int:
    """The note on the staff's bottom line under a clef token such as `clef-F4` or `clef-G2_8vb`, in steps above C0."""
    shape, line = clef_token[5], int(clef_token[6])
    return CLEF_NOTE[shape] - 2 * (line - 1) - (7 if clef_token.endswith("_8vb") else 0)


def tuplet_name(count: int, time_of: int, shown: str | None) -> str:
    name = {(3, 2): "triplet", (6, 4): "sextuplet"}.get((count, time_of), f"tuplet {count} in {time_of}")
    return f"{name}, {({'bracket': 'bracket and number', 'number': 'number only'}).get(shown, 'no mark')}"


def symbol_things(symbol: dict, bottom: int) -> list[str]:
    """What one note, rest or chord note of the truth holds. `bottom` is the staff's bottom line under the clef in force."""
    kind = symbol["type"]
    found = []
    if kind in ("measure-rest", "multirest"):
        return ["rest of whole measures"]
    if kind not in ("note", "grace", "rest"):
        return found
    if symbol.get("duration") in (32, 64):
        found.append(f"{symbol['duration']}{'nd' if symbol['duration'] == 32 else 'th'} note")
    if symbol.get("dots", 0) >= 2:
        found.append("double dot")
    if symbol.get("tuplet"):
        found.append(tuplet_name(*symbol["tuplet"], symbol.get("tuplet_shown")))
    if kind == "rest":
        return found
    place = STEPS.index(symbol["step"].lower()) + 7 * symbol["octave"] - bottom   # steps above the bottom line: 0 to 8 are on the staff
    if place >= 10:
        found.append(f"ledger lines above: {(place - 8) // 2}")
    elif place <= -2:
        found.append(f"ledger lines below: {-place // 2}")
    if symbol.get("accidental_shown"):
        found.append("accidental drawn")
    if abs(symbol.get("alter", 0)) == 2:
        found.append("double sharp or flat")
    if symbol.get("tie"):
        found.append("tied")
    if kind == "grace":
        found.append("grace note")
    found += [f"mark: {name}" for name in symbol.get("articulations") or []]
    if symbol.get("tremolo"):
        found.append("tremolo strokes")
    if symbol.get("rocked"):
        found.append("tremolo between two notes")
    found += [name for name in ("fermata", "trill") if symbol.get(name)]
    return found


def line_things(symbols: list[dict], bottom: int) -> tuple[set[str], int]:
    """What a run of the truth's symbols holds, and the bottom line in force after it."""
    found: set[str] = set()
    tied = False        # a tie is open from the last note or chord
    chord: list[dict] = []   # the notes of the chord being read: its top note, then the ones under it

    def close() -> None:
        if len(chord) > 1:
            found.update({"chord", f"chord of {len(chord)}"})
            if any(note.get("tie") for note in chord):
                found.add("tie on a chord")
        chord.clear()

    for symbol in symbols:
        kind = symbol["type"]
        token = symbol.get("token", "")
        if kind == "clef":
            bottom = bottom_line(token)
            found.add(f"clef: {token[5:]}")
        elif kind == "key" and abs(symbol["sharps"]) >= 4:
            found.add(f"key of {abs(symbol['sharps'])} sharps or flats")
        elif kind == "barline" and tied:
            found.add("tied over a barline")
        if token and not token.startswith(("note-", "grace-")) and not (token.startswith("rest-") and token != "rest-measure"):
            found.add(token)
        found.update(symbol_things(symbol, bottom))
        if kind == "note" and symbol.get("under"):
            chord.append(symbol)
            tied = tied or symbol.get("tie") in ("i", "m")
        elif kind in ("note", "rest", "measure-rest", "multirest"):
            close()
            tied = symbol.get("tie") in ("i", "m")
            if kind == "note":
                chord.append(symbol)
    close()
    return found, bottom


def page_things(truth: dict) -> set[str]:
    """What a page's truth holds: every name that `--holds` can ask for."""
    found: set[str] = set()
    bottom = bottom_line("clef-G2")
    for system in truth["systems"]:
        here, bottom = line_things(system["symbols"], bottom)
        found |= here
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", help="a corpus of real scores, made by `python -m sheet_synth.scores`")
    parser.add_argument("--composed", type=int, default=0, help="count what this many composed pages hold")
    arguments = parser.parse_args()
    print("A page can be asked for by any of these, or by a token that is not a note's or a rest's:\n")
    print(KINDS)
    if arguments.corpus:
        index = json.loads((arguments.corpus / "index.json").read_text())
        counts: dict[str, list[int]] = {}
        for line in index["lines"]:
            for name, count in line["things"].items():
                held = counts.setdefault(name, [0, 0])
                held[0] += 1
                held[1] += count
        print(f"\nIn {arguments.corpus}, of {len(index['lines'])} lines and {index['usable']} measures that can be drawn:\n")
        for name, (lines, measures) in sorted(counts.items(), key=lambda item: -item[1][1]):
            print(f"{measures:7d} measures in {lines:4d} lines  {name}")
    if arguments.composed:
        from .example import engraved_page   # here, not at the top: `example` imports this module

        pages = Counter()
        for index in range(arguments.composed):
            pages.update(page_things(engraved_page(0, index)[0].truth))
        print(f"\nOf {arguments.composed} composed pages:\n")
        for name, count in pages.most_common():
            print(f"{100 * count / arguments.composed:6.1f}%  {name}")


if __name__ == "__main__":
    main()
