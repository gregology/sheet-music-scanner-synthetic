"""Asks whether a reader trained on real photos can read ours: runs homr on examples and scores it against the truth.

Usage: uv run python checks/read_with_homr.py <folder of examples> [--count 40]

homr reads real phone photos of printed music almost without error. If it reads these photos as well, they are
within what a photo-trained reader takes for a photo; if it fails where the light is good, something about them is
unlike the real thing. This is a check on the pictures. It makes no labels: the truth stays our own.

Scored as the model's test sets score a reading: the fewest notes to insert, delete or replace to
turn homr's notes into the true ones, as a share of the true notes. Rests and grace notes are left out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from music21 import converter, note, stream
from rapidfuzz.distance import Levenshtein


def true_notes(truth: dict) -> list[tuple[str, Fraction]]:
    notes = []
    for system in truth["systems"]:
        for symbol in system["symbols"]:
            if symbol["type"] == "note":
                length = Fraction(4, symbol["duration"]) * (2 - Fraction(1, 2 ** symbol["dots"]))
                if symbol["tuplet"]:
                    length *= Fraction(symbol["tuplet"][1], symbol["tuplet"][0])
                notes.append((symbol["pitch"].replace("b", "-"), length))
    return notes


def read_notes(path: Path) -> list[tuple[str, Fraction]]:
    score = converter.parse(path, format="musicxml")
    notes = []
    for part in score.parts:
        for element in part.recurse().getElementsByClass(note.Note):
            if not element.duration.isGrace:
                notes.append((element.pitch.nameWithOctave, Fraction(element.duration.quarterLength).limit_denominator(96)))
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("examples", type=Path)
    parser.add_argument("--count", type=int, default=40)
    arguments = parser.parse_args()
    arguments.examples = arguments.examples.resolve()   # homr runs from another folder, where a relative path would not be found
    rows = []
    for truth_path in sorted(arguments.examples.rglob("*.json"))[:arguments.count]:
        if truth_path.name == "dataset.json":
            continue
        truth = json.loads(truth_path.read_text())
        truth_notes = true_notes(truth)
        if not truth_notes:
            continue
        with tempfile.TemporaryDirectory(dir=arguments.examples) as work:
            copy = Path(work) / "photo.jpg"
            shutil.copyfile(truth_path.with_suffix(".jpg"), copy)
            done = subprocess.run(["uvx", "--python", "3.12", "--from", "homr[cpu]", "homr", str(copy)], capture_output=True, cwd=work)
            read = Path(work) / "photo.musicxml"
            if done.returncode:
                print(f"{truth_path.stem}: homr failed: {done.stderr.decode()[-200:]}", file=sys.stderr)
            try:
                found = read_notes(read) if done.returncode == 0 and read.exists() else []
            except Exception:
                found = []
        notes = Levenshtein.distance(truth_notes, found) / len(truth_notes)
        pitch = Levenshtein.distance([p for p, _ in truth_notes], [p for p, _ in found]) / len(truth_notes)
        camera, shot = truth["camera"], truth["shot"]
        row = {"example": truth_path.stem, "notes": round(100 * notes, 1), "pitch": round(100 * pitch, 1), "true": len(truth_notes), "read": len(found), "light": camera["light"], "shake": camera["shake_px"],
               "defocus": camera["defocus_px"], "shape": truth["sheet"]["shape"], "tilt": shot["tilt_deg"], "in_photo": truth["page"]["in_photo"], "clef": truth["systems"][0]["tokens"][0], "level": truth["level"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    for light in ("bright", "indoor", "dim", "dark"):
        group = [row for row in rows if row["light"] == light]
        if group:
            print(f"{light}: {len(group)} examples, notes wrong: median {sorted(r['notes'] for r in group)[len(group) // 2]}%, mean {sum(r['notes'] for r in group) / len(group):.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
