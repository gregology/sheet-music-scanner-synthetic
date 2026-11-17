"""The music a reader is tested on, as fingerprints, so that no page is drawn from it.

The model's survey and the experiment's real sheets are parts of OpenScore's string quartets. A large corpus of
scores people have typed in holds other people's copies of the same quartets, and arrangements of them, under other
titles. So a score is not left out by its title. Each line's notes are turned into fingerprints, and a line that
shares enough of them with the test music is a copy of it, whoever typed it and in whatever key.

A fingerprint is `RUN` steps in a row: from each note to the next, how far the pitch moves and how the length
changes. It is the same in every key and at every speed. Rests and grace notes are left out and tied notes are one
note, since two people type those differently. README.md, "Music a reader is tested on".

Usage: uv run python -m sheet_synth.tested [--out scores/tested] [--cache ~/.cache/sheet_synth/quartets] [--workers 8]

Downloads every score of OpenScore's string quartets at the commit the tests take them at, and writes
`fingerprints.npy` and `index.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.parse
import urllib.request
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .scores import event_length, measure_from, read_score

SOURCE = "https://github.com/OpenScore/StringQuartets"
COMMIT = "2614b0b2958dd1ce3f52e172a43e5f4d0e29570f"   # the commit `experiment/make_real_sheets.py` and the survey take their scores at
RAW = "https://raw.githubusercontent.com/OpenScore/StringQuartets"
RUN = 12      # steps in one fingerprint: thirteen notes in a row
VARIED = 5    # a fingerprint counts only if it has this many different steps: a scale or a repeated note is everybody's
SHARED = 12   # a line with this many different fingerprints of the test music in a row is a copy of it: 24 notes that move and last alike
SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def melody(line: dict) -> list[tuple[int, Fraction]]:
    """A stored line's notes in order, each as its pitch in semitones and its length in quarter notes: a chord's top
    note, no rests, no grace notes, and notes tied together as one."""
    notes: list[tuple[int, Fraction]] = []
    tied = False   # the note before has a tie towards this one
    for data in line["measures"]:
        for cell in measure_from(data).cells:
            for event in cell.events:
                if event.grace:
                    continue
                if event.rest:
                    tied = False
                    continue
                pitch = 12 * (event.di // 7) + SEMITONES[event.di % 7] + event.alter
                length = event_length(event, cell)
                if tied and notes and notes[-1][0] == pitch and event.tie in ("t", "m"):
                    notes[-1] = (pitch, notes[-1][1] + length)
                else:
                    notes.append((pitch, length))
                tied = event.tie in ("i", "m")
    return notes


def prints_at(notes: list[tuple[int, Fraction]]) -> list[int | None]:
    """For each run of `RUN` steps in a line's notes, its fingerprint, or None for a run too plain to be somebody's."""
    steps = [(b[0] - a[0], (b[1] / a[1]).numerator, (b[1] / a[1]).denominator) for a, b in zip(notes, notes[1:]) if a[1] and b[1]]
    found: list[int | None] = []
    for at in range(len(steps) - RUN + 1):
        run = steps[at:at + RUN]
        found.append(int.from_bytes(hashlib.blake2b(repr(run).encode(), digest_size=8).digest(), "big") if len(set(run)) >= VARIED else None)
    return found


def fingerprints(notes: list[tuple[int, Fraction]]) -> np.ndarray:
    """The fingerprints of a line's notes."""
    return np.array([found for found in prints_at(notes) if found is not None], dtype=np.uint64)


def shared(line: dict, tested: np.ndarray) -> int:
    """The longest stretch a line has in common with the test music: how many different fingerprints in a row are the
    test music's. A figure played over and over is one fingerprint however often it comes, so an accompaniment that
    happens to be in a quartet too does not count as a copy. A run too plain to have a fingerprint neither counts
    nor ends the stretch. `tested` is sorted."""
    at_each = prints_at(melody(line))
    mine = np.array([found for found in at_each if found is not None], dtype=np.uint64)
    if not len(mine) or not len(tested):
        return 0
    at = np.searchsorted(tested, mine)
    known = iter((tested[np.minimum(at, len(tested) - 1)] == mine).tolist())
    longest, stretch = 0, set()
    for found in at_each:
        if found is None:
            continue
        if next(known):
            stretch.add(found)
            longest = max(longest, len(stretch))
        else:
            stretch = set()
    return longest


def is_tested(line: dict, tested: np.ndarray) -> bool:
    return shared(line, tested) >= SHARED


def load(folder: Path) -> np.ndarray:
    path = Path(folder) / "fingerprints.npy"
    if not path.exists():
        raise FileNotFoundError(f"no fingerprints of the test music in {folder}: make them with `python -m sheet_synth.tested --out {folder}`")
    return np.load(path)


def of_score(path: Path) -> tuple[str, int, np.ndarray] | str:
    try:
        lines = read_score(path)
    except Exception as failure:
        return f"{path.stem}: {type(failure).__name__}: {failure}"
    found = [fingerprints(melody(line)) for line in lines]
    return path.stem, len(lines), np.concatenate(found) if found else np.array([], dtype=np.uint64)


def build(out: Path, cache: Path, workers: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    listed = cache / "files.json"
    if not listed.exists():   # every file of the repository at that commit; a score's file is not always named by its number in `scores.tsv`
        urllib.request.urlretrieve(f"https://api.github.com/repos/OpenScore/StringQuartets/git/trees/{COMMIT}?recursive=1", listed)
    tree = json.loads(listed.read_text())
    assert not tree.get("truncated"), "GitHub cut the list of files short"
    wanted = sorted(item["path"] for item in tree["tree"] if item["path"].startswith("scores/") and item["path"].endswith(".mxl"))
    with_one = {str(Path(path).parent) for path in wanted}
    missing = sorted({str(Path(item["path"]).parent) for item in tree["tree"] if item["path"].endswith((".mscx", ".mscz"))} - with_one)
    folder = cache / "mxl"
    folder.mkdir(exist_ok=True)
    for where in wanted:
        path = folder / Path(where).name
        if not path.exists():
            urllib.request.urlretrieve(f"{RAW}/{COMMIT}/" + urllib.parse.quote(where), path)
    paths = [folder / Path(where).name for where in wanted]
    with Pool(workers) as pool:
        read = pool.map(of_score, paths, chunksize=4)
    unread = [item for item in read if isinstance(item, str)]
    good = [item for item in read if not isinstance(item, str)]
    prints = np.unique(np.concatenate([found for _, _, found in good]))
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.save(buffer, prints)
    (out / "fingerprints.npy").write_bytes(buffer.getvalue())
    described = {"source": SOURCE, "commit": COMMIT, "run": RUN, "varied": VARIED, "shared": SHARED, "scores": [name for name, _, _ in good], "lines": sum(count for _, count, _ in good),
                 "fingerprints": len(prints), "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(), "scores_with_no_musicxml": missing, "unread": unread}
    (out / "index.json").write_text(json.dumps(described, ensure_ascii=False, indent=1))
    print(f"{len(prints)} fingerprints of {described['lines']} lines of {len(good)} scores in {out}; {len(missing)} scores have no MusicXML in the repository, {len(unread)} were not read")
    for line in missing + unread:
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("scores/tested"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/sheet_synth/quartets")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    build(arguments.out, arguments.cache, arguments.workers)


if __name__ == "__main__":
    main()
