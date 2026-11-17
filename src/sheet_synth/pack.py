"""Packs a generated dataset for Hugging Face: tar shards a loader can stream, and the dataset's card.

Usage: uv run python -m sheet_synth.pack <generated folder> <packed folder> [--name train]

Each shard of a thousand examples becomes `<name>-00000.tar`, holding `00000042.jpg` and `00000042.json` side by
side, which is the WebDataset layout Hugging Face reads without a loading script. The card is the folder's
`README.md`. Nothing is uploaded. README.md, "Publish a dataset".
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

CARD = """---
license: cc-by-4.0
task_categories:
- image-to-text
- object-detection
tags:
- music
- sheet-music
- optical-music-recognition
- synthetic
pretty_name: Sheet Music Scanner synthetic photos
size_categories:
- {size}
configs:
- config_name: default
  data_files:
  - split: train
    path: "{name}-*.tar"
---

# Sheet Music Scanner synthetic photos

{count:,} synthetic phone photos of printed sheet music for single-line instruments, each with its exact score and
the position of everything on the page. Made to train a model that reads a phone photo of sheet music into a score.

Every photo is generated: a made-up piece, or a page cut from a real score, is engraved, marked by hand, printed on paper, bent, laid on a surface in
a 3D scene, lit, and photographed by a simulated iPhone camera. Because the music is the generator's own data before it's
drawn, the answer to each photo is known exactly; no label was made by a recogniser or a person.

- **Generator, and the full description of every field:** https://github.com/gregology/sheet-music-scanner-synthetic
- **The model trained on it:** https://huggingface.co/gregology/sheet-music-scanner
- **Photo:** {width} x {height} px JPEG.
- **Generator version:** {commits}. **Seed:** {seed}. Any example can be made again from the seed and its number.
- **Numbers with no example:** {failed:,}, each turned away by one of the generator's checks.

## An example

`00000042.jpg` is the photo and `00000042.json` its truth:

- `systems`: each line of music, top to bottom, with its `tokens` (clef, key, time signature, notes, rests,
  barlines, left to right), its `symbols` with their boxes, the other drawn `marks`, and the path of its five staff
  lines in the photo.
- `handwriting`: pencil and pen marks, among them finger numbers with the note each belongs to.
- `page`: the page's outline in the photo and a lattice that maps any page position to a pixel, so the photo can
  be flattened.
- `paper`, `sheet`, `room`, `shot`, `camera`: every choice the generator made: paper and print, the sheet's bend,
  the surface and light, the camera's angle, and the light level, noise, blur and JPEG quality.

## What it covers

- Five clefs, keys up to six sharps or flats, eleven time signatures, changes of key, time and clef, ties, slurs,
  articulation, grace notes, triplets, repeats, lyrics, chord names, printed and handwritten finger numbers.
- Five music fonts, two page sizes, staff heights from 5.7 to 10.2 mm.
- Flat, bowed, wavy and folded sheets and book pages; daylight through a window, ceiling lights, lamps; the
  photographer's shadow; light from bright to too dark to read.

## What it does not

- **One engraver.** Every page is laid out by Verovio. The fonts change the shapes of the signs, not their spacing.
{music}
- **One staff, one voice.** No piano music as a pianist reads it; chords only as one stem's notes.
- **The darkest and most blurred photos are meant to be unreadable,** so that a model can learn to ask for a better
  one. Filter on `camera.light`, `camera.shake_px` and `page.in_photo` for photos a reader should manage.
- **Trained on, and tested on real photos.** The model trained on these photos reads real photos of printed music;
  its card gives the numbers and the test sets: https://huggingface.co/gregology/sheet-music-scanner

## Licence and credit

Creative Commons Attribution 4.0. Use it for anything, model training and commercial use included, and credit
Greg Clarke. The generator's code is MIT.
{credit}"""

COMPOSED = "- **No real music.** The tunes are random walks with the notation of real music and not its habits."
REAL = "- **Real music on {share:.0%} of the photos,** one line of a real score each; the rest are made-up tunes, random walks with the notation of real music and not its habits. A truth's `source` says which."
# What a dataset owes for the real scores in it, by the corpus's name. PDMX is under CC BY 4.0, which asks for this credit wherever its scores go.
CREDITS = {
    "lieder": "- The [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder), under CC0 1.0: no rights reserved, no credit required.",
    "pdmx": ("- [PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing](https://zenodo.org/records/15571083), by Phillip Long, Zachary Novack, Taylor Berg-Kirkpatrick and Julian McAuley "
             "(ICASSP 2025), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its scores were typed into musescore.com by its users, who marked each as public domain or CC0. Changed here: one "
             "line of a score is read, cut to a page and engraved anew. A truth's `source.about` gives the score's number on musescore.com and the licence its uploader gave it."),
}


def credit(settings: dict) -> tuple[str, str]:
    """The card's line about the music and its credits for real scores, from a dataset's settings."""
    corpora = settings.get("scores") or []
    names = sorted({name.split("@")[0] for name in ([corpora] if isinstance(corpora, str) else corpora)})
    if not names:
        return COMPOSED, ""
    return REAL.format(share=settings["real"]), "\nThe real scores are from:\n\n" + "\n".join(CREDITS[name] for name in names) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generated", type=Path)
    parser.add_argument("packed", type=Path)
    parser.add_argument("--name", default="train")
    arguments = parser.parse_args()
    arguments.packed.mkdir(parents=True, exist_ok=True)
    settings = json.loads((arguments.generated / "dataset.json").read_text())
    count = failed = 0
    for shard in sorted(folder for folder in arguments.generated.iterdir() if folder.is_dir() and folder.name.isdigit()):
        failed += len(list(shard.glob("*.failed")))
        truths = sorted(shard.glob("*.json"))   # an example is whole only once its truth is written
        if not truths:
            continue
        with tarfile.open(arguments.packed / f"{arguments.name}-{shard.name}.tar", "w") as tar:
            for truth in truths:
                tar.add(truth.with_suffix(".jpg"), arcname=f"{truth.stem}.jpg")
                tar.add(truth, arcname=truth.name)
        count += len(truths)
        print(f"{shard.name}: {len(truths)} examples")
    size = next(label for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (10**9, "100K<n<1M")) if count < limit)
    music, credits = credit(settings)
    (arguments.packed / "README.md").write_text(CARD.format(music=music, credit=credits, size=size, name=arguments.name, count=count, failed=failed, width=settings["width"], height=settings["height"], seed=settings["seed"], commits=", ".join(settings.get("commits", ["unknown"]))))
    print(f"{count} examples packed into {arguments.packed}")


if __name__ == "__main__":
    main()
