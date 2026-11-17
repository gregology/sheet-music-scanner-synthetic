"""Writes examples to disk: `<out>/<shard>/<number>.jpg` and `.json`, a thousand to a shard.

Usage: uv run python -m sheet_synth.generate --out out/v1 --seed 1 --count 1000 [--start 0] [--width 3024 --height 4032]
       [--samples 32] [--workers 1] [--device CPU|OPTIX|CUDA] [--real 0.5 [--scores scores/lieder ...]] [--holds <name> ...]

`--real` is the share of examples whose music is a real score from the corpus in `--scores`; the rest are composed.
With several corpora, each measure of real music is as likely as any other, whichever corpus it is in.
`--holds` names something every page must hold (`python -m sheet_synth.things` lists the names).

An example already on disk is skipped, so a run that stops can be started again with the same arguments. One that
fails a check leaves `<number>.failed` in place of its photo and truth. With `--workers N` the numbers are shared out
among N processes. README.md, "Generate a dataset".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FORMAT = 3      # the truth's layout; raise it when a field changes meaning. 2: chords, and grace notes that show accidentals. 3: `source`; real scores, with ties on a chord's notes, tremolos between two notes and any tuplet
SHARD = 1000
GPU_BUSY = 75       # a worker's exit code when something else has the GPU
WAIT_FOR_GPU = 60   # seconds before it is started again
GPU_HEADROOM = 6000  # MiB that must be free before a render starts; a render takes about 3,000


def gpu_free() -> int:
    """MiB of GPU memory nobody is using."""
    asked = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(asked.stdout.split()[0]) if asked.returncode == 0 and asked.stdout.strip() else 10**6


def paths(out: Path, index: int) -> tuple[Path, Path]:
    folder = out / f"{index // SHARD:05d}"
    return folder / f"{index:08d}.jpg", folder / f"{index:08d}.json"


def failed_path(truth_path: Path) -> Path:
    """Where an example that failed a check records why, in place of its photo and truth."""
    return truth_path.with_suffix(".failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1, help="the dataset's seed: with an example's number it fixes the example")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--width", type=int, default=3024, help="the photo's size; the default is an iPhone's 12 megapixel photo")
    parser.add_argument("--height", type=int, default=4032)
    parser.add_argument("--samples", type=int, default=32, help="light paths traced for each pixel")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default=os.environ.get("SHEET_SYNTH_DEVICE", "CPU"), choices=["CPU", "OPTIX", "CUDA"])
    parser.add_argument("--real", type=float, default=0.0, help="the share of examples whose music is a real score")
    parser.add_argument("--scores", type=Path, nargs="+", default=[Path("scores/lieder")], help="the corpus of real scores, made by `python -m sheet_synth.scores` or `python -m sheet_synth.pdmx`; several may be given")
    parser.add_argument("--holds", action="append", default=[], help="a thing every page must hold; may be given more than once")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    if not 0 <= arguments.real <= 1:
        parser.error("--real is a share, from 0 to 1")
    from .things import is_name

    for name in arguments.holds:
        if not is_name(name):
            parser.error(f"--holds: nothing is named {name!r}; `python -m sheet_synth.things` lists the names")

    # Written once, before any worker starts: two workers writing it at once left one reading half a file.
    settings = {"format": FORMAT, "seed": arguments.seed, "width": arguments.width, "height": arguments.height, "samples": arguments.samples}
    if arguments.real:
        from .scores import Corpus   # which corpus, by its commit and its contents: other scores are another dataset
        names = [Corpus(folder).name for folder in arguments.scores]
        settings |= {"real": arguments.real, "scores": names[0] if len(names) == 1 else names}
    if arguments.holds:
        settings |= {"holds": arguments.holds}
    described = arguments.out / "dataset.json"
    commit = os.environ.get("SHEET_SYNTH_COMMIT", "unknown")   # the generator's version, given by whoever starts the run
    if arguments.worker is None:
        if described.exists():
            before = json.loads(described.read_text())
            keys = set(settings) | {"real", "scores", "holds"}
            assert {key: before.get(key) for key in keys} == {key: settings.get(key) for key in keys}, f"{described} was made with other settings"
            if commit not in before["commits"]:
                print(f"this run's generator ({commit}) differs from the one the dataset was started with ({before['commits']})", file=sys.stderr)
                described.write_text(json.dumps(before | {"commits": before["commits"] + [commit]}, indent=1))
        else:
            described.write_text(json.dumps(settings | {"commits": [commit]}, indent=1))

    if arguments.worker is None and (arguments.workers > 1 or arguments.device != "CPU"):
        # Each worker is a process of its own. One that finds the GPU taken stops with GPU_BUSY and is started again a
        # minute later, in a fresh process, since a renderer that has run out of GPU memory may not recover in place.
        def start(worker: int) -> subprocess.Popen:
            return subprocess.Popen([sys.executable, "-m", "sheet_synth.generate", *sys.argv[1:], "--worker", str(worker)])

        running = {worker: start(worker) for worker in range(arguments.workers)}
        worst = 0
        while running:
            time.sleep(5)
            for worker, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                if code == GPU_BUSY:
                    time.sleep(WAIT_FOR_GPU)
                    running[worker] = start(worker)
                else:
                    worst = max(worst, code)
                    del running[worker]
        sys.exit(worst)

    os.environ["SHEET_SYNTH_DEVICE"] = arguments.device
    from .example import Music, make_example   # after the device is chosen: this loads Blender

    folders = tuple(str(folder) for folder in arguments.scores)
    music = Music(arguments.real, (folders[0] if len(folders) == 1 else folders) if arguments.real else None, tuple(arguments.holds))

    worker = arguments.worker or 0
    work = arguments.out / f".work-{worker}"
    work.mkdir(exist_ok=True)
    made = 0
    began = time.monotonic()
    for index in range(arguments.start, arguments.start + arguments.count):
        if index % arguments.workers != worker:
            continue
        photo_path, truth_path = paths(arguments.out, index)
        if truth_path.exists():
            continue
        photo_path.parent.mkdir(exist_ok=True)
        # Short of memory, a render can lose only its denoising and still be saved, noisier than the rest. So a render
        # does not start unless there is room to finish.
        waited = False
        while arguments.device != "CPU" and gpu_free() < GPU_HEADROOM:
            if not waited:
                print(f"{index}: worker {worker} is waiting for {GPU_HEADROOM} MiB of GPU memory to be free", file=sys.stderr, flush=True)
            waited = True
            time.sleep(30)
        try:
            photo, truth = make_example(arguments.seed, index, arguments.width, arguments.height, work, arguments.samples, music)
        except Exception as failure:
            if arguments.device != "CPU" and any(word in str(failure) for word in ("memory", "CUDA", "OPTIX", "came back empty")):
                # Something else has the GPU: a language model sharing the card takes 20 GB of its 24. The example
                # is not skipped. This worker stops, and the run starts it again in a minute to try the same number.
                print(f"{index}: the GPU is busy ({failure}); worker {worker} will try again in {WAIT_FOR_GPU} s", file=sys.stderr, flush=True)
                sys.exit(GPU_BUSY)
            print(f"{index}: failed: {type(failure).__name__}: {failure}", file=sys.stderr, flush=True)   # one bad example must not end a run of thousands
            # Recorded beside the examples, so whoever reads the dataset can tell a number that failed from one not yet
            # made. Training that waited for a shard's thousandth example waited for good. A run started again tries it.
            failed_path(truth_path).write_text(f"{type(failure).__name__}: {failure}\n")
            continue
        photo_path.write_bytes(photo)
        partial = truth_path.with_suffix(".part")
        partial.write_text(json.dumps(truth, separators=(",", ":")))
        partial.rename(truth_path)   # the truth appears last and whole: its presence marks the example done
        failed_path(truth_path).unlink(missing_ok=True)
        made += 1
        if made % 10 == 0 or made == 1:
            print(f"worker {worker}: {made} made, {(time.monotonic() - began) / made:.1f} s each, last {index}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
