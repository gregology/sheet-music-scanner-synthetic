"""A corpus of real scores from PDMX: scores people typed into MuseScore and marked as free of copyright.

PDMX (Long, Novack, Berg-Kirkpatrick and McAuley, 2024) is 254,077 scores from musescore.com, each marked public
domain or CC0 by whoever uploaded it, published as a dataset under CC BY 4.0. Unlike the Lieder corpus it holds
parts for every kind of instrument. `build` downloads its table and its MusicXML files from Zenodo, reads every line
of every chosen score as `scores.py` reads the Lieder corpus, and writes a corpus folder that `generate --scores`
takes. README.md, "Real scores".

Three things are left out before any page is cut:

- Music a reader is tested on: a score with a line that shares its notes with the test music (`tested.py`).
- A second copy of a song: only PDMX's own `deduplicated` subset is read.
- A score whose licence on the website and in the file disagree: 12% of PDMX. `--with-licence-conflicts` reads them too.

Usage: uv run python -m sheet_synth.pdmx [--out scores/pdmx] [--cache ~/.cache/sheet_synth/pdmx] [--tested scores/tested]
       [--with-licence-conflicts] [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import tested
from .scores import read_score, runs_of

RECORD = "15571083"   # Zenodo's number for the version of 2025-06-01, the one with the `license_conflict` column
SOURCE = f"https://zenodo.org/records/{RECORD}"
FILES = {"PDMX.csv": "30392ccf38bb63ce70e7afae70f9c88c", "mxl.tar.gz": "49ffd75ecf5489c0be6d41182eb11ff7"}   # MD5, as Zenodo lists them
LICENCE = "CC BY 4.0 for the dataset; each score is public domain or CC0, as its uploader marked it on musescore.com"

fingerprints: np.ndarray = np.array([], dtype=np.uint64)   # the test music's, in each worker


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        while block := file.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def chosen(table: Path, conflicts: bool) -> list[dict]:
    """The scores to read, in the table's order: one of each song, whole files, and no licence conflict unless asked for."""
    csv.field_size_limit(10**9)
    with open(table, newline="") as file:
        return [{"mxl": row["mxl"][2:], "musescore": int(re.search(r"/(\d+)\.json$", row["metadata"]).group(1)), "licence": row["license"], "licence_conflict": row["license_conflict"] == "True",
                 "title": row["song_name"] if row["song_name"] != "NA" else "", "composer": row["composer_name"] if row["composer_name"] != "NA" else None}
                for row in csv.DictReader(file)
                if row["subset:deduplicated"] == "True" and row["subset:all_valid"] == "True" and (conflicts or row["license_conflict"] != "True")]


def start(prints: np.ndarray) -> None:
    global fingerprints
    fingerprints = prints


def read_one(job: tuple[Path, Path, dict]) -> dict:
    """Reads one score and writes its lines. What comes back is small: the index's rows and the counts."""
    path, out, about = job
    try:
        lines = read_score(path)
    except Exception as failure:   # one score that cannot be read is a line in the index, not the end of the build
        return {"unread": f"{path.stem}: {type(failure).__name__}: {failure}"}
    most = max((tested.shared(line, fingerprints) for line in lines), default=0)
    if most >= tested.SHARED:
        return {"tested": {"score": path.stem, "musescore": about["musescore"], "title": about["title"], "composer": about["composer"], "shared": most}}
    rows, reasons, measures, cut = [], Counter(), 0, 0
    folder = out / path.parent.parent.name / path.parent.name
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
        line["title"] = line["title"] or about["title"]
        line["composer"] = line["composer"] or about["composer"]
        line["about"] = {"musescore": about["musescore"], "licence": about["licence"], "licence_conflict": about["licence_conflict"]}
        data = json.dumps(line, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()
        name = f"{folder.relative_to(out)}/{path.stem}-{len(rows) + 1}.json.gz"
        folder.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(out / name, "wb", mtime=0) as file:
            file.write(data)
        rows.append({"file": name, "score": line["score"], "title": line["title"], "composer": line["composer"], "label": line["label"], "staff": line["staff"], "sha256": hashlib.sha256(data).hexdigest(),
                     "measures": len(line["bad"]), "usable": len(usable), "runs": runs, "things": dict(Counter(name for at in usable for name in line["things"][at]))})
    return {"rows": rows, "reasons": reasons, "measures": measures, "cut": cut}


def write_corpus(out: Path, jobs: list[tuple[Path, dict]], prints: np.ndarray, described: dict, workers: int = 1) -> dict:
    """Reads the scores into a corpus folder and writes its `index.json`. Returns the index."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.rglob("*.json.gz"):
        old.unlink()
    work = [(path, out, about) for path, about in jobs]
    if workers > 1:
        with Pool(workers, initializer=start, initargs=(prints,)) as pool:
            read = list(pool.imap(read_one, work, chunksize=16))
    else:
        start(prints)
        read = [read_one(job) for job in work]
    index, unread, turned, reasons, digest = [], [], [], Counter(), hashlib.sha256()
    measures = cut = 0
    for result in read:
        if "unread" in result:
            unread.append(result["unread"])
        elif "tested" in result:
            turned.append(result["tested"])
        else:
            reasons.update(result["reasons"])
            measures += result["measures"]
            cut += result["cut"]
            for row in result["rows"]:
                digest.update(row.pop("sha256").encode())
                index.append(row)
    index_of = described | {"sha256": digest.hexdigest(), "scores": len(jobs) - len(unread) - len(turned), "measures": measures, "usable": sum(line["usable"] for line in index), "other_voices_left_out_in": cut,
                            "turned_away": dict(reasons.most_common()), "tested_music_left_out": turned, "unread": unread, "lines": index}
    (out / "index.json").write_text(json.dumps(index_of, ensure_ascii=False))
    return index_of


def build(out: Path, cache: Path, tested_folder: Path, conflicts: bool, limit: int | None, workers: int) -> None:
    prints = tested.load(tested_folder)
    cache.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = cache / name
        if not path.exists():
            print(f"downloading {name} from {SOURCE}", flush=True)
            urllib.request.urlretrieve(f"{SOURCE}/files/{name}?download=1", path)
        checked = cache / f"{name}.md5"
        if not checked.exists() or checked.read_text() != digest:
            if md5(path) != digest:
                raise SystemExit(f"{path} is not the file Zenodo lists: delete it and run again")
            checked.write_text(digest)
    scores = chosen(cache / "PDMX.csv", conflicts)[:limit]
    wanted = {score["mxl"] for score in scores if not (cache / score["mxl"]).exists()}
    if wanted:
        print(f"unpacking {len(wanted)} scores", flush=True)
        with tarfile.open(cache / "mxl.tar.gz") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if name in wanted:
                    (cache / name).parent.mkdir(parents=True, exist_ok=True)
                    (cache / name).write_bytes(tar.extractfile(member).read())
    jobs = [(cache / score["mxl"], score) for score in scores if (cache / score["mxl"]).exists()]
    print(f"{len(jobs)} of {len(scores)} chosen scores have a MusicXML file; reading them", flush=True)
    described = {"name": "pdmx", "source": SOURCE, "commit": RECORD, "licence": LICENCE, "converted_with": "PDMX's own MusicXML files", "subset": "deduplicated and all_valid" + ("" if conflicts else ", no licence conflict"),
                 "licence_conflicts_included": conflicts, "limit": limit, "tested_music": json.loads((tested_folder / "index.json").read_text())["sha256"]}
    index = write_corpus(out, jobs, prints, described, workers)
    print(f"{len(index['lines'])} lines of {index['scores']} scores in {out}: {index['usable']} of {index['measures']} measures can be drawn; {len(index['unread'])} scores not read; {len(index['tested_music_left_out'])} left out as music a reader is tested on")
    for reason, count in index["turned_away"].items():
        print(f"  {count:8d}  {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("scores/pdmx"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/sheet_synth/pdmx")
    parser.add_argument("--tested", type=Path, default=Path("scores/tested"), help="the test music's fingerprints, made by `python -m sheet_synth.tested`")
    parser.add_argument("--with-licence-conflicts", action="store_true", help="also read the scores whose licence on the website and in the file disagree")
    parser.add_argument("--limit", type=int, default=None, help="read only the first N chosen scores: for a trial")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    build(arguments.out, arguments.cache, arguments.tested, arguments.with_licence_conflicts, arguments.limit, arguments.workers)


if __name__ == "__main__":
    main()
