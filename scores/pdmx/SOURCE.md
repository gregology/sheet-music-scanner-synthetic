# scores/pdmx

Lines of real music for the generator's second source, from scores people typed into MuseScore ([README](../../README.md#pdmx)). The files here are made, not committed: `uv run python -m sheet_synth.pdmx` downloads PDMX's table and its MusicXML files and writes one `.json.gz` for each line and `index.json`. It needs the test music's fingerprints first: `uv run python -m sheet_synth.tested`.

| | |
|---|---|
| Source | [PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing](https://zenodo.org/records/15571083), version 9 of 2025-06-01, DOI [10.5281/zenodo.15571083](https://doi.org/10.5281/zenodo.15571083) |
| Made by | Phillip Long, Zachary Novack, Taylor Berg-Kirkpatrick and Julian McAuley, University of California San Diego. Paper: ICASSP 2025, [arXiv:2409.10831](https://arxiv.org/abs/2409.10831) |
| Files used | `PDMX.csv`, 225.4 MB, MD5 `30392ccf38bb63ce70e7afae70f9c88c`; `mxl.tar.gz`, 1.9 GB, MD5 `49ffd75ecf5489c0be6d41182eb11ff7`. Both are checked against these sums before use |
| Licence of the dataset | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as the Zenodo record gives it: re-use is allowed on the condition that the makers are credited |
| Licence of each score | Its uploader marked it public domain (`publicdomain`, 210,364 of 254,077) or CC0 (`cc-zero`, 43,713) on musescore.com; the table's `license` column. For 31,221 (12.3%) the licence the website shows and the one inside the file disagree (`license_conflict`). PDMX's makers follow the website's and recommend leaving the conflicts out |
| Read | The `deduplicated` subset, one arrangement of each song, with `all_valid` files: 102,596 scores. Of those, the 25,275 with a licence conflict are left out unless `--with-licence-conflicts` is given: 77,321 scores are read |
| Left out | Every score with a line that shares a stretch of notes with the music a reader is tested on ([`scores/tested`](../tested/SOURCE.md)), whatever its title |
| Checked | 2026-11-15: the record, its licence, its files and their sums, from the Zenodo page |
| Conflicts | Left out by default, as PDMX's makers recommend. I generated datasets v2 and v3 with them in (`--with-licence-conflicts`), trusting each uploader's marking; no score is left out for its title naming a game, film or song. Those pages are counted in the [README](../../README.md#the-datasets) and left out of anything I publish |

**The credit a dataset made with it must carry,** on its card ([`pack.py`](../../src/sheet_synth/pack.py) writes it):

> Real scores from PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing, by Phillip Long, Zachary Novack, Taylor Berg-Kirkpatrick and Julian McAuley (ICASSP 2025), https://zenodo.org/records/15571083, under CC BY 4.0. Changed: one line of a score is read, cut to a page and engraved anew.

Each page's truth names its score: `source.about` gives the score's number on musescore.com, the licence its uploader gave it, and whether that licence is in conflict.

`index.json` records the record, the subset, whether conflicts were read, a hash of every line, the hash of the test music's fingerprints it was built against, the scores left out as test music with their titles, how many measures can be drawn, and why the rest were turned away. A dataset made from the corpus records it by that record and hash.
