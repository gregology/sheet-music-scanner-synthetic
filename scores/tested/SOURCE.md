# scores/tested

Fingerprints of the music a reader is tested on, so that no page is drawn from it ([README](../../README.md#music-a-reader-is-tested-on)). The files here are made, not committed: `uv run python -m sheet_synth.tested` downloads the scores and writes `fingerprints.npy` and `index.json`.

| | |
|---|---|
| Source | [OpenScore String Quartets](https://github.com/OpenScore/StringQuartets): every score in the repository, 196 MusicXML files |
| Commit | [`2614b0b2958dd1ce3f52e172a43e5f4d0e29570f`](https://github.com/OpenScore/StringQuartets/tree/2614b0b2958dd1ce3f52e172a43e5f4d0e29570f), the one the model's survey and the ten real pages I photographed take their scores at ([the test set](https://huggingface.co/datasets/gregology/sheet-music-scanner-eval)) |
| Licence | CC0 1.0 Universal. Nothing of the scores is kept here but the fingerprints: 1,356,643 numbers, from which no music can be read back |
| Made | 2026-11-15: 787 lines of 196 scores |

Nothing here is drawn, printed or trained on. A corpus builder reads `fingerprints.npy` and leaves out every score that shares a stretch of music with it. `index.json` records the commit, the scores, the settings of the fingerprints and a hash of the file, and a corpus built against it records that hash.
