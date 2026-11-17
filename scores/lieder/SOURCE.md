# scores/lieder

Lines of real music for the generator's second source ([README](../../README.md#real-scores)). The files here are made, not committed: `uv run python -m sheet_synth.scores` downloads the scores, converts them and writes one `.json.gz` for each line and `index.json`.

| | |
|---|---|
| Source | [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder): 1,352 songs for voice and piano, most from the 19th century, typed in by volunteers from public-domain editions |
| Commit | [`38c5db510224d9facdc4b08d741fc788cfb58ea8`](https://github.com/OpenScore/Lieder/tree/38c5db510224d9facdc4b08d741fc788cfb58ea8), 2026-09-09 |
| Licence | [CC0 1.0 Universal](https://github.com/OpenScore/Lieder/blob/38c5db510224d9facdc4b08d741fc788cfb58ea8/LICENSE.txt): no rights reserved. Each score says so itself: the MusicXML MuseScore writes from it carries `<rights>OpenScore (CC0)</rights>` |
| Checked | 2026-11-15, from the repository's `LICENSE.txt` and GitHub's licence field (`cc0-1.0`) |
| Converted with | MuseScore 4.7.6, `.mscx` to MusicXML |

It holds none of the music a reader is tested on: the survey and my real test photos are string quartets, from OpenScore's other corpus.

`index.json` records the commit, the converter's version, a hash of every line, how many measures can be drawn, and why the rest were turned away. A dataset made from the corpus records the corpus by that commit and hash.
