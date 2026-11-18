# Sheet Music Scanner: synthetic training data

This makes fake phone photos of sheet music, each with its exact score, to train a model that reads a photo of a page into a score on an iPhone. A page of music, made up here or cut from a real score, is engraved, marked in pencil, printed on paper, bent, laid on a surface in a Blender scene, lit, and photographed by a simulated phone camera. The music is the generator's own data before it's drawn, so the answer for every photo is known exactly. No label comes from a person or from another recogniser.

The model trained on it has never seen a real photo, and reads my 81 iPhone photos of real string-quartet parts with 1.10% of notes wrong, 0.21% after a check that each measure adds up ([the model](https://huggingface.co/gregology/sheet-music-scanner), [its code](https://github.com/gregology/sheet-music-scanner-model), [the test set](https://huggingface.co/datasets/gregology/sheet-music-scanner-eval)).

| | |
|---|---|
| Made so far | 55,995 training photos in five datasets, plus a 2,000-photo test set ([the datasets](#the-datasets)) |
| Photo | 3024 x 4032 JPEG, an iPhone's 12 megapixels, about 1.6 MB |
| Speed | 4 to 5 seconds a photo with four workers on one RTX 4090 |
| Truth | Every line's tokens, every sign's box on the page and in the photo, each staff line's path, and every choice the generator made |
| Licence | Code MIT. Datasets I publish CC BY 4.0 ([licence](#licence)) |

## An example

One example is a photo and its truth: `00000042.jpg` and `00000042.json`. Both are a function of the dataset's seed and the example's number, so any example can be made again.

1. **Compose** a piece ([the music](#the-music)), or **cut** a page from a real score ([real scores](#real-scores)).
2. **Engrave** it with Verovio ([the engraving](#the-engraving)).
3. **Print** it on paper ([paper and print](#paper-and-print)).
4. **Mark** it by hand ([handwriting](#handwriting)).
5. **Bend** the sheet, lay it on a surface, light it and **render** it ([the scene](#the-scene)).
6. **Develop** the render the way a phone would ([the camera](#the-camera)).
7. **Check** the render shows the page its truth describes ([checks](#every-render-is-checked-against-its-truth)).

The truth records what's on the page and where, twice: on the flat page and in the photo. It also records the paper, the sheet's shape, the room, the shot and the camera settings, so a model can learn how hard a photo is and when to ask for another.

## The music

[`compose.py`](src/sheet_synth/compose.py) writes each piece from a seed.

- **One staff, one voice,** for single-line instruments. A note can have one to three more on its stem, as a string player's double stops do.
- **Four levels,** from a beginner's quarter and half notes to dense sixteenths, triplets, ties and ledger lines.
- **Clefs:** treble, bass, alto, tenor, and treble an octave down. **Keys** up to six sharps or flats. **Time signatures:** 2/4, 3/4, 4/4, common, cut, 3/8, 6/8, 9/8, 12/8, 5/4, 7/8, 3/2, 6/4.
- **Triplets** of eighths, quarters and sixteenths, marked with a bracket and number, the number alone, or nothing once the piece has shown a few. Unmarked, only the beaming and the measure's length tell them from plain notes.
- **Grace notes** alone or in beamed groups of up to four, with accidentals.
- **Parts.** Half the pieces above the first level look like an ensemble player's part: rests of 2 to 99 measures with the count above, slurs and dots throughout, bowings, harmonics, and a page head with the instrument at the top left.
- **Changes mid-piece** of key, time and clef, with the courtesy sign at the end of the line before.
- **Everything around the notes:** accidentals, dots, ties, slurs, articulations, fermatas, trills, dynamics, hairpins, tempo words, rehearsal letters, repeats with endings, pickups, lyrics and chord names.

The tunes are random walks with scales, repeated notes and broken chords mixed in. They have the notation of real music and few of its habits, which is why real scores came next.

### What real pages changed

The first readers, trained only on composed pages (dataset v0), got 1.3% of notes wrong on photos of my made-up test sheets and 11.9% on photos of real quartet parts. Clean renders of the real parts scored about the same as photos of them, so the gap was in the music, not the photography. I sorted the mistakes by what was printed where they fell, and changed the composer for dataset v1:

| On the real pages | In v0 | From v1 |
|---|---|---|
| Triplets with a number and no bracket, quarter-note triplets, unmarked triplets: 87 of the best reader's 184 mistakes | Eighth-note triplets, always bracketed | Three kinds of triplet, marked three ways |
| Multi-measure rests drawn as a thick bar with the count above; a printed 2 read as 5, 8 and 12 | Counts up to 4, drawn as whole and double whole rests | The thick bar on three pages in four, any count from 2 to 99 |
| Double and triple stops: 22 mistakes | None | Chords |
| Grace notes in beamed groups, with accidentals | One slashed grace note | Groups of one to four |
| A floor of boards, a black music stand, a stack of sheets with edges showing | Noise patterns, a sheet alone | Boards, a stand, a stack |
| Sheets bowed 40 to 55 degrees edge to edge | One sheet in sixteen over 22 degrees | One in six |

That took real music from 11.9% to 2.6% of notes wrong. Most of what was left needed music the composer doesn't write.

## Real scores

[`scores.py`](src/sheet_synth/scores.py) reads a real score into the same data the composer writes, one line at a time, and cuts a page of measures from it. Everything after that is the same: engraving, handwriting, paper, scene, camera. The truth is still the generator's own data, the notes from the score's file placed by the engraver's drawing. Nothing is read from a picture.

**The Lieder corpus.** The [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder), CC0: 1,352 songs for voice and piano, mostly 19th century. A line is the singer's part, or one hand of the piano. 4,044 lines, 208,427 drawable measures. MuseScore 4 converts each score to MusicXML once, so it has to be installed for the build:

```sh
uv run python -m sheet_synth.scores        # writes scores/lieder, 21 MB
```

**A page** is 48 to 96 measures in a row from one line. Three pages in ten start where a song starts, five anywhere, two near the end.

**What a line keeps:** pitches, lengths, chords, ties, tuplets and how each is marked, beams, grace notes, tremolos, clefs, keys, time signatures, repeats, slurs, dynamics, articulations, printed fingering and the first verse of words.

**What it drops:** other voices on the same staff (15.6% of drawable measures, mostly piano; each example records how many), courtesy accidentals the edition prints and the notation rule doesn't need, the edition's line breaks, and hidden signs. A measure the data can't hold, such as a note on the other piano staff, an 8va line or a clef mid-measure, is turned away, and no page runs across one. That's 12.8% of the Lieder corpus's measures.

**Two checks on the reading.** [`checks/read_scores_with_music21.py`](checks/read_scores_with_music21.py) reads the same measures with music21, which shares no code with mine, and compares every note's pitch, length and chord position. On 150 Lieder pages, 147 agreed and music21 couldn't open three. On 150 PDMX pages, 143 agreed; five of the seven that didn't were the check's own fault (chord names counted as notes), and two are measure rests whose file disagrees with its own time signature. Separately, MuseScore drops some clef changes from the MusicXML it writes (87 at the start of a measure, maybe 30 more), so each measure's opening clef is taken from the MuseScore file itself.

A page from a real score says where it's from:

```json
"source": {"kind": "score", "corpus": "lieder@38c5db510224/e2f2c3c7b6ba", "score": "lc6583477", "line": "lc6583477-2.json.gz",
           "label": "Piano", "part": 1, "staff": 1, "voice": "1", "from_measure": 12, "measures": 48,
           "measures_with_another_voice_left_out": 21}
```

### PDMX

[`pdmx.py`](src/sheet_synth/pdmx.py) builds a second, much larger corpus from [PDMX](https://zenodo.org/records/15571083) (Long, Novack, Berg-Kirkpatrick and McAuley, ICASSP 2025): scores people typed into musescore.com and marked public domain or CC0, published together under CC BY 4.0. It has parts for every kind of instrument, and pop, film music and hymns beside the classics.

```sh
uv run python -m sheet_synth.tested     # first: fingerprints of the test music, a few minutes
uv run python -m sheet_synth.pdmx       # downloads 2.1 GB, writes scores/pdmx, 1.1 GB; under an hour with 8 workers
```

- **Which scores.** PDMX's `deduplicated` subset with all-valid files: 102,596 scores. 25,275 of those have one licence on musescore.com and another inside the file, which PDMX flags as `license_conflict`. They're left out by default, as PDMX's makers recommend, and `--with-licence-conflicts` reads them too. Without them it reads 77,321 scores.
- **What it holds:** 218,570 lines and 14.2 million drawable measures, about 68 times the Lieder corpus. Against Lieder it has 4,404 measures with a tremolo between two notes (958), 30,620 with an unmarked triplet (3,291), 20,212 with a double sharp or flat (1,301) and 157 time signatures (25).
- **What's turned away:** 15% of measures. Mostly percussion and tablature clefs, drum notes with no pitch, and hidden notes. PDMX has no MuseScore files, so a clef change the MusicXML lost can't be put back. Instead a measure whose notes sit on average past the third ledger line is turned away. On the Lieder corpus, where the true clefs are known, that catches the worst 42% of measures under a lost clef and 0.7% of good ones.

Each page's truth names its score and licence, so it can be credited or taken out later:

```json
"source": {"kind": "score", "corpus": "pdmx@15571083/…", "score": "QmZ…", "line": "17/22/QmZ…-1.json.gz",
           "about": {"musescore": 5437997, "licence": "publicdomain", "licence_conflict": false}, …}
```

`generate --scores scores/lieder scores/pdmx` draws from both, every measure as likely as any other.

### Music a reader is tested on

The readers are tested on OpenScore's string quartets: the survey of 434 real pages and the ten real pages I photographed are parts from them. PDMX holds other people's copies and arrangements of the same quartets, under other titles, and a page drawn from one would train a reader on its own test. So scores are matched by their notes, not their titles.

[`tested.py`](src/sheet_synth/tested.py) turns every line of all 196 quartets into fingerprints, and `pdmx.py` leaves out any score with a line that shares a long stretch with them ([record](scores/tested/SOURCE.md)).

- **A fingerprint** is thirteen notes in a row, as the twelve steps between them: how far the pitch moves and how the length changes. It's the same in any key and at any tempo. Rests and grace notes are skipped and tied notes are one note, because people type those differently.
- **Too plain to count:** a run with fewer than 5 different steps, like a scale or a repeated note, gets no fingerprint.
- **A copy** is a line with 12 different test fingerprints in a row, about 24 notes that move and last alike. A figure repeated over and over counts once, so a common accompaniment pattern doesn't make a thousand pieces copies.

It leaves out 302 of 77,319 scores (0.4%). Eleven of the twelve PDMX scores whose titles name a test quartet by opus number are caught, among them Mendelssohn's Op. 44 No. 1 with 1,017 matching fingerprints in a row and Beethoven's Op. 18 No. 1 arranged for four marimbas. The twelfth is a different piece with a similar title. Others are caught with no title match at all. The cost is false positives: 223 of the 302 share only 12 to 20 fingerprints, often pop songs or Bach running alongside a quartet for a while.

It can't catch a copy with the rhythm changed throughout, a part that only accompanies, or a test part rewritten as chords whose top note is another voice.

## Pages that hold a named thing

When a reader turns out to misread something, the generator can be asked for pages that hold it:

```sh
uv run python -m sheet_synth.generate --out out/tremolo --seed 8 --count 500 --real 1 --holds "tremolo between two notes" --device OPTIX
```

Every page then holds that thing. `--holds` can be given more than once. The names are the ones the model's `tools/sort_mistakes.py` sorts mistakes by, so a row of its table pastes straight into the command.

| Name | Is |
|---|---|
| `32nd note`, `64th note`, `double dot` | A note or rest of that length |
| `triplet, no mark`, `triplet, number only`, `triplet, bracket and number` | Three in the time of two, by how it's marked |
| `sextuplet, ...`, `tuplet 2 in 3, ...` | Six in four, and any other tuplet, marked the same three ways |
| `tied`, `tied over a barline`, `tie on a chord` | |
| `chord`, `chord of 3`, `accidental drawn`, `double sharp or flat`, `grace note`, `rest of whole measures` | |
| `ledger lines above: 4`, `ledger lines below: 2` | A note that far off the staff |
| `tremolo strokes`, `tremolo between two notes` | Strokes through one stem, or between two notes |
| `mark: stacc`, `fermata`, `trill` | Marks on a note |
| `clef-C3`, `time-6/8`, `key-5b`, `repeat-start` | Any token that isn't a note or a timed rest |

[`things.py`](src/sheet_synth/things.py) does the lookup. A real score's index says what each measure holds, and the page is cut to start up to twelve measures before one that holds it. The composer is asked again with the next seed, up to 200 times, until its page holds it. A thing the composer never writes, such as a tremolo between two notes, needs `--real 1`. Either way the example is still a function of its seed.

`uv run python -m sheet_synth.things scores/lieder --composed 200` lists every name with how often each source holds it.

## The engraving

[`engrave.py`](src/sheet_synth/engrave.py) draws each piece with [Verovio](https://www.verovio.org) 6.4.1 in a random style: one of five music fonts (Leipzig, Bravura, Gootville, Leland, Petaluma), US Letter or A4, margins of 10 to 24 mm, staff heights of 5.5 to 10.2 mm (mostly near 7, as published music is), and random note spacing and line weights. Only the first page is kept, and the truth holds only what was drawn on it.

## The truth

The truth is what's drawn on the page and where. Every note goes into the MEI with an id of ours, and Verovio's SVG says which ids landed on the page, on which line, in which box. Nothing is read back from pixels. Page positions are fractions from the top left corner, `[left, top, right, bottom]`.

```json
{
  "format": 3, "title": "Etude in A", "label": "Cello",
  "page": {"size": "letter", "width_mm": 215.9, "height_mm": 279.4},
  "systems": [
    {
      "staff": {"lines": [0.114, 0.120, 0.126, 0.131, 0.137], "left": 0.044, "right": 0.950, "lines_px": [[[412.0, 655.3], …], …]},
      "symbols": [{"type": "clef", "clef": "bass", "token": "clef-F4", "box": [0.05, 0.11, 0.07, 0.14], "quad": [[…], …]}],
      "marks": [{"type": "fingering", "of": "n12", "text": "3", "box": [0.21, 0.10, 0.22, 0.11]}],
      "tokens": ["clef-F4", "key-3#", "time-9/8", "note-B2_8.", "barline"]
    }
  ]
}
```

- **`symbols`** are what a reader must get right, in reading order: clefs, key and time signatures, notes, rests, barlines. A note carries its pitch, length, accidental, tie and anything printed on it. A note in a triplet has `tuplet_shown`: `bracket`, `number` or `none`.
- **`marks`** are the other signs with their boxes: accidentals, dots, beams, slurs, ties, hairpins, words, fingering, lyrics. Stems and flags are left out.
- **In the photo:** `page.outline_px` is the page's edge, `page.grid_px` a 33 by 25 lattice that places any page point in the photo (enough to flatten it), `staff.lines_px` each staff line as 17 points, `quad` every box's four corners, and `cut_off` anything partly out of frame. They come from the same arithmetic that shapes the sheet's mesh and places the camera ([`geometry.py`](src/sheet_synth/geometry.py)); a test renders dots on a bent sheet and finds each within two pixels.

### Tokens

`tokens` is each line as a flat sequence, for training a reader directly.

| Token | Meaning |
|---|---|
| `clef-G2`, `clef-F4`, `clef-C3`, `clef-C4`, `clef-G2_8vb` | A clef: its shape and the line it sits on |
| `key-3#`, `key-2b`, `key-0` | A key signature, up to seven |
| `time-3/4`, `time-C`, `time-C\|` | A time signature |
| `note-Bb4_8.` | Sounding pitch, then length. `4..` is double-dotted, `64` a 64th |
| `note-F##4_4`, `note-Bbb3_4` | Double sharp, double flat |
| `note-E5_8:3`, `note-E5_16:6` | A tuplet note: three in the time of two, six in four |
| `note-E5_8:2/3` | Any other tuplet names both numbers: two in the time of three |
| `chord` | The notes either side sound together, top note first |
| `grace-D5_16` | A grace note at its drawn length |
| `rest-4`, `rest-measure`, `multirest-8` | Rests |
| `tie` | Follows each tied note, so one note of a chord can hold while the others move |
| `tremolo` | Between two notes or chords played in turn. Each is written at its printed length, the length of both: `note-C4_2 tremolo note-E4_2` fills a half note, not a whole |
| `barline`, `barline-double`, `barline-end`, `repeat-start`, `repeat-end` | Barlines |

The truth's `format` is 3, which added `source` and what real scores bring: ties on chord notes, tremolos, double sharps and flats, any tuplet, 64ths, double dots, seven-sharp keys and any time signature. Composed pages have the same tokens in formats 2 and 3. Format 1 (dataset v0) has no chords.

## Handwriting

[`handwriting.py`](src/sheet_synth/handwriting.py) marks 55% of pages the way a player does: finger numbers over or under notes, bowings, accidentals written in, arrows, breath marks, ticks, circles, and words like "slow". Mostly pencil, sometimes black, blue or red pen, and the odd highlighter. Every mark is drawn as pen strokes through jittered control points, so no two are alike and no handwriting font or sample is used. Marks are listed under `handwriting` with the note each belongs to, and are not in `tokens`.

## Paper and print

[`paper.py`](src/sheet_synth/paper.py) turns the black-on-white page into a printed sheet at 400 dpi: white, cream, aged or grey paper with grain, browned edges, a stain, the back or the sheet below showing through, a fold line; printed by laser, inkjet (soft and banded), offset, photocopier (hard edges, specks) or faded. Nothing moves, so the truth stays right.

## The scene

[`scene.py`](src/sheet_synth/scene.py) builds a room in [Blender](https://www.blender.org) 5.0 and renders it with Cycles. Everything is made in code; no texture, model or lighting file is loaded.

- **The sheet:** flat, bowed, a book page falling into the spine, wavy, a lifted corner, folds. Paper bends in one direction without stretching. Nearly half the bows turn 22 to 60 degrees edge to edge, like a stack that's been carried.
- **The surface:** floorboards with seams, a music stand's dark desk with the room behind, wood, cloth, stone or more paper. Six sheets in ten sit on a stack of one to nine others with their edges showing. I added boards and the stand after the first line finder traced staff lines in floorboard seams.
- **The light:** sun through a window with its edge across the page, a ceiling light, a low lamp, soft even light, or daylight and a lamp together. In three shots of four the phone, hands and photographer cast shadows. A coated page shines back at a lamp in two of five.
- **The shot:** hand-held, mostly near straight above, the page filling 80% to 100% of the frame. One in eight is steep, up to 48 degrees; one in sixteen cuts the page off. The lens is the iPhone 16's main camera, now and then its 13 mm or 48 mm.

## The camera

[`camera.py`](src/sheet_synth/camera.py) turns the clean render into what a phone would save. Noise goes in where a sensor adds it, on the light before any processing, so dim photos are noisier:

| Light | Electrons a white page fills a pixel with | Share |
|---|---|---|
| `bright` | 12,000 to 40,000 | 40% |
| `indoor` | 1,500 to 9,000 | 35% |
| `dim` | 300 to 1,500 | 17% |
| `dark` | 50 to 300 | 8% |

Then shake and missed focus, the phone's processing (exposure and white balance slightly off, noise reduction that eats fine detail, lifted shadows, sharpening), and JPEG at quality 60 to 95. The darkest photos are meant to be unreadable, for teaching a model to ask for another.

## Generate a dataset

```sh
uv sync
uv run python -m sheet_synth.generate --out out/v1 --seed 1 --count 1000 --device OPTIX --workers 4
```

- Examples go to `<out>/<shard>/<number>.jpg` and `.json`, a thousand to a shard. A stopped run carries on with the same arguments.
- `--real 0.75` makes three in four examples from real scores; `--scores` names the corpora. `--holds "<name>"` makes only [pages that hold a named thing](#pages-that-hold-a-named-thing). Without either, an example is exactly what it was before they existed.
- `dataset.json` records the seed, the photo size, each corpus by its commit and a hash of its lines, and the generator's commit from `SHEET_SYNTH_COMMIT`.
- **It shares the GPU by waiting.** My RTX 4090 also serves a language model. When a render runs out of memory or comes back black, the worker stops and retries the same number a minute later, and a render won't start until 6 GB is free. No example is skipped for lack of memory.
- The GPU has to do the denoising: on the CPU, denoising took 5 of a small render's 6.4 seconds.

### Every render is checked against its truth

An example that fails a check isn't saved; `<number>.failed` records why. Before writing, [`shows_its_page`](src/sheet_synth/example.py) looks for ink in the clean render where the truth says the notes are, and drops a render with ink under fewer than half.

This exists because of a bug I found early on. With Blender told to keep the renderer loaded between examples, the GPU drew the first example's page in every later example of that worker. The photos looked fine and the labels were wrong: 22 of 36 examples I checked showed the wrong page. The setting is off, and the check scored those wrong pages at 0.34 or under and right ones at 1.0. It also turns away about one good example in 10,000: hollow notes printed faint and out of focus.

### Checked against a reader of real photos

[`checks/read_with_homr.py`](checks/read_with_homr.py) runs [homr](https://github.com/liebharc/homr), a reader trained on real photos, on examples and scores it against the truth. It's a check that the pictures look like photos, and it makes no labels; homr is AGPL, and none of its code or weights is in the generator or the model. On 34 well-lit examples, homr read 33 of the clean pages with 5% of notes wrong or fewer, and 24 of the photos with 10% or fewer. It found three faults, all fixed: photos too small to read the staff lines, a photocopy effect that erased them, and the wrong-page bug above.

### Publish a dataset

```sh
uv run python -m sheet_synth.pack out/v2 out/v2-packed
```

[`pack.py`](src/sheet_synth/pack.py) writes each shard as a WebDataset tar, photo and truth side by side, which Hugging Face reads without a loading script, and writes the card with the share of real music and the credit each corpus needs. Nothing is uploaded.

## The datasets

All generated on one RTX 4090 with four workers. "Turned away" is examples the checks refused. The generator commits are from the private repo this one was started from, so they won't resolve here; each dataset's `dataset.json` records its own.

| Dataset | Seed | Generator | Examples | Size | Music | Made for |
|---|---|---|---|---|---|---|
| v0 | 1 | `b2328ed` | 29,997 (3 turned away) | 31 GB | Composed, format 1 | The first readers |
| v1 | 2 | `3beb748` | 10,000 | 12 GB | Composed | [What real pages changed](#what-real-pages-changed) |
| t1 | 3 | `3beb748` | 2,000 | 2.6 GB | Composed, as v1 | Testing only. Never trained on |
| v2 | 4 | `3e39fcc` | 11,998 (2 turned away) | 14 GB | Three in four real (`--real 0.75`), Lieder and PDMX with its licence conflicts | Real notation |
| v3-tremolo | 5 | `f422f0f` | 2,000 | 2.4 GB | All real, each page holds a tremolo between two notes | The reader's worst sign on real pages |
| v3-triplet | 6 | `f422f0f` | 2,000 | 2.5 GB | Three in four real, each page holds an unmarked triplet | Not used ([below](#a-dataset-that-made-things-worse)) |

How many training pages came from PDMX scores with a licence conflict:

| Dataset | Composed | Lieder | PDMX | PDMX with a licence conflict |
|---|---|---|---|---|
| v0, v1, t1 | all | | | none |
| v2 | 3,017 | 108 | 6,651 | 2,222 (18.5%) |
| v3-tremolo | | 251 | 1,174 | 575 (28.8%) |

I built v2 and v3 with the conflicts in, trusting each uploader's public-domain marking. The published model, version 1.0, trains without those pages, and a reader trained with them scored the same within the noise between runs. I'll leave them out of any dataset I publish too.

What each dataset changed, for a reader through the same line finder, as the share of notes wrong ([the model's notes](https://github.com/gregology/sheet-music-scanner-model) have every run):

| Trained on | 434 clean renders of real quartet parts | My 81 photos of real music |
|---|---|---|
| v0 and v1 | 6.16% | 2.51% |
| v0 to v2 | 3.79% | 1.59% |
| v0 to v2 and v3-tremolo | 3.26% | 0.94% |
| The same, without the licence-conflict pages (version 1.0) | 3.32% | 1.13% |

On the renders, tremolos between two notes went from 764 of 922 wrong to 498 with the v3-tremolo pages.

### A dataset that made things worse

v3-triplet asked for 2,000 pages that each hold a triplet with no mark, the reader's other big miss on real pages. Trained with it, the reader got tremolos right but started reading plain eighths as triplets: 342 times on the renders where it had done so 96 times, and my real-music photos went from 1.59% wrong to 2.19%. An unmarked triplet looks exactly like three plain eighths on its own line; only the measure's length tells them apart. So 2,000 pages full of them taught the reader to guess. I dropped v3-triplet, and the app leaves those triplets to a check that adds up each measure and mends the ones that don't fit.

### How large

The first dataset was 30,000 examples, about 210,000 lines, which is roughly what comparable readers trained on: Camera-PrIMuS 87,678 lines, homr about 190,000. Rather than guess at more, I trained on 10,000, 20,000 and 30,000 and compared on real photos. Through the same line finder the three read my photos of made-up sheets at 1.3%, 1.2% and 1.3% of notes wrong, so more of the same wasn't helping. The later datasets change what's on the page instead.

## Run the tests

```sh
uv sync
uv run pytest
```

Python 3.11, because Blender's Python module needs it. The tests render three small pictures, about half a minute. To see a page with its truth drawn over it:

```python
import random
from sheet_synth.compose import Composer
from sheet_synth.engrave import Style, engrave
from sheet_synth.preview import page_overlay

composer = Composer(5)
page = engrave(composer.piece(), Style.draw(random.Random(5), composer.level))
page_overlay(page.raster(150), page.truth).save("out/page.png")
```

## What it leaves out

- **Other engravers.** Every layout is Verovio's; the fonts change the shapes, not the spacing. Real scores still come out looking like Verovio, not like the edition.
- **More than one staff or voice.** No piano music as a pianist reads it, and a real score's second voice is dropped.
- **Octave lines, measure repeats, cue notes, nested tuplets, clefs mid-measure.** Real measures with them are turned away.
- **Real textures.** Surfaces are procedural, not photographs of wood or cloth. No lens distortion.
- **Volta brackets and titles** are drawn but not in the truth.

## Licence

- **The code** is [MIT](LICENSE).
- **Datasets I publish** are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): use them for anything, model training and commercial use included, and credit Greg Clarke.
- **The music** is made up here, or from the [OpenScore Lieder Corpus](scores/lieder/SOURCE.md) (CC0), or from [PDMX](scores/pdmx/SOURCE.md) (CC BY 4.0), whose credit `pack.py` puts on every card that uses it. Each PDMX score is marked public domain or CC0 by its uploader; that's the uploader's claim, and PDMX's makers found it in conflict for 12% of scores.
- **The tools:** Verovio (LGPL-3.0) and its fonts (SIL Open Font License) draw the pages and Blender (GPL) renders them. None of those licences reaches the pictures they make.
