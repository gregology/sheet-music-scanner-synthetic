import csv
import random
import zipfile

import numpy as np
import pytest

from sheet_synth import pdmx, tested
from sheet_synth.example import Music, engraved_page
from sheet_synth.scores import Corpus, read_score
from sheet_synth.scores import write_corpus as write_lieder

from test_scores import QUARTERS, Q, note, score

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def tune(seed, count=48, low=60):
    """A melody nobody else has: semitones above C0, moving by a different step each time."""
    rng = random.Random(seed)
    pitches = [low]
    while len(pitches) < count:
        pitches.append(min(84, max(55, pitches[-1] + rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7]))))
    return pitches


def name(pitch):
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def measures(pitches, kind="quarter", length=Q, to_a_measure=4):
    return ["".join(note(name(pitch), kind, length) for pitch in pitches[at:at + to_a_measure]) for at in range(0, len(pitches), to_a_measure)]


def line_of(folder, bars, stem="song", **how):
    path = folder / f"{stem}.musicxml"
    path.write_text(score(*bars, **how))
    return read_score(path)[0]


def prints_of(*lines):
    return np.unique(np.concatenate([tested.fingerprints(tested.melody(line)) for line in lines]))


# The test music is in a corpus under other titles, typed by other people, moved to other keys for other instruments
# and written in longer notes. A check by title or by the notes as written lets those copies through, and a reader
# is then tested on music it was trained on.
def test_a_copy_of_test_music_in_another_key_and_in_longer_notes_is_caught(tmp_path):
    quartet = tune(1)
    known = prints_of(line_of(tmp_path, measures(quartet), "quartet"))
    copy = line_of(tmp_path, measures([pitch + 2 for pitch in quartet], "half", 2 * Q, 2), "copy")
    assert tested.is_tested(copy, known)
    assert tested.shared(copy, known) >= 30


# A copy that starts with something of its own, or breaks off, is still the test music where it runs along with it.
def test_a_stretch_of_test_music_inside_another_piece_is_caught(tmp_path):
    quartet = tune(1)
    known = prints_of(line_of(tmp_path, measures(quartet), "quartet"))
    medley = line_of(tmp_path, measures(tune(2, 32) + quartet[8:40] + tune(3, 32)), "medley")
    assert tested.is_tested(medley, known)


# Music that only happens to be by the same hand must stay in the corpus: a check that turned away every score with
# a few steps in common would empty it.
def test_other_music_is_not_a_copy(tmp_path):
    known = prints_of(line_of(tmp_path, measures(tune(1)), "quartet"))
    for seed in range(2, 12):
        assert tested.shared(line_of(tmp_path, measures(tune(seed)), "other"), known) == 0


# An accompaniment figure is in a quartet and in a thousand other pieces. Counting every time it comes round made
# 45 of 2,000 scores into copies, a piano sonata and a pop song among them, where two were.
def test_a_figure_played_over_and_over_is_not_a_copy(tmp_path):
    figure = [60, 64, 67, 72, 76, 72, 67, 64]
    known = prints_of(line_of(tmp_path, measures(tune(1) + figure * 6), "quartet"))
    accompaniment = line_of(tmp_path, measures(figure * 12), "song")
    assert 0 < tested.shared(accompaniment, known) < tested.SHARED
    assert not tested.is_tested(accompaniment, known)


# One person writes a long note, another two notes tied, and one leaves a rest where the other holds on: read as
# written, the same music would not look the same.
def test_tied_notes_are_one_note_and_rests_are_left_out(tmp_path):
    whole = line_of(tmp_path, [note("C4", "whole", 4 * Q), note("E4", "half", 2 * Q) + note("G4", "half", 2 * Q)], "whole")
    tied = line_of(tmp_path, [note("C4", "half", 2 * Q, ties=("start",)) + note("C4", "half", 2 * Q, ties=("stop",)), note("E4", "half", 2 * Q) + note("G4", "half", 2 * Q)], "tied")
    rested = line_of(tmp_path, [note("C4", "whole", 4 * Q), note("E4", "half", 2 * Q) + note("G4", "quarter") + note(None, "quarter")], "rested")
    assert tested.melody(whole) == tested.melody(tied)
    assert [pitch for pitch, _ in tested.melody(rested)] == [pitch for pitch, _ in tested.melody(whole)]


# A scale or a note said twelve times is in all music. With fingerprints for those, every line with a scale in it
# shared a stretch with the test music.
def test_a_plain_run_has_no_fingerprint(tmp_path):
    scale = line_of(tmp_path, measures(list(range(60, 84)) + list(range(84, 60, -1))), "scale")
    assert len(tested.fingerprints(tested.melody(scale))) == 0


def table(folder, rows):
    names = ["path", "metadata", "mxl", "license", "license_conflict", "song_name", "composer_name", "subset:deduplicated", "subset:all_valid"]
    with open(folder / "PDMX.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, names)
        writer.writeheader()
        for number, (stem, conflict, unique) in enumerate(rows):
            writer.writerow({"path": f"./data/1/2/{stem}.json", "metadata": f"./metadata/5/{5000 + number}.json", "mxl": f"./mxl/1/2/{stem}.mxl", "license": "publicdomain", "license_conflict": str(conflict),
                             "song_name": f"Song {stem}", "composer_name": "NA", "subset:deduplicated": str(unique), "subset:all_valid": "True"})
    return folder / "PDMX.csv"


# 12% of PDMX's scores carry one licence on the website and another in the file. Read by default, they would be in
# every dataset without anyone having chosen that; and a second copy of a song would count its music twice.
def test_licence_conflicts_and_second_copies_are_left_out_unless_asked_for(tmp_path):
    path = table(tmp_path, [("a", False, True), ("b", True, True), ("c", False, False)])
    assert [score["mxl"] for score in pdmx.chosen(path, conflicts=False)] == ["mxl/1/2/a.mxl"]
    with_conflicts = pdmx.chosen(path, conflicts=True)
    assert [score["mxl"] for score in with_conflicts] == ["mxl/1/2/a.mxl", "mxl/1/2/b.mxl"]
    assert with_conflicts[1]["licence_conflict"] and with_conflicts[0]["musescore"] == 5000 and with_conflicts[0]["composer"] is None


def packed(folder, stem, bars):
    """A score as PDMX keeps it: MusicXML in a zip that names it."""
    path = folder / "mxl" / "1" / "2" / f"{stem}.mxl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as file:
        file.writestr("META-INF/container.xml", '<?xml version="1.0"?><container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>')
        file.writestr("score.xml", score(*bars))
    return path


@pytest.fixture(scope="module")
def pdmx_folder(tmp_path_factory):
    """A corpus of two PDMX scores, one of them a copy of test music in another key, and the fingerprints it was checked against."""
    folder = tmp_path_factory.mktemp("pdmx")
    quartet = tune(1)
    known = prints_of(line_of(folder, measures(quartet), "quartet"))
    jobs = [(packed(folder, "QmOwn", measures(tune(7))), {"musescore": 11, "licence": "cc-zero", "licence_conflict": False, "title": "Own", "composer": "Somebody"}),
            (packed(folder, "QmCopy", measures([pitch - 3 for pitch in quartet])), {"musescore": 12, "licence": "publicdomain", "licence_conflict": False, "title": "Quartet for flutes", "composer": None})]
    index = pdmx.write_corpus(folder / "corpus", jobs, known, {"name": "pdmx", "source": "a test", "commit": "1", "licence": "a test"})
    return folder / "corpus", index


# The whole score goes, not only the line that matched: its other lines are the same piece's other parts.
def test_a_score_that_copies_test_music_is_left_out_whole_and_said_so(pdmx_folder):
    folder, index = pdmx_folder
    assert [line["score"] for line in index["lines"]] == ["QmOwn"]
    assert [(left["score"], left["title"]) for left in index["tested_music_left_out"]] == [("QmCopy", "Quartet for flutes")]
    assert not list(folder.rglob("QmCopy*"))


# A dataset under CC BY must be able to say whose score each page is from and under which licence its uploader put
# it. A page that recorded only a file name could not.
def test_a_page_from_pdmx_says_whose_score_it_is(pdmx_folder):
    folder, _ = pdmx_folder
    corpus = Corpus(folder)
    assert corpus.name.startswith("pdmx@1/")
    piece, source = corpus.piece(random.Random(3))
    assert source["about"] == {"musescore": 11, "licence": "cc-zero", "licence_conflict": False}
    assert source["score"] == "QmOwn" and source["line"] == "1/2/QmOwn-1.json.gz"
    assert piece.title == "A Test Song"


# With no MuseScore file to vouch for the clef, a change of clef the MusicXML lost shows only in where the notes
# sit: nine ledger lines off the staff. Such a measure made a page no edition prints.
def test_notes_far_off_the_staff_are_turned_away_where_nothing_vouches_for_the_clef(tmp_path):
    low = "".join(note(pitch) for pitch in ("C2", "D2", "E2", "F2"))   # bass notes under a treble clef
    line = line_of(tmp_path, [QUARTERS, low, QUARTERS])
    assert line["bad"] == [None, "notes far off the staff for the clef", None]


@pytest.fixture(scope="module")
def lieder_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lieder")
    path = folder / "song.musicxml"
    path.write_text(score(*[QUARTERS] * 12))
    write_lieder(folder / "corpus", [path], "a test")
    return folder / "corpus"


# A dataset from two corpora must draw from both, and a dataset from one must stay what it was: a draw between
# corpora that was made even for one would have changed every real page of a dataset already recorded.
def test_several_corpora_are_drawn_from_by_their_size_and_one_corpus_is_as_before(pdmx_folder, lieder_folder):
    both = Music(1.0, (str(lieder_folder), str(pdmx_folder[0])))
    corpora = [engraved_page(9, index, both)[2]["corpus"].split("@")[0] for index in range(16)]
    assert set(corpora) == {"lieder", "pdmx"}
    assert corpora == [engraved_page(9, index, both)[2]["corpus"].split("@")[0] for index in range(16)]
    one, as_tuple = Music(1.0, str(lieder_folder)), Music(1.0, (str(lieder_folder),))
    for index in range(4):
        assert engraved_page(9, index, one)[0].svg == engraved_page(9, index, as_tuple)[0].svg
