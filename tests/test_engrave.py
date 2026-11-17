import random
from fractions import Fraction

import pytest

from sheet_synth.compose import Composer, compose, key_alter, to_mei
from sheet_synth.engrave import Style, engrave

SEEDS = range(60)


def page_of(seed):
    composer = Composer(seed)
    piece = composer.piece()
    return piece, engrave(piece, Style.draw(random.Random(f"style{seed}"), composer.level))


@pytest.fixture(scope="module")
def pages():
    return [page_of(seed) for seed in SEEDS]


# A note the engraver drew but the truth left out, or one the truth lists twice, would teach a reader wrong labels.
def test_every_note_on_the_page_is_in_the_truth_once(pages):
    for piece, page in pages:
        listed = [symbol["id"] for system in page.truth["systems"] for symbol in system["symbols"] if symbol["type"] in ("note", "grace", "rest")]
        assert len(listed) == len(set(listed))
        drawn = {note.id for measure in piece.measures for event in measure.events for note in [event] + event.under if f'id="{note.id}"' in page.svg}
        assert set(listed) == drawn


# A measure whose lengths do not add up to its time signature is not music an engraver would print.
def test_each_measure_fills_its_time_signature(pages):
    for piece, _ in pages:
        time = piece.time
        for measure in piece.measures:
            time = measure.time or time
            if measure.pickup or not measure.cells:
                continue
            total = Fraction(0)
            for cell in measure.cells:
                for event in cell.events:
                    if not event.grace:
                        total += Fraction(4, event.dur) * (2 - Fraction(1, 2 ** event.dots)) * (Fraction(cell.tuplet[1], cell.tuplet[0]) if cell.tuplet else 1)
            assert total == Fraction(4 * time[0], time[1]), (measure.id, time)


# Symbols out of reading order would scramble a line's token sequence.
def test_a_lines_symbols_run_left_to_right(pages):
    for _, page in pages:
        for system in page.truth["systems"]:
            centres = [(symbol["box"][0] + symbol["box"][2]) / 2 for symbol in system["symbols"] if not symbol.get("under")]   # a chord's lower notes stand under its top note
            assert centres == sorted(centres)
            assert system["tokens"][0].startswith("clef-")


# A box outside the page, or a staff without five lines, means the drawing was read in the wrong coordinates.
def test_boxes_and_staves_lie_on_the_page(pages):
    for _, page in pages:
        assert page.truth["systems"]
        for system in page.truth["systems"]:
            assert len(system["staff"]["lines"]) == 5
            for item in system["symbols"] + system["marks"]:
                x0, y0, x1, y1 = item["box"]
                assert -0.01 <= x0 <= x1 <= 1.01 and -0.01 <= y0 <= y1 <= 1.01, item


# An accidental the key signature already gives, drawn again, or one it does not give, left off, changes the pitch read.
def test_an_accidental_is_shown_only_where_the_pitch_differs_from_what_the_measure_gives():
    for seed in SEEDS:
        piece = compose(seed)
        key = piece.key
        for measure in piece.measures:
            key = measure.key if measure.key is not None else key
            state = {}
            for event in measure.events:
                if event.rest:
                    continue
                for note in [event] + event.under:
                    expected = state.get(note.di, key_alter(key, note.step))
                    if note.tie not in ("t", "m"):
                        assert (note.shown is not None) == (note.alter != expected), (seed, note.id)
                    state[note.di] = note.alter


# The same seed giving a different page would make the dataset impossible to regenerate or to resume.
def test_the_same_seed_gives_the_same_page():
    first, second = page_of(7), page_of(7)
    assert to_mei(first[0]) == to_mei(second[0])
    assert first[1].truth == second[1].truth


def test_key_signatures_alter_the_right_letters():
    assert [key_alter(2, step) for step in "fcgd"] == [1, 1, 0, 0]
    assert [key_alter(-3, step) for step in "beadg"] == [-1, -1, -1, 0, 0]


# A chord read bottom first, or split from its top note by a neighbour, would give a reader the wrong order to learn.
def test_a_chords_notes_follow_its_top_note_from_the_top_down(pages):
    chords = 0
    for _, page in pages:
        for system in page.truth["systems"]:
            symbols = system["symbols"]
            for place, symbol in enumerate(symbols):
                if symbol.get("under"):
                    above = symbols[place - 1]
                    assert above["id"] == symbol["under"] or above.get("under") == symbol["under"]
                    assert (above["octave"], "CDEFGAB".index(above["step"])) > (symbol["octave"], "CDEFGAB".index(symbol["step"]))
                    assert system["tokens"].count("chord") == sum(1 for other in symbols if other.get("under"))
                    chords += 1
    assert chords > 20


# The first readers met triplets without brackets, grace notes in groups and rests of many measures only on real
# pages, and misread them. Pages that stop drawing them would train the next reader the same way.
def test_the_pages_hold_what_real_parts_hold(pages):
    tokens = [token for _, page in pages for system in page.truth["systems"] for token in system["tokens"]]
    shown = {symbol.get("tuplet_shown") for _, page in pages for system in page.truth["systems"] for symbol in system["symbols"]}
    assert {"number", "bracket", "none"} <= shown
    assert any(token.endswith("_4:3") for token in tokens) and any(token.endswith("_8:3") for token in tokens) and any(token.endswith("_16:6") for token in tokens)
    assert sum(token.startswith("grace-") for token in tokens) > 20
    counts = {int(token.split("-")[1]) for token in tokens if token.startswith("multirest-")}
    assert len(counts) > 6 and min(counts) <= 4 and max(counts) > 16
