import random

import pytest

from sheet_synth.compose import Cell, Composer, Event, Measure
from sheet_synth.engrave import Style, engrave, tuplet_token
from sheet_synth.example import Music, engraved_page, stage
from sheet_synth.scores import MIN_RUN, Corpus, join_rests, mend_ties, read_score, runs_of, source_clefs, window, write_corpus
from sheet_synth.things import bottom_line, is_name, page_things, symbol_things

Q = 48   # the test scores' divisions of a quarter note


def note(pitch, kind="quarter", length=Q, *, chord=False, voice=1, staff=1, dots=0, ties=(), beam=None, tuplet=None, of=None, tremolo=None, grace=False, hidden=False, cue=False, courtesy=None):
    """One MusicXML note, or a rest when `pitch` is None, written as MuseScore writes it."""
    parts = []
    if grace:
        parts.append('<grace slash="yes"/>')
    if chord:
        parts.append("<chord/>")
    if pitch is None:
        parts.append('<rest measure="yes"/>' if kind is None else "<rest/>")
    else:
        step, sign, octave = pitch[0], pitch[1:-1], pitch[-1]
        alter = sign.count("#") - sign.count("b")
        parts.append(f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") + f"<octave>{octave}</octave></pitch>")
    if not grace:
        parts.append(f"<duration>{length}</duration>")
    parts += [f'<tie type="{end}"/>' for end in ties]
    parts.append(f"<voice>{voice}</voice>")
    if kind is not None:
        parts.append(f'<type size="cue">{kind}</type>' if cue else f"<type>{kind}</type>")
    parts += ["<dot/>"] * dots
    if courtesy:
        parts.append(f"<accidental>{courtesy}</accidental>")
    if of:
        parts.append(f"<time-modification><actual-notes>{of[0]}</actual-notes><normal-notes>{of[1]}</normal-notes></time-modification>")
    parts.append(f"<staff>{staff}</staff>")
    if beam:
        parts.append(f'<beam number="1">{beam}</beam>')
    signs = [f'<tied type="{end}"/>' for end in ties]
    if tuplet:
        signs.append(f"<tuplet {tuplet}/>")
    if tremolo:
        signs.append(f'<ornaments><tremolo type="{tremolo}">3</tremolo></ornaments>')
    if signs:
        parts.append("<notations>" + "".join(signs) + "</notations>")
    return ('<note print-object="no">' if hidden else "<note>") + "".join(parts) + "</note>"


def triplet(*pitches, mark=""):
    """Three eighths in the time of two, under one beam, the tuplet's sign written with `mark`."""
    first, middle, last = pitches
    return (note(first, "eighth", Q // 3, of=(3, 2), beam="begin", tuplet=f'type="start" {mark}') + note(middle, "eighth", Q // 3, of=(3, 2), beam="continue")
            + note(last, "eighth", Q // 3, of=(3, 2), beam="end", tuplet='type="stop"'))


def score(*measures, staves=1, key=0, time=(4, 4)):
    clefs = '<clef number="1"><sign>G</sign><line>2</line></clef>' + ('<clef number="2"><sign>F</sign><line>4</line></clef>' if staves > 1 else "")
    head = (f"<attributes><divisions>{Q}</divisions><key><fifths>{key}</fifths></key><time><beats>{time[0]}</beats><beat-type>{time[1]}</beat-type></time>"
            + (f"<staves>{staves}</staves>" if staves > 1 else "") + clefs + "</attributes>")
    body = "".join(f'<measure number="{number + 1}">{head if number == 0 else ""}{measure}</measure>' for number, measure in enumerate(measures))
    return ('<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0"><work><work-title>A Test Song</work-title></work>'
            '<identification><creator type="composer">A. Composer</creator></identification>'
            f'<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1">{body}</part></score-partwise>')


def lines_of(folder, *measures, source=None, **how):
    path = folder / "song.musicxml"
    path.write_text(score(*measures, **how))
    return read_score(path, source)


def museScore(folder, *measures, old=False):
    """A MuseScore file of one staff. Each measure is its signs in order; MuseScore 2 (`old`) kept them in the measure itself."""
    body = "".join(f"<Measure>{signs}</Measure>" if old else f"<Measure><voice>{signs}</voice></Measure>" for signs in measures)
    path = folder / "song.mscx"
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?><museScore version="3.02"><Score><Staff id="1">{body}</Staff></Score></museScore>')
    return path


def page_of(line, count=None):
    piece = window(line, 0, count or len(line["measures"]), random.Random(1))
    return piece, engrave(piece, Style.draw(random.Random(1), 2))


def tokens_of(page, signs=False):
    """The page's tokens, without the clefs, key and time signatures unless `signs`."""
    return [token for system in page.truth["systems"] for token in system["tokens"] if signs or not token.startswith(("clef-", "key-", "time-"))]


QUARTERS = "".join(note(pitch) for pitch in ("C4", "D4", "E4", "F4"))
TREMOLO = note("C4", "half", Q, of=(2, 1), tremolo="start") + note("E4", "half", Q, of=(2, 1), tremolo="stop") + note("G4", "half", 2 * Q)


@pytest.fixture(scope="module")
def corpus_folder(tmp_path_factory):
    """A corpus of one song: ten measures, one that cannot be drawn, nine more with a tremolo between two notes in the last but one."""
    folder = tmp_path_factory.mktemp("scores")
    path = folder / "song.musicxml"
    path.write_text(score(*[QUARTERS] * 10, note("C4", hidden=True) + "".join(note(pitch) for pitch in ("D4", "E4", "F4")), *[QUARTERS] * 7, TREMOLO, QUARTERS))
    write_corpus(folder / "corpus", [path], "a test")
    return folder / "corpus"


# A chord read bottom-up, or a tie given to the whole chord when one of its notes holds, writes tokens for notes
# that are not tied and misses the one that is.
def test_a_chord_is_read_from_the_top_down_with_each_notes_own_tie(tmp_path):
    first = note("C4", "half", 2 * Q) + note("E4", "half", 2 * Q, chord=True, ties=["start"]) + note("G4", "half", 2 * Q, chord=True)
    second = note("C4", "half", 2 * Q) + note("E4", "half", 2 * Q, chord=True, ties=["stop"])
    [line] = lines_of(tmp_path, first + second)
    piece, page = page_of(line)
    top = piece.measures[0].events[0]
    assert [(top.step, top.octave)] + [(under.step, under.octave) for under in top.under] == [("g", 4), ("e", 4), ("c", 4)]
    assert [top.tie] + [under.tie for under in top.under] == [None, "i", None]
    assert tokens_of(page) == ["note-G4_2", "chord", "note-E4_2", "tie", "chord", "note-C4_2", "note-E4_2", "chord", "note-C4_2", "barline"]
    assert {"chord of 3", "chord of 2", "tie on a chord"} <= page_things(page.truth)


# The real pages' commonest mistake was a triplet with no mark. A reader that drew every tuplet with a bracket
# would never show one, however the score prints it.
def test_a_tuplets_mark_is_the_scores_own(tmp_path):
    measure = triplet("C5", "D5", "E5", mark='show-number="none"') + triplet("C5", "D5", "E5", mark='bracket="yes"') + triplet("C5", "D5", "E5") + note("C5")
    [line] = lines_of(tmp_path, measure)
    piece, page = page_of(line)
    assert [(cell.tuplet, cell.show) for cell in piece.measures[0].cells[:3]] == [((3, 2), "none"), ((3, 2), "bracket"), ((3, 2), "number")]
    assert {"triplet, no mark", "triplet, bracket and number", "triplet, number only"} <= page_things(page.truth)
    assert tokens_of(page)[:3] == ["note-C5_8:3", "note-D5_8:3", "note-E5_8:3"]


# A duplet written `:2` would be read as two in the time of one, half its length, and its measure would not add up.
def test_a_tuplet_that_is_not_in_the_time_of_the_power_of_two_below_names_both_numbers(tmp_path):
    assert [tuplet_token(tuplet) for tuplet in [(3, 2), (6, 4), (5, 4), (7, 4), (2, 3), (4, 3), (9, 8)]] == [":3", ":6", ":5", ":7", ":2/3", ":4/3", ":9"]
    duplet = note("C5", "eighth", 3 * Q // 4, of=(2, 3), beam="begin", tuplet='type="start" bracket="yes"') + note("D5", "eighth", 3 * Q // 4, of=(2, 3), beam="end", tuplet='type="stop"')
    [line] = lines_of(tmp_path, duplet + note("E5", "quarter", 3 * Q // 2, dots=1), time=(6, 8))
    assert line["bad"] == [None]
    assert tokens_of(page_of(line)[1])[:3] == ["note-C5_8:2/3", "note-D5_8:2/3", "note-E5_4."]


# A tremolo between two notes prints each at the length of both together. Read as the file's sounding lengths, the
# truth would name quarter notes where half notes are drawn; read without the sign, it would not say why two half
# notes and a third fill a measure of four quarters.
def test_a_tremolo_between_two_notes_keeps_the_printed_lengths(tmp_path):
    [line] = lines_of(tmp_path, TREMOLO)
    piece, page = page_of(line)
    assert line["bad"] == [None]
    assert piece.measures[0].cells[0].rock == 3
    assert tokens_of(page) == ["note-C4_2", "tremolo", "note-E4_2", "note-G4_2", "barline"]
    assert "tremolo between two notes" in page_things(page.truth)


# A measure whose notation the piece's data cannot hold, if drawn anyway, is a picture with wrong truth: a hidden
# rest leaves a hole, a small note is drawn full size, a clef inside the measure moves every note after it.
@pytest.mark.parametrize("measure, staves, reason", [
    (note("C4", hidden=True) + QUARTERS[len(note("C4")):], 1, "a hidden note or rest"),
    (note("C4", cue=True) + QUARTERS[len(note("C4")):], 1, "small notes"),
    (note("C4") + note("C3", staff=2) + note("E4") + note("F4"), 2, "a note on the other staff"),
    (note("C4") + '<attributes><clef number="1"><sign>F</sign><line>4</line></clef></attributes>' + note("D4") + note("E4") + note("F4"), 1, "a clef inside the measure"),
    ('<direction><direction-type><octave-shift type="down" size="8"/></direction-type><staff>1</staff></direction>' + QUARTERS
     + '<direction><direction-type><octave-shift type="stop" size="8"/></direction-type><staff>1</staff></direction>', 1, "an 8va or 8vb line"),
    (note("C4", "breve", 8 * Q), 1, "a note or rest of a length that is not drawn: breve"),
    (note("D4", "eighth", grace=True) + note("F4", "eighth", grace=True, chord=True) + QUARTERS, 1, "a grace note that is a chord"),
    (note("C4") + f"<forward><duration>{Q}</duration><voice>1</voice><staff>1</staff></forward>" + note("E4") + note("F4"), 1, "a gap in the voice"),
    (QUARTERS + '<barline location="right"><bar-style>dashed</bar-style></barline>', 1, "a barline that is not drawn"),
])
def test_a_measure_that_cannot_be_written_is_turned_away(tmp_path, measure, staves, reason):
    lines = lines_of(tmp_path, QUARTERS, measure, QUARTERS, staves=staves)
    assert lines[0]["bad"] == [None, reason, None]


# MuseScore 4.7.6 leaves a change of clef written as the first thing in a measure out of its MusicXML: one in fifty of
# the corpus's. The notes after it kept their pitches and were drawn under the old clef, a left hand in the treble
# clef nine ledger lines below the staff. The score's own file says which clef each measure starts with.
@pytest.mark.parametrize("old", [False, True])
def test_a_clef_the_conversion_left_out_is_taken_from_the_scores_own_file(tmp_path, old):
    bass = "<Clef><concertClefType>F</concertClefType></Clef>"
    source = museScore(tmp_path, "<Chord/>", bass + "<Chord/>", "<Chord/><Clef><concertClefType>C1</concertClefType></Clef>", "<Chord/>", old=old)
    assert source_clefs(source) == [(4, {1: "bass", 3: None})]
    low = "".join(note(pitch) for pitch in ("C3", "D3", "E3", "F3"))
    [line] = lines_of(tmp_path, QUARTERS, low, low, low, source=source)
    assert [state[0] for state in line["states"]] == ["treble", "bass", "bass", None]
    assert line["bad"] == [None, None, None, "a clef that is not drawn"]
    assert "clef-F4" in tokens_of(page_of(line, 3)[1], signs=True) and not any(name.startswith("ledger lines below") for name in line["things"][1])
    [line] = lines_of(tmp_path, QUARTERS, low, low, source=source)   # a file whose measures do not line up cannot say which clef is where
    assert set(line["bad"]) == {"measures that do not line up with the score's own file"}


# A hidden time signature drawn as a printed one would put a sign on the page that the score does not print, before
# a measure the score prints as a free one.
def test_a_hidden_time_signature_is_not_drawn(tmp_path):
    hidden = '<attributes><time print-object="no"><beats>5</beats><beat-type>4</beat-type></time></attributes>' + QUARTERS + note("G4")
    [line] = lines_of(tmp_path, QUARTERS, hidden, QUARTERS)
    piece, page = page_of(line)
    assert line["bad"] == [None, None, None] and [measure.pickup for measure in piece.measures] == [False, True, False]
    assert [token for token in tokens_of(page, signs=True) if token.startswith("time-")] == ["time-4/4"]


# Two voices on a staff drawn as one would stack their notes into chords that are not in the score. The first voice
# is drawn alone, and the measures where another was left out are recorded for whoever asks how real the page is.
def test_only_a_staffs_first_voice_is_read_and_the_measure_is_recorded(tmp_path):
    two = QUARTERS + f"<backup><duration>{4 * Q}</duration></backup>" + note("C3", "whole", 4 * Q, voice=2)
    [line] = lines_of(tmp_path, QUARTERS, two)
    assert line["cut"] == [1] and line["bad"] == [None, None]
    assert tokens_of(page_of(line)[1]).count("chord") == 0


# The model writes a pitch as a place on the staff and a drawn accidental by one rule. A courtesy accidental the file
# asks for but the rule does not would be a sign on the page that no pitch sequence explains.
def test_accidentals_are_drawn_by_the_generators_rule_not_the_files(tmp_path):
    first = note("F#4") + note("F#4") + note("G4", "half", 2 * Q)
    second = note("F4", courtesy="natural") + note("G4") + note("A4", "half", 2 * Q)
    [line] = lines_of(tmp_path, first, second)
    piece, _ = page_of(line)
    assert [event.shown for measure in piece.measures for event in measure.events] == ["s", None, None, None, None, None]


# A page cut from the middle of a score would start with the end of a tie from the measure before, or end with the
# start of one: a `tie` token with no second note, which the model's pitch rule reads as an accidental carried on.
def test_a_tie_cut_by_the_edge_of_the_page_is_taken_away():
    first = Event("a", di=30, tie="t", under=[Event("b", di=28, tie="m")])
    middle = Event("c", di=28, tie="t")
    last = Event("d", di=31, tie="i")
    mend_ties([Measure("m1", [Cell([first, middle])]), Measure("m2", [Cell([last])])])
    assert (first.tie, first.under[0].tie, middle.tie, last.tie) == (None, "i", "t", None)


# Empty measures printed one by one where a part prints a single rest with a count: the rest the first readers
# misread most would be missing from every real page.
def test_empty_measures_in_a_row_become_one_rest():
    measures = [Measure(f"m{number}", whole_rest=True) for number in range(5)]
    measures[3].key = 2
    joined = join_rests(measures + [Measure("m5", [Cell([Event("n")])])])
    assert [(measure.multirest, measure.whole_rest, measure.key) for measure in joined[:2]] == [(3, False, None), (2, False, 2)]
    assert len(joined) == 3


# A page that runs across a measure that could not be read would draw nothing, or the wrong thing, in its place.
def test_a_page_is_cut_only_from_measures_in_a_row_that_can_be_drawn(corpus_folder):
    assert runs_of([None] * (MIN_RUN - 1) + ["a hidden note or rest"] + [None] * MIN_RUN) == [[MIN_RUN, 2 * MIN_RUN]]
    corpus = Corpus(corpus_folder)
    assert corpus.index["turned_away"] == {"a hidden note or rest": 1} and corpus.index["usable"] == 19
    for seed in range(40):
        _, source = corpus.piece(random.Random(seed))
        assert not source["from_measure"] <= 10 < source["from_measure"] + source["measures"]


# The same seed giving another page of the score would make a dataset impossible to regenerate or to resume.
def test_the_same_seed_cuts_the_same_page(corpus_folder):
    corpus = Corpus(corpus_folder)
    assert corpus.piece(random.Random("7/3/music")) == corpus.piece(random.Random("7/3/music"))
    music = Music(1.0, str(corpus_folder))
    assert [tokens_of(engraved_page(7, index, music)[0]) for index in range(3)] == [tokens_of(engraved_page(7, index, music)[0]) for index in range(3)]


# Asked for pages with a rare thing, a generator that cut pages anywhere would return pages without it: the point of
# asking is that every page holds the thing a reader gets wrong.
def test_a_real_page_asked_to_hold_a_thing_holds_it(corpus_folder):
    music = Music(1.0, str(corpus_folder), ("tremolo between two notes",))
    for index in range(6):
        page, level, source = engraved_page(9, index, music)
        assert "tremolo between two notes" in page_things(page.truth)
        assert source["kind"] == "score" and level is None
    with pytest.raises(LookupError):
        Corpus(corpus_folder).piece(random.Random(1), ["64th note"])


# The same for the composer, which is asked again until its page holds the thing; and the page must still be a
# function of the seed.
def test_a_composed_page_asked_to_hold_a_thing_holds_it():
    music = Music(holds=("triplet, number only",))
    for index in range(3):
        page, _, source = engraved_page(4, index, music)
        assert "triplet, number only" in page_things(page.truth)
        assert source == {"kind": "composer"}
        assert tokens_of(page) == tokens_of(engraved_page(4, index, music)[0])


# The first datasets were made by the composer alone. If the new options changed the music of an example made
# without them, a run of v0 or v1 resumed today would mix two kinds of page under one seed.
def test_without_the_new_options_an_example_is_the_one_it_was():
    for seed, index in [(1, 0), (2, 41), (3, 988)]:
        composer = Composer(f"{seed}/{index}/music")
        before = engrave(composer.piece(), Style.draw(stage(seed, index, "style"), composer.level))
        page, level, source = engraved_page(seed, index)
        assert page.svg == before.svg and level == composer.level and source == {"kind": "composer"}


# A share of real pages that came out all real or all composed, or that changed with the example's other choices,
# would make a mixed dataset something else than what was asked for.
def test_the_share_of_real_pages_is_the_one_asked_for(corpus_folder):
    kinds = [engraved_page(5, index, Music(0.5, str(corpus_folder)))[2]["kind"] for index in range(24)]
    assert 6 <= kinds.count("score") <= 18
    assert kinds[:8] == [engraved_page(5, index, Music(0.5, str(corpus_folder)))[2]["kind"] for index in range(8)]


# A page of real music draws things no composed page has: tremolos between two notes, ties inside chords, double
# sharps, 64th notes. A note of theirs drawn but missing from the truth, or listed and not drawn, is a wrong label.
def test_every_note_of_a_real_page_is_in_the_truth_once(tmp_path):
    sixty_fourths = "".join(note(pitch, "64th", Q // 16, beam="begin" if place == 0 else "end" if place == 7 else "continue") for place, pitch in enumerate(["C6", "D6"] * 4))
    odd = note("F##4", "quarter", 7 * Q // 4, dots=2) + note("Bbb4", "16th", Q // 4) + sixty_fourths + note(None, "eighth", Q // 2) + note("A5", "quarter")
    chords = note("C4", "half", 2 * Q) + note("E4", "half", 2 * Q, chord=True, ties=["start"]) + note("C4", "half", 2 * Q) + note("E4", "half", 2 * Q, chord=True, ties=["stop"])
    [line] = lines_of(tmp_path, TREMOLO, odd, chords)
    assert line["bad"] == [None, None, None]
    piece, page = page_of(line)
    listed = [symbol["id"] for system in page.truth["systems"] for symbol in system["symbols"] if symbol["type"] in ("note", "grace", "rest")]
    drawn = {each.id for measure in piece.measures for event in measure.events for each in [event] + event.under if f'id="{each.id}"' in page.svg}
    assert len(listed) == len(set(listed)) == 19 and set(listed) == drawn
    assert {"note-F##4_4..", "note-Bbb4_16", "note-C6_64", "rest-8"} <= set(tokens_of(page))
    assert {"double sharp or flat", "double dot", "64th note", "ledger lines above: 2", "ledger lines below: 1"} <= page_things(page.truth)


# Ledger lines counted from the wrong line of the staff, or without the clef, would send pages of ordinary notes when
# notes high above the staff were asked for.
def test_ledger_lines_are_counted_from_the_clefs_staff():
    def held(step, octave, clef):
        return symbol_things({"type": "note", "step": step, "octave": octave}, bottom_line(clef))

    assert held("c", 4, "clef-G2") == ["ledger lines below: 1"] and held("c", 4, "clef-F4") == ["ledger lines above: 1"]
    assert held("c", 6, "clef-G2") == ["ledger lines above: 2"] and held("g", 5, "clef-G2") == [] and held("d", 4, "clef-G2") == []
    assert held("c", 4, "clef-C3") == [] and held("c", 4, "clef-G2_8vb") == [] and held("a", 4, "clef-G2_8vb") == ["ledger lines above: 1"]


# A name misspelt on the command line would be looked for on every page and never found: a run of thousands of
# examples that all fail. And a thing a page reports under a name the command refuses could never be asked for.
def test_a_page_can_be_asked_for_by_every_name_it_reports_and_no_other(corpus_folder):
    pages = [engraved_page(6, index)[0] for index in range(12)] + [engraved_page(6, index, Music(1.0, str(corpus_folder)))[0] for index in range(4)]
    reported = set().union(*(page_things(page.truth) for page in pages))
    assert len(reported) > 30 and [name for name in reported if not is_name(name)] == []
    assert not any(is_name(name) for name in ["triplet no mark", "32nd", "ledger lines above", "note-C4_4", "rest-4", "clef: treble"])
