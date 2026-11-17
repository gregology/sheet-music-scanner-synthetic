"""Writes a made-up piece for a single-line instrument, and its MEI.

The piece is plain data (`Piece`, `Measure`, `Cell`, `Event`), so its true contents are known without reading the
MEI back. Every element that is drawn carries an id of ours, and the engraver's drawing is matched to the piece by
those ids. What the piece may contain, and how often: README.md, "The music".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from xml.sax.saxutils import escape

STEPS = "cdefgab"
SHARP_ORDER = "fcgdaeb"
ACCIDENTAL = {1: "s", -1: "f", 0: "n", 2: "x", -2: "ff"}     # the sign drawn
SOUNDING = {1: "s", -1: "f", 2: "ss", -2: "ff"}             # what a note with no sign of its own sounds

# shape, line, octave displacement, and the lowest and highest diatonic index a tune stays within at the easiest level
# (diatonic index = octave * 7 + step, with C = 0; middle C is 28).
CLEFS = {
    "treble": ("G", 2, 0, 28, 40),
    "bass": ("F", 4, 0, 16, 28),
    "alto": ("C", 3, 0, 22, 34),
    "tenor": ("C", 4, 0, 20, 32),
    "treble8": ("G", 2, -8, 21, 33),
}


@dataclass
class Event:
    """One note or rest."""

    id: str
    rest: bool = False
    di: int = 28                      # diatonic index of a note
    alter: int = 0                    # how many semitones the note sounds above its letter
    dur: int = 4                      # 1 whole, 2 half, 4 quarter ... 64
    dots: int = 0
    grace: bool = False
    slash: bool = False               # a grace note with a stroke through its stem
    under: list["Event"] = field(default_factory=list)   # the other notes of a chord: played with this one, drawn below it, highest first
    tremolo: int = 0                  # strokes through the stem: the note is repeated fast for its length
    shown: str | None = None          # the accidental drawn before the note: "s", "f", "n", "x" (double sharp) or "ff"
    tie: str | None = None            # "i" starts a tie, "t" ends one, "m" does both
    artics: list[str] = field(default_factory=list)
    fermata: bool = False
    trill: bool = False
    fingering: str | None = None      # a printed finger number
    lyric: tuple[str, str] | None = None   # (syllable, its place in the word: "s" alone, "i", "m", "t")
    dynamic: str | None = None
    chord: str | None = None          # a chord name above the note
    beam: int | None = None           # in a score read from a file: which of its measure's beams the note is under

    @property
    def step(self) -> str:
        return STEPS[self.di % 7]

    @property
    def octave(self) -> int:
        return self.di // 7

    @property
    def pitch(self) -> str:
        """`B-4`, `F#5`, `C4`: letter, alteration, octave."""
        return f"{self.step.upper()}{'#' * self.alter if self.alter > 0 else '-' * -self.alter}{self.octave}"


@dataclass
class Cell:
    """A beat's worth of events, beamed together. `tuplet` is (notes played, in the time of)."""

    events: list[Event]
    tuplet: tuple[int, int] | None = None
    show: str = "bracket"             # how a tuplet is marked: "bracket" (with its number), "number" alone, or "none"
    above: bool | None = None         # the side its number is drawn on; None leaves it to the engraver
    given: bool = False               # the beams are a real score's own, each event's `beam`, not made by the run rule
    rock: int = 0                     # a tremolo between the cell's two notes, with this many strokes: they are played in turn, fast


@dataclass
class Measure:
    id: str
    cells: list[Cell] = field(default_factory=list)
    key: int | None = None            # a key signature that starts here: sharps, or flats if negative
    time: tuple[int, int, str | None] | None = None   # a time signature that starts here: count, unit, symbol
    clef: str | None = None           # a clef that starts here, drawn small at the end of the measure before
    left: str | None = None           # "rptstart"
    right: str | None = None          # "rptend", "dbl", "end"
    ending: int | None = None         # which numbered ending the measure is under
    multirest: int = 0                # the measure stands for this many empty measures
    whole_rest: bool = False
    pickup: bool = False              # shorter or longer than its time signature says, as an upbeat is
    tempo: str | None = None
    direction: str | None = None      # words under the first note: "rit.", "dolce"
    rehearsal: str | None = None

    @property
    def events(self) -> list[Event]:
        return [event for cell in self.cells for event in cell.events]


@dataclass
class Piece:
    title: str
    composer: str | None
    label: str | None                 # the instrument's name before the first staff
    clef: str
    key: int
    time: tuple[int, int, str | None]
    measures: list[Measure]
    subtitle: str | None = None       # under the title: "Op. 51 No. 2"
    movement: str | None = None       # over the first line: "I."
    part: bool = False                # laid out as an ensemble player's part: the instrument's name at the top left of the page
    airy: bool = False                # the part's head is set with a blank line between its lines
    slurs: list[tuple[str, str]] = field(default_factory=list)          # (first note's id, last note's id)
    hairpins: list[tuple[str, str, str]] = field(default_factory=list)  # ("cres" or "dim", first id, last id)


def key_alter(sharps: int, step: str) -> int:
    """What the key signature does to a letter: +1, -1 or 0."""
    if sharps > 0:
        return 1 if step in SHARP_ORDER[:sharps] else 0
    return -1 if step in SHARP_ORDER[::-1][:-sharps] else 0


# Rhythm cells: each a list of lengths. "4" is a quarter, "8." a dotted eighth, "r4" a quarter rest. Sorted by the
# level at which they first appear. SIMPLE cells are counted in quarter beats, COMPOUND ones fill a dotted quarter.
SIMPLE = {
    1: [(0, ["4"]), (0, ["4"]), (0, ["r4"]), (1, ["8", "8"]), (1, ["8", "8"]), (2, ["8.", "16"]), (2, ["16", "16", "16", "16"]),
        (2, ["8", "16", "16"]), (2, ["16", "16", "8"]), (2, ["r8", "8"]), (2, ["8", "r8"]), (3, ["16", "8", "16"]),
        (3, ["r16", "16", "16", "16"]), (3, ["16", "16", "r8"]), (3, ["32", "32", "32", "32", "8"]), (3, ["16.", "32", "8"])],
    2: [(0, ["2"]), (0, ["2"]), (0, ["r2"]), (1, ["4.", "8"]), (2, ["8", "4", "8"]), (2, ["4", "8.", "16"]), (3, ["8", "4.", ]), (3, ["r8", "4", "8"])],
    3: [(0, ["2."]), (2, ["4.", "8", "4"]), (3, ["4", "r4", "4"])],
    4: [(0, ["1"]), (1, ["2.", "4"]), (2, ["4.", "8", "2"])],
    6: [(0, ["1."])],
}
TRIPLETS = [["8", "8", "8"], ["8", "8", "8"], ["8", "8", "8"], ["4", "8"], ["8", "r8", "8"], ["r8", "8", "8"], ["8", "4"]]   # three eighths in the time of two
WIDE_TRIPLETS = [["4", "4", "4"], ["4", "4", "4"], ["4", "4", "4"], ["2", "4"], ["4", "2"], ["4", "r4", "4"], ["r4", "4", "4"]]   # three quarters in the time of two
SIXES = [["16"] * 6, ["16"] * 6, ["8", "16", "16", "16", "16"], ["16", "16", "16", "8."]]   # six sixteenths in the time of four
COMPOUND = {
    1: [(0, ["4."]), (0, ["4."]), (0, ["8", "8", "8"]), (0, ["8", "8", "8"]), (0, ["4", "8"]), (1, ["8", "4"]), (1, ["r4."]), (2, ["8", "r8", "8"]),
        (2, ["16", "16", "16", "16", "16", "16"]), (2, ["8.", "16", "8"]), (2, ["8", "16", "16", "8"]), (3, ["4", "16", "16"]), (3, ["r8", "8", "8"]),
        (3, ["16", "16", "8", "8"])],
    2: [(0, ["2."]), (1, ["2", "4"]), (2, ["4.", "4", "8"])],
}
# count, unit, symbol, and the beats a measure divides into: (beats, "s" for quarter beats or "c" for dotted-quarter).
TIMES = [
    ((4, 4, None), [(4, "s")], 30), ((4, 4, "common"), [(4, "s")], 10), ((3, 4, None), [(3, "s")], 20), ((2, 4, None), [(2, "s")], 12),
    ((2, 2, "cut"), [(4, "s")], 5), ((6, 8, None), [(2, "c")], 12), ((3, 8, None), [(1, "c")], 3), ((9, 8, None), [(3, "c")], 3),
    ((12, 8, None), [(4, "c")], 2), ((5, 4, None), [(3, "s"), (2, "s")], 2), ((7, 8, None), [(2, "s"), (1, "c")], 1),
    ((3, 2, None), [(6, "s")], 3), ((6, 4, None), [(3, "s"), (3, "s")], 2),
]

TITLE_A = ["Etude", "Study", "Exercise", "Air", "Minuet", "March", "Waltz", "Song", "Dance", "Prelude", "Melody", "Lullaby", "Gavotte", "Reel", "Jig", "Romance", "Scherzo", "Andante", "Allegro", "Largo"]
TITLE_B = ["in the Morning", "of the River", "for Two Hands", "at Dusk", "on a Ground", "of the Old Mill", "in Spring", "without Words", "for a Rainy Day", "of the North"]
KEY_NAMES = ["C", "G", "D", "A", "E", "B", "F sharp", "C sharp", "F", "B flat", "E flat", "A flat", "D flat", "G flat"]
COMPOSERS = ["Trad.", "Anon.", "arr. J. Smith", "M. Lindqvist", "A. Ferrand", "arr. K. Oduya", "H. Becker", "T. Marlow", "L. Okabe", "R. Castellan", "after an old tune", "C. Whitlow, Op. 12"]
LABELS = ["Violin", "Flute", "Oboe", "Clarinet in B flat", "Trumpet", "Cello", "Viola", "Voice", "Alto Sax", "Trombone", "Bassoon", "Horn in F", "Recorder", "Violin I", "Violin II"]
TEMPI = ["Allegro", "Andante", "Moderato", "Allegretto", "Adagio", "Largo", "Presto", "Vivace", "Lento", "Andantino", "Con moto", "Tempo di valse", "Slowly", "Brightly", "With energy", "Allegro ma non troppo"]
DIRECTIONS = ["rit.", "a tempo", "dolce", "cresc.", "dim.", "poco rit.", "espress.", "leggiero", "cantabile", "accel.", "simile", "marcato"]
DYNAMICS = ["p", "mp", "mf", "f", "pp", "ff", "fp", "sfz"]
PART_DYNAMICS = DYNAMICS + ["p", "f", "sf", "sf", "sfp", "fz", "ppp", "fff"]
ARTICULATIONS = ["stacc", "stacc", "acc", "ten", "marc"]
PART_ARTICULATIONS = ["stacc", "stacc", "stacc", "stacc", "acc", "ten", "marc", "upbow", "dnbow", "harm"]
PART_DIRECTIONS = DIRECTIONS + ["cresc.", "dim.", "dolce", "espress.", "pizz.", "arco", "sul G", "div.", "unis.", "con sord.", "sempre", "molto cresc.", "poco a poco"]
SUBTITLES = ["Op. {a}", "Op. {a} No. {b}", "No. {b}", "K. {c}", "Op. {a}, No. {b}", "BWV {c}", "Hob. III:{a}"]
MOVEMENTS = ["I.", "II.", "III.", "IV.", "I", "II", "1.", "2.", "Finale", "Menuetto", "Scherzo", "Trio"]
PART_TITLES = ["String Quartet No. {b}", "Quartet in {key}", "Symphony No. {b}", "Trio No. {b}", "Serenade", "Divertimento in {key}", "Quintet in {key}", "Sonata No. {b}", "Concerto in {key}", "Suite No. {b}", "Overture"]
PART_COMPOSERS = ["A. Ferrand", "M. Lindqvist", "H. Becker", "T. Marlow", "L. Okabe", "R. Castellan", "C. Whitlow", "J. Halvorsen", "E. Brandt", "P. Novak", "G. Morel", "F. Lehner"]
# What a second note is stacked under the first, in staff steps below it, and how often: thirds and sixths mostly.
STOPS = [(2, 30), (5, 26), (7, 14), (4, 14), (3, 10), (1, 3), (6, 3)]
SYLLABLES = [["la"], ["the"], ["and"], ["sing"], ["day"], ["light"], ["we"], ["morn", "ing"], ["mu", "sic"], ["riv", "er"], ["o", "ver"], ["sil", "ver"], ["to", "geth", "er"],
             ["mel", "o", "dy"], ["a", "way"], ["home"], ["sweet"], ["gar", "den"], ["win", "ter"], ["sum", "mer"], ["sleep"], ["rose"], ["wan", "der", "ing"], ["heart"], ["my"], ["love"]]
CHORD_KINDS = ["", "", "", "m", "m", "7", "m7", "maj7", "sus4", "dim", "6", "9"]


def length(token: str) -> Fraction:
    """A cell token's length in quarter notes."""
    token = token.lstrip("r")
    return Fraction(4, int(token.rstrip("."))) * (2 - Fraction(1, 2 ** token.count(".")))


class Composer:
    """Draws a piece from a seed. `level` runs from 0, a beginner's page, to 3."""

    def __init__(self, seed: int | str):
        self.rng = random.Random(seed)
        self.next_id = 0

    def id(self, kind: str) -> str:
        self.next_id += 1
        return f"{kind}{self.next_id}"

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def piece(self) -> Piece:
        rng = self.rng
        self.level = level = rng.choices([0, 1, 2, 3], weights=[15, 30, 35, 20])[0]
        clef = rng.choices(list(CLEFS), weights=[58, 20, 8, 4, 10])[0]
        key = rng.choice([-2, -1, 0, 0, 1, 2]) if level == 0 else rng.choices(range(-6, 7), weights=[2, 4, 7, 10, 12, 12, 14, 12, 12, 10, 7, 4, 2])[0]
        time, beats = self.pick_time()
        self.clef, self.key, self.beats = clef, key, beats
        self.set_range()
        self.alter_chance = [0.0, 0.03, 0.08, 0.15][level] * rng.choice([0, 1, 1, 2])
        # An ensemble player's part, as a quartet's or an orchestra's: rests of many measures, slurs over most notes,
        # two notes at once, and the instrument's name at the top of the page. Real parts are where the first readers
        # went wrong (README.md, "What real pages changed").
        self.part = part = level >= 1 and self.chance(0.5)
        # How often a beat is a triplet. A piece in triplets has them on most beats, as a real one does.
        self.triplets = rng.choice([0.08, 0.2, 0.2, 0.45, 0.85]) if level >= 1 and self.chance(0.45) else 0.0
        self.triplet_kinds = rng.choice([(1, 0, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (2, 1, 1), (1, 0, 1)])   # eighths, quarters, sixes
        self.stops = rng.choice([0.04, 0.08, 0.15, 0.25]) if level >= 1 and self.chance(0.6 if part else 0.15) else 0.0
        self.rest_chance = rng.choice([0.0, 0.05, 0.1, 0.2])
        self.habits = rng.choice([(1, 0, 0, 0, 0), (6, 2, 1, 1, 1), (3, 2, 2, 2, 2), (2, 3, 1, 1, 3)])   # walk, scale, repeat, rock, arpeggio
        self.rock = 2

        measures: list[Measure] = []
        # Most pages are full, as printed music is; a part's always.
        count = rng.choice([32, 48, 64, 64, 80, 80, 96, 96, 120, 120, 160] if part else [8, 16, 32, 48, 64, 64, 80, 80, 96, 96, 120, 120])
        pickup = self.chance(0.2)
        changes = sorted(rng.sample(range(4, max(5, count - 2)), k=min(rng.choice([0, 0, 0, 1, 1, 2, 3]), max(0, count - 6)))) if count > 8 else []
        long_rests = rng.choice([0.04, 0.08, 0.15]) if part and self.chance(0.8) or self.chance(0.08) else 0.0
        whole_rests = rng.choice([0.03, 0.03, 0.08]) if part else 0.03
        for index in range(count):
            measure = Measure(self.id("m"))
            if index in changes:
                self.change(measure)
            if index == 0 and pickup:
                measure.pickup = True
                measure.cells = self.cells([(1, self.beats[-1][1])])
            elif long_rests and self.chance(long_rests * (3 if index == 0 else 1)) and not (measures and measures[-1].multirest) and index < count - 1:
                # Mostly a few measures, sometimes a whole section, as a part's rests are.
                measure.multirest = rng.choices([rng.randint(2, 9), rng.randint(10, 24), rng.randint(25, 99)], weights=[68, 24, 8])[0]
            elif index > 0 and self.chance(whole_rests):
                measure.whole_rest = True
            else:
                measure.cells = self.cells(self.beats)
            measures.append(measure)
        measures[-1].right = "end"
        if not measures[-1].multirest and not measures[-1].whole_rest:
            self.close(measures[-1])

        piece = Piece(self.title(key), rng.choice(COMPOSERS) if self.chance(0.7) else None, rng.choice(LABELS) if self.chance(0.4) else None, clef, key, time, measures)
        if part:
            self.head(piece)
        self.graces(piece)
        self.accidentals(piece)
        self.repeats(piece)
        self.tuplet_marks(piece)
        self.marks(piece)
        return piece

    def head(self, piece: Piece) -> None:
        """A part's page head: the work, its number, the movement, the composer, and whose part it is."""
        rng = self.rng
        numbers = {"a": rng.randint(1, 130), "b": rng.randint(1, 16), "c": rng.randint(1, 1080), "key": KEY_NAMES[piece.key % 13 if piece.key >= 0 else 7 - piece.key]}
        piece.part, piece.airy = True, self.chance(0.4)
        piece.title = rng.choice(PART_TITLES).format(**numbers)
        piece.subtitle = rng.choice(SUBTITLES).format(**numbers) if self.chance(0.7) else None
        piece.movement = rng.choice(MOVEMENTS) if self.chance(0.6) else None
        piece.composer = rng.choice(PART_COMPOSERS) if self.chance(0.9) else None
        piece.label = rng.choice(LABELS)

    def pick_time(self):
        choices = TIMES[:4] if self.level == 0 else TIMES
        time, beats, _ = self.rng.choices(choices, weights=[weight for *_, weight in choices])[0]
        return time, beats

    def set_range(self) -> None:
        _, _, _, low, high = CLEFS[self.clef]
        stretch = [0, 1, 3, 5][self.level]
        self.low, self.high = low - self.rng.randint(0, stretch), high + self.rng.randint(0, stretch)
        self.at = self.rng.randint(self.low + 2, self.high - 2)

    def change(self, measure: Measure) -> None:
        """A new key, time or clef from this measure on."""
        kind = self.rng.choice(["key", "key", "time", "clef"]) if self.level else "key"
        if kind == "key":
            measure.key = self.key = self.rng.choice([k for k in range(-6, 7) if k != self.key and abs(k - self.key) <= 4])
        elif kind == "time":
            (measure.time, self.beats) = self.pick_time()
        else:
            measure.clef = self.clef = self.rng.choice([c for c in CLEFS if c != self.clef])
            self.set_range()

    def cells(self, beats: list[tuple[int, str]]) -> list[Cell]:
        """Fills a run of beats with cells."""
        made = []
        for count, kind in beats:
            left = count
            while left:
                table = SIMPLE if kind == "s" else COMPOUND
                if kind == "s" and self.triplets and self.chance(self.triplets):
                    shape = self.rng.choices(["eighths", "quarters", "sixes"], weights=self.triplet_kinds)[0]
                    if shape == "quarters" and left >= 2:
                        made.append(self.cell(self.rng.choice(WIDE_TRIPLETS), (3, 2)))
                        left -= 2
                    elif shape == "sixes" and self.level >= 2:
                        made.append(self.cell(self.rng.choice(SIXES), (6, 4)))
                        left -= 1
                    else:
                        made.append(self.cell(self.rng.choice(TRIPLETS), (3, 2)))
                        left -= 1
                    continue
                span = self.rng.choice([s for s in table if s <= left for _ in range(3 if s == 1 else 1)])
                made.append(self.cell(self.rng.choice([cell for level, cell in table[span] if level <= self.level])))
                left -= span
        return made

    def cell(self, tokens: list[str], tuplet: tuple[int, int] | None = None) -> Cell:
        events = []
        figure = self.rng.choices(["walk", "scale", "repeat", "rock", "arpeggio"], weights=self.habits)[0]
        turn = self.rng.choice([-1, 1])
        for token in tokens:
            rest = token.startswith("r") or (not tuplet and len(tokens) == 1 and self.chance(self.rest_chance))
            event = Event(self.id("r" if rest else "n"), rest=rest, dur=int(token.lstrip("r").rstrip(".")), dots=token.count("."))
            if not rest:
                first = not any(not other.rest for other in events)
                event.di, event.alter = self.next_pitch() if figure == "walk" or first else self.figure_pitch(figure, turn, sum(not other.rest for other in events))
                if self.stops and not tuplet and event.dur <= 8 and self.chance(self.stops):
                    self.stack(event)
            events.append(event)
        return Cell(events, tuplet)

    def figure_pitch(self, figure: str, turn: int, number: int) -> tuple[int, int]:
        """The next note of a figure real tunes are full of: a scale, one note repeated, two notes rocking, a broken chord."""
        if figure == "rock":
            if number == 1:
                self.rock = self.rng.choice([1, 2, 2, 3, 4, 5, 7])
            move = turn * self.rock * (1 if number % 2 else -1)
        else:
            move = {"scale": turn, "repeat": 0, "arpeggio": 2 * turn}[figure]
        if not self.low <= self.at + move <= self.high:
            move = -move if self.low <= self.at - move <= self.high else 0
        self.at += move
        return self.at, self.altered(self.at)

    def altered(self, di: int) -> int:
        """The key signature's alteration of a note, changed by one now and then."""
        alter = key_alter(self.key, STEPS[di % 7])
        if self.chance(self.alter_chance):
            changed = alter + self.rng.choice([-1, 1])
            if -1 <= changed <= 1:
                alter = changed
        return alter

    def stack(self, event: Event) -> None:
        """Puts one or two more notes under a note, as a string player's double and triple stops are."""
        gaps = [gap for gap, _ in STOPS]
        below = event.di
        for _ in range(self.rng.choices([1, 2, 3], weights=[80, 16, 4])[0]):
            below -= self.rng.choices(gaps, weights=[weight for _, weight in STOPS])[0] if not event.under else self.rng.choice([3, 4, 4, 5, 5])
            if below < self.low - 3:
                break
            event.under.append(Event(self.id("n"), di=below, alter=self.altered(below), dur=event.dur, dots=event.dots))

    def next_pitch(self) -> tuple[int, int]:
        """Mostly steps, some thirds, a few leaps, as a tune moves; turned back at the ends of the range."""
        move = self.rng.choices([0, 1, 2, 3, 4, 5, 7], weights=[6, 52, 22, 9, 6, 3, 2])[0] * self.rng.choice([-1, 1])
        self.at += move if self.low <= self.at + move <= self.high else -move
        self.at = max(self.low, min(self.high, self.at))
        return self.at, self.altered(self.at)

    def close(self, measure: Measure) -> None:
        """Ends the piece on one long key note."""
        tonic = (self.key * 4) % 7   # the key note's letter, as a step: each sharp moves it up a fifth
        home = min((di for di in range(self.low, self.high + 1) if di % 7 == tonic), key=lambda di: abs(di - self.at))
        whole = sum((Fraction(count) * (1 if kind == "s" else Fraction(3, 2)) for count, kind in self.beats), Fraction(0))
        shapes = {Fraction(4): (1, 0), Fraction(3): (2, 1), Fraction(2): (2, 0), Fraction(6): (1, 1), Fraction(3, 2): (4, 1), Fraction(1): (4, 0)}
        if whole in shapes:
            dur, dots = shapes[whole]
            last = Event(self.id("n"), di=home, alter=key_alter(self.key, STEPS[home % 7]), dur=dur, dots=dots)
            measure.cells = [Cell([last])]
            if self.stops and self.chance(0.5):
                self.stack(last)   # a part often ends on a chord
        else:
            last = measure.events[-1]
            last.rest, last.di, last.alter, last.under = False, home, key_alter(self.key, STEPS[home % 7]), []

    def accidentals(self, piece: Piece) -> None:
        """Ties some notes, then decides which notes show an accidental: one that differs from what the key signature and
        the measure so far give its line or space."""
        events = [event for measure in piece.measures for event in measure.events]
        if self.level and self.chance(0.5):
            for first, second in zip(events, events[1:]):   # only to the very next note: a tie cannot pass a rest or a grace note
                if first.rest or second.rest or first.grace or second.grace:
                    continue
                if self.chance(0.06) and not first.tie and not first.under and not second.under:
                    second.di, second.alter = first.di, first.alter
                    first.tie, second.tie = "i", ("t" if second.tie is None else "m")
        show_accidentals(piece)

    def repeats(self, piece: Piece) -> None:
        measures = piece.measures
        if len(measures) < 12 or not self.chance(0.25):
            if len(measures) > 10 and self.chance(0.3):
                self.rng.choice(measures[3:-3]).right = "dbl"
            return
        start = self.rng.randrange(0, len(measures) - 9)
        end = start + self.rng.choice([4, 6, 8])
        if start:
            measures[start].left = "rptstart"
        if self.chance(0.5):
            measures[end - 1].ending, measures[end].ending = 1, 2
        measures[end - 1].right = "rptend"

    def marks(self, piece: Piece) -> None:
        """The signs around the notes: words, loudness, slurs, articulation, fingering, lyrics, chord names."""
        rng, level = self.rng, self.level
        measures = piece.measures
        notes = [event for measure in measures for event in measure.events if not event.rest and not event.grace]
        if not notes:
            return
        if self.chance(0.9 if piece.part else 0.55):
            measures[0].tempo = rng.choice(TEMPI)
        style = "part" if piece.part else rng.choices(["plain", "marked", "song", "lead", "fingered"], weights=[25, 40, 12, 10, 13])[0]
        part = style == "part"
        letters = iter("ABCDEFGHJK")
        for index, measure in enumerate(measures):
            if not measure.events:
                continue
            if style != "plain" and index and index % 8 == 0 and self.chance(0.3):
                measure.rehearsal = next(letters, None)
            if style in ("marked", "fingered", "part") and self.chance(0.1 if part else 0.06):
                measure.direction = rng.choice(PART_DIRECTIONS if part else DIRECTIONS)
            first = next((event for event in measure.events if not event.rest and not event.grace), None)
            if first and style != "plain" and (index == 0 or self.chance(0.2 if part else 0.12)):
                first.dynamic = rng.choice(PART_DYNAMICS if part else DYNAMICS)
            if first and style == "lead" and self.chance(0.8):
                first.chord = "ABCDEFG"[rng.randrange(7)] + rng.choice(CHORD_KINDS)
        if style in ("marked", "fingered", "part"):
            # A part is slurred and dotted throughout or hardly at all, as a real one is.
            slurred = rng.choice([0.08, 0.2, 0.35, 0.5]) if part else 0.12
            dotted = rng.choice([0.03, 0.08, 0.2, 0.4]) if part else 0.08 * (level + 1) / 2
            shaking = rng.choice([0.05, 0.3]) if part and self.chance(0.12) else 0.0
            for event in notes:
                if self.chance(dotted):
                    event.artics.append(rng.choice(PART_ARTICULATIONS if part else ARTICULATIONS))
                if level >= 2 and event.dur <= 2 and self.chance(0.05):
                    event.trill = True
                elif shaking and event.dur <= 8 and not event.tie and self.chance(shaking):
                    event.tremolo = rng.choice([1, 2, 3, 3])
            index = 0
            while index < len(notes) - 2:
                if self.chance(slurred):
                    span = rng.choice([1, 1, 1, 2, 2, 3, 3, 4, 5, 7]) if part else rng.randint(1, 5)
                    last = min(index + span, len(notes) - 1)
                    piece.slurs.append((notes[index].id, notes[last].id))
                    index = last + 1
                elif self.chance(0.05 if part else 0.03):
                    last = min(index + rng.randint(2, 6), len(notes) - 1)
                    piece.hairpins.append((rng.choice(["cres", "dim"]), notes[index].id, notes[last].id))
                    index = last + 1
                else:
                    index += 1
        if self.chance(0.25):
            rng.choice(notes[len(notes) // 2:]).fermata = True
        if style == "fingered":
            top = rng.choice([4, 5])
            for event in notes:
                if self.chance(0.2):
                    event.fingering = str(rng.randint(0 if top == 4 else 1, top))
        if style in ("song", "lead"):
            word: list[str] = []
            for event in notes:
                if event.tie in ("t", "m"):
                    continue
                if not word:
                    word = list(rng.choice(SYLLABLES))
                    place = "s" if len(word) == 1 else "i"
                else:
                    place = "t" if len(word) == 1 else "m"
                event.lyric = (word.pop(0), place)

    def graces(self, piece: Piece) -> None:
        """Grace notes before some beats: one small note, slashed or not, or a run of two to four beamed ones that
        climbs or falls into the note, often slurred to it."""
        rng = self.rng
        if not (self.level >= 1 and self.chance(0.4 if self.part else 0.2)):
            return
        rate = rng.choice([0.03, 0.06, 0.12, 0.25])
        key = piece.key
        for measure in piece.measures:
            key = measure.key if measure.key is not None else key
            for cell in measure.cells:
                main = cell.events[0]
                if cell.tuplet or main.rest or not self.chance(rate):
                    continue
                size = rng.choices([1, 2, 3, 4], weights=[55, 20, 20, 5])[0]
                dur, slash = (rng.choice([(8, True), (8, True), (8, False), (16, False)]) if size == 1 else (rng.choice([16, 16, 16, 32]), False))
                turn = rng.choice([-1, 1])
                run = []
                for step in range(size, 0, -1):
                    di = max(self.low - 2, min(self.high + 2, main.di + turn * step))
                    alter = key_alter(key, STEPS[di % 7])
                    if self.chance(0.1) and -1 <= alter + turn <= 1:
                        alter += turn    # a leading note: raised on the way up, lowered on the way down
                    run.append(Event(self.id("n"), di=di, alter=alter, dur=dur, grace=True, slash=slash))
                cell.events[0:0] = run
                if self.chance(0.6):
                    piece.slurs.append((run[0].id, main.id))

    def tuplet_marks(self, piece: Piece) -> None:
        """How each triplet is marked. Beamed ones carry their number alone, as most engravers draw them; unbeamed
        ones a bracket and number, or the number alone. Once a piece has shown a few, it may leave the rest unmarked:
        a reader then has only the beaming and the measure's length to go on."""
        rng = self.rng
        beamed_show = rng.choice(["number", "number", "number", "number", "bracket"])
        open_show = rng.choice(["bracket", "bracket", "number"])
        side = rng.choice([None, None, None, True, False])
        unmarked_after = rng.choice([1, 2, 4, 8]) if self.chance(0.2) else None
        seen = 0
        for measure in piece.measures:
            for cell in measure.cells:
                if not cell.tuplet:
                    continue
                beamed = all(not event.rest and event.dur >= 8 for event in cell.events)
                cell.show = beamed_show if beamed else open_show
                cell.above = side
                if beamed and unmarked_after is not None and seen >= unmarked_after:
                    cell.show = "none"
                seen += 1

    def title(self, key: int) -> str:
        rng = self.rng
        name = rng.choice(TITLE_A)
        kind = rng.randrange(5)
        if kind == 0:
            return f"{name} No. {rng.randint(1, 48)}"
        if kind == 1:
            return f"{name} in {KEY_NAMES[key % 13 if key >= 0 else 7 - key]}"
        if kind == 2:
            return f"{name} {rng.choice(TITLE_B)}"
        if kind == 3:
            return f"{rng.randint(1, 60)}. {name}"
        return name


def show_accidentals(piece: Piece) -> None:
    """Decides which notes show an accidental: one that differs from what the key signature and the measure so far
    give its line or space. A note tied from the one before shows none."""
    key = piece.key
    for measure in piece.measures:
        key = measure.key if measure.key is not None else key
        state: dict[int, int] = {}
        for event in measure.events:
            if event.rest:
                continue
            for note in [event] + event.under:   # a chord's notes from the top down, as its tokens run
                expected = state.get(note.di, key_alter(key, note.step))
                note.shown = None
                if note.tie in ("t", "m"):
                    state[note.di] = note.alter
                elif note.alter != expected:
                    note.shown = ACCIDENTAL[note.alter]
                    state[note.di] = note.alter


def beam_runs(cell: Cell) -> list[list[Event]]:
    """The cell's events in runs, each run of two or more under one beam. A made-up piece beams every run of eighths
    or shorter; a real score's beams are the ones its file gives."""
    runs: list[list[Event]] = []
    if cell.given:
        for place, event in enumerate(cell.events):
            open_beam = next((other.beam for other in reversed(runs[-1]) if not other.grace), None) if runs else None
            # A grace note between two notes of one beam stays under it.
            inside = event.grace and event.beam is None and open_beam is not None and any(other.beam == open_beam and not other.grace for other in cell.events[place + 1:])
            if runs and open_beam is not None and (event.beam == open_beam or inside):
                runs[-1].append(event)
            else:
                runs.append([event])
        return runs
    for event in cell.events:
        beamable = not event.rest and event.dur >= 8
        if runs and beamable and all(not e.rest and e.dur >= 8 and e.grace == event.grace for e in runs[-1]):
            runs[-1].append(event)   # grace notes beam with each other, never with the notes they lead to
        else:
            runs.append([event])
    return runs


def event_mei(event: Event) -> str:
    if event.rest:
        dots = f' dots="{event.dots}"' if event.dots else ""
        return f'<rest xml:id="{event.id}" dur="{event.dur}"{dots}/>'
    length = f'dur="{event.dur}"' + (f' dots="{event.dots}"' if event.dots else "")
    if event.grace:
        length += ' grace="unacc" stem.mod="1slash"' if event.slash else ' grace="acc"'
    elif event.tremolo:
        length += f' stem.mod="{event.tremolo}slash"'

    def pitch(note: Event, tied: bool = False) -> str:
        shown = f' accid="{note.shown}"' if note.shown else (f' accid.ges="{SOUNDING[note.alter]}"' if note.alter else "")
        tie = f' tie="{note.tie}"' if tied and note.tie else ""   # each note of a chord is tied on its own
        return f'xml:id="{note.id}" pname="{note.step}" oct="{note.octave}"{shown}{tie}'

    inner = "".join(f'<artic xml:id="{event.id}a{i}" artic="{artic}"/>' for i, artic in enumerate(event.artics))
    if event.lyric:
        text, place = event.lyric
        syllable = f'<syl wordpos="{place}" con="d">' if place in ("i", "m") else ("<syl>" if place == "s" else '<syl wordpos="t">')
        inner += f'<verse xml:id="{event.id}v" n="1">{syllable}{escape(text)}</syl></verse>'
    if event.under:
        notes = "".join(f"<note {pitch(note, True)}/>" for note in [event] + event.under)
        return f'<chord xml:id="{event.id}k" {length}>{inner}{notes}</chord>'
    attributes = f"{pitch(event)} {length}"
    if event.tie:
        attributes += f' tie="{event.tie}"'
    return f"<note {attributes}>{inner}</note>" if inner else f"<note {attributes}/>"


def measure_mei(measure: Measure, number: int, piece: Piece, next_clef: str | None) -> str:
    attributes = f'xml:id="{measure.id}" n="{number}"'
    if measure.pickup:
        attributes += ' metcon="false"'
    if measure.left:
        attributes += f' left="{measure.left}"'
    if measure.right:
        attributes += f' right="{measure.right}"'
    layer = ""
    if measure.multirest:
        layer = f'<multiRest xml:id="{measure.id}r" num="{measure.multirest}"/>'
    elif measure.whole_rest:
        layer = f'<mRest xml:id="{measure.id}r"/>'
    def under_one_beam(cell: Cell) -> int | None:
        """The beam a real score's cell lies wholly under, if it does."""
        beams = {event.beam for event in cell.events}
        return cell.events[0].beam if cell.given and len(beams) == 1 else None

    cells = measure.cells
    at = 0
    while at < len(cells):
        # One beam over several cells, as over two triplets: the cells go inside the beam.
        beam = under_one_beam(cells[at])
        end = at + 1
        while beam is not None and end < len(cells) and under_one_beam(cells[end]) == beam:
            end += 1
        joined = end - at > 1
        group = ""
        for cell in cells[at:end]:
            inner = ""
            for run in ([cell.events] if joined else beam_runs(cell)):
                notes = "".join(event_mei(event) for event in run)
                inner += f"<beam>{notes}</beam>" if len(run) > 1 and not joined else notes
            if cell.rock:
                # The strokes a note's own length already has are drawn as beams from stem to stem; the rest float between.
                own = {8: 1, 16: 2, 32: 3}.get(cell.events[0].dur, 0)
                inner = f'<fTrem beams="{max(cell.rock, own)}" beams.float="{max(cell.rock - own, 0)}">{inner}</fTrem>'
            if cell.tuplet:
                marked = f'num.visible="{str(cell.show != "none").lower()}" bracket.visible="{str(cell.show == "bracket").lower()}"'
                if cell.above is not None and cell.show != "none":
                    marked += f' num.place="{"above" if cell.above else "below"}" bracket.place="{"above" if cell.above else "below"}"'
                inner = f'<tuplet num="{cell.tuplet[0]}" numbase="{cell.tuplet[1]}" {marked}>{inner}</tuplet>'
            group += inner
        layer += f"<beam>{group}</beam>" if joined else group
        at = end
    if next_clef:
        shape, line, displaced, *_ = CLEFS[next_clef]
        octave = ' dis="8" dis.place="below"' if displaced else ""
        layer += f'<clef xml:id="{measure.id}c" shape="{shape}" line="{line}"{octave}/>'
    controls = ""
    if measure.tempo:
        controls += f'<tempo xml:id="{measure.id}t" tstamp="1" staff="1" place="above">{escape(measure.tempo)}</tempo>'
    if measure.rehearsal:
        controls += f'<reh xml:id="{measure.id}h" tstamp="1" staff="1"><rend rend="box">{measure.rehearsal}</rend></reh>'
    events = measure.events
    first = next((event for event in events if not event.rest), None)
    if measure.direction and first:
        controls += f'<dir xml:id="{measure.id}d" startid="#{first.id}" staff="1" place="below">{escape(measure.direction)}</dir>'
    for event in events:
        if event.dynamic:
            controls += f'<dynam xml:id="{event.id}y" startid="#{event.id}" staff="1" place="below">{event.dynamic}</dynam>'
        if event.fermata:
            controls += f'<fermata xml:id="{event.id}u" startid="#{event.id}" staff="1" place="above"/>'
        if event.trill:
            controls += f'<trill xml:id="{event.id}w" startid="#{event.id}" staff="1"/>'
        if event.fingering:
            controls += f'<fing xml:id="{event.id}f" startid="#{event.id}" staff="1" place="above">{event.fingering}</fing>'
        if event.chord:
            controls += f'<harm xml:id="{event.id}c" startid="#{event.id}" staff="1" place="above">{escape(event.chord)}</harm>'
    ids = {event.id for event in events}
    for index, (start, end) in enumerate(piece.slurs):
        if start in ids:
            controls += f'<slur xml:id="s{index}" startid="#{start}" endid="#{end}" staff="1"/>'
    for index, (form, start, end) in enumerate(piece.hairpins):
        if start in ids:
            controls += f'<hairpin xml:id="h{index}" form="{form}" startid="#{start}" endid="#{end}" staff="1" place="below"/>'
    return f'<measure {attributes}><staff n="1"><layer n="1">{layer}</layer></staff>{controls}</measure>'


def key_signature(sharps: int) -> str:
    return "0" if sharps == 0 else f"{abs(sharps)}{'s' if sharps > 0 else 'f'}"


def time_attributes(time: tuple[int, int, str | None]) -> str:
    count, unit, symbol = time
    return f'meter.count="{count}" meter.unit="{unit}"' + (f' meter.sym="{symbol}"' if symbol else "")


def to_mei(piece: Piece) -> str:
    """The piece as MEI 5, with our ids on everything the truth is read from."""
    shape, line, displaced, *_ = CLEFS[piece.clef]
    body = ""
    open_ending = None
    measures = piece.measures
    for index, measure in enumerate(measures):
        if measure.ending != open_ending:
            body += "</ending>" if open_ending else ""
            body += f'<ending n="{measure.ending}">' if measure.ending else ""
            open_ending = measure.ending
        if measure.key is not None or measure.time is not None:
            changes = (f' key.sig="{key_signature(measure.key)}"' if measure.key is not None else "") + (" " + time_attributes(measure.time) if measure.time else "")
            body += f"<scoreDef{changes}/>"
        following = measures[index + 1].clef if index + 1 < len(measures) else None
        body += measure_mei(measure, index + (0 if measures[0].pickup else 1), piece, following)
    body += "</ending>" if open_ending else ""
    composer = f'<respStmt><persName role="composer">{escape(piece.composer)}</persName></respStmt>' if piece.composer else ""
    label = f"<label>{escape(piece.label)}</label>" if piece.label and not piece.part else ""
    head = ""
    if piece.part:
        # A part's page head, as publishers lay it out: whose part at the top left, the work in the middle with its
        # number and movement under it, the composer at the right.
        def words(across: str, down: str, text: str | None) -> str:
            return f'<rend halign="{across}" valign="{down}">{escape(text)}</rend>' if text else ""

        # The lines under the title are set inside its block, each after a line break: lines given places of their
        # own are drawn on top of each other. A second break opens the head out, as some publishers set it.
        gap = "<lb/><lb/>" if piece.airy else "<lb/>"
        under = "".join(f'{gap}<rend fontsize="{"large" if piece.airy else "small"}">{escape(text)}</rend>' for text in (piece.subtitle, piece.movement) if text)
        size = "" if piece.airy else ' fontsize="x-large"'
        title = f'<rend fontsize="x-large">{escape(piece.title)}</rend>' if piece.airy else escape(piece.title)
        head = f'<pgHead>{words("left", "top", piece.label)}<rend halign="center" valign="top"{size}>{title}{under}</rend>{words("right", "bottom", piece.composer)}</pgHead>'
    clef = f'clef.shape="{shape}" clef.line="{line}"' + (' clef.dis="8" clef.dis.place="below"' if displaced else "")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.0">'
        f"<meiHead><fileDesc><titleStmt><title>{escape(piece.title)}</title>{composer}</titleStmt><pubStmt/></fileDesc></meiHead>"
        f'<music><body><mdiv><score><scoreDef key.sig="{key_signature(piece.key)}" {time_attributes(piece.time)}>{head}'
        f'<staffGrp><staffDef n="1" lines="5" {clef}>{label}</staffDef></staffGrp></scoreDef>'
        f"<section>{body}</section></score></mdiv></body></music></mei>\n"
    )


def compose(seed: int | str) -> Piece:
    return Composer(seed).piece()
