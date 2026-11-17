import random

import numpy as np

from sheet_synth import camera
from sheet_synth.example import place, printed_page
from sheet_synth.geometry import Camera, Sheet, page_grid, to_picture


def marked_page():
    """The first few examples' pages, and the first that carries handwriting."""
    pages = [printed_page(3, index) for index in range(12)]
    return pages, next(page for page in pages if page[1]["handwriting"])


PAGES, (MARKED, MARKED_TRUTH) = marked_page()


# Print faults or handwriting that wiped out the notes would leave photos whose truth names notes nobody can see.
def test_the_printed_page_keeps_ink_on_every_note():
    for sheet, truth in PAGES:
        height, width = sheet.shape[:2]
        paper = float(np.median(sheet))
        for system in truth["systems"]:
            for symbol in system["symbols"]:
                if symbol["type"] == "note":
                    x0, y0, x1, y1 = symbol["box"]
                    inside = sheet[int(y0 * height):int(y1 * height) + 1, int(x0 * width):int(x1 * width) + 1]
                    assert inside.min() < paper - 0.15, symbol["id"]


# A finger number recorded on a note that is not on the page, or outside the page, is a label for nothing.
def test_handwritten_finger_numbers_name_notes_on_the_page():
    notes = {symbol["id"] for system in MARKED_TRUTH["systems"] for symbol in system["symbols"]}
    fingerings = [mark for mark in MARKED_TRUTH["handwriting"] if mark["type"] == "fingering"]
    for mark in MARKED_TRUTH["handwriting"]:
        x0, y0, x1, y1 = mark["box"]
        assert -0.02 <= x0 < x1 <= 1.02 and -0.02 <= y0 < y1 <= 1.02, mark
    for mark in fingerings:
        assert mark["of"] in notes and mark["text"] in "012345"


# Handwriting that leaves no trace on the page would be truth without a picture.
def test_handwriting_darkens_the_page_where_the_truth_says():
    height, width = MARKED.shape[:2]
    for mark in MARKED_TRUTH["handwriting"]:
        if mark["type"] == "highlight":
            continue
        x0, y0, x1, y1 = mark["box"]
        inside = MARKED[int(y0 * height):int(y1 * height) + 1, int(x0 * width):int(x1 * width) + 1]
        assert inside.min() < np.median(MARKED) - 0.1, mark


# The same seed giving a different page would make the dataset impossible to regenerate or to resume.
def test_the_same_seed_prints_the_same_page():
    again, truth = printed_page(3, 0)
    assert truth == PAGES[0][1]
    assert np.array_equal(again, PAGES[0][0])


# Truth placed in the photo must cover everything the page's truth holds, or a reader's labels have holes.
def test_every_symbol_and_mark_is_placed_in_the_photo():
    sheet = Sheet.draw(random.Random(1), 0.2159, 0.2794)
    view, _ = Camera.draw(random.Random(2), sheet, 768, 1024)
    truth = printed_page(3, 1)[1]
    place(truth, page_grid(sheet, view), 768, 1024)
    assert len(truth["page"]["grid_px"]) == 33 and len(truth["page"]["grid_px"][0]) == 25
    for system in truth["systems"]:
        assert len(system["staff"]["lines_px"]) == 5
        assert all(len(item["quad"]) == 4 for item in system["symbols"] + system["marks"])


# Low light that came out as clean as daylight would teach nothing about the photos people take in the evening.
def test_less_light_makes_a_noisier_photo():
    flat = np.full((256, 192, 3), 0.5, np.float32)
    settings = camera.draw_settings(random.Random(0)) | {"shake_px": 0.0, "defocus_px": 0.0, "vignette": 0.0, "smoothing": 0.0, "sharpen": 0.0, "shadow_lift": 0.0}
    bright = camera.develop(flat, settings | {"electrons": 30000}, 1)
    dark = camera.develop(flat, settings | {"electrons": 100}, 1)
    assert bright.shape == (256, 192, 3) and bright.dtype == np.uint8
    assert dark.std() > 5 * bright.std()


# A photocopy effect once erased every staff line on the page, leaving notes floating on blank paper. The thinnest
# staff lines the engraver draws must survive every way of printing.
def test_staff_lines_survive_every_kind_of_print():
    import io

    from PIL import Image

    from sheet_synth.compose import Composer
    from sheet_synth.engrave import Style, engrave
    from sheet_synth.paper import PRINTS, print_page

    style = Style(scale=30, options={"staffLineWidth": 0.1})
    page = engrave(Composer(4).piece(), style)
    grey = np.asarray(Image.open(io.BytesIO(page.raster(400))).convert("L"), np.float32) / 255
    height, width = grey.shape
    seen = set()
    for seed in range(60):
        sheet, made = print_page(grey, random.Random(seed), 400)
        if made["print"] in seen:
            continue
        seen.add(made["print"])
        paper = float(np.median(sheet))
        for system in page.truth["systems"]:
            staff = system["staff"]
            for line in staff["lines"]:
                row = sheet[int(line * height) - 3:int(line * height) + 4, int(staff["left"] * width):int(staff["right"] * width)].mean(axis=2).min(axis=0)
                assert np.mean(row < paper - 0.08) > 0.8, (made["print"], made)   # ink along at least four fifths of the line
        if seen == set(PRINTS):
            break
    assert seen == set(PRINTS)


# A renderer once drew an earlier example's page under a later example's truth: every label wrong, nothing visibly
# amiss. The guard must pass a picture that shows the truth's notes and fail one that shows another page.
def test_a_picture_of_another_page_is_caught():
    from sheet_synth.example import shows_its_page

    (sheet, truth), (other, _) = PAGES[1], PAGES[2]
    height, width = sheet.shape[:2]
    lattice = np.stack(np.meshgrid(np.linspace(0, width, 49), np.linspace(0, height, 65)), axis=-1)   # the flat page as its own photo
    place(truth, lattice, width, height)
    assert shows_its_page(sheet, truth) > 0.9
    assert shows_its_page(other[:height, :width], truth) < 0.5
