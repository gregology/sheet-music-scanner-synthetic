import json
import subprocess
import sys
import tarfile


# A run stopped mid-example leaves a photo with no truth. Packed, it would be an example with no answer.
def test_only_whole_examples_are_packed(tmp_path):
    shard = tmp_path / "made" / "00000"
    shard.mkdir(parents=True)
    (tmp_path / "made" / "dataset.json").write_text(json.dumps({"format": 1, "seed": 1, "width": 30, "height": 40, "samples": 4, "commits": ["abc1234"]}))
    for number in (0, 1):
        (shard / f"{number:08d}.jpg").write_bytes(b"photo")
        (shard / f"{number:08d}.json").write_text("{}")
    (shard / "00000002.jpg").write_bytes(b"photo")   # its truth was never written
    (shard / "00000003.failed").write_text("RuntimeError: the render does not show the page its truth describes\n")
    subprocess.run([sys.executable, "-m", "sheet_synth.pack", str(tmp_path / "made"), str(tmp_path / "packed")], check=True)
    with tarfile.open(tmp_path / "packed" / "train-00000.tar") as tar:
        assert tar.getnames() == ["00000000.jpg", "00000000.json", "00000001.jpg", "00000001.json"]
    card = (tmp_path / "packed" / "README.md").read_text()
    assert "license: cc-by-4.0" in card and "abc1234" in card and "2 synthetic phone photos" in card
    assert "Numbers with no example:** 1," in card


# PDMX is under CC BY 4.0: its scores may go into a dataset only with its credit. A card that still said the music
# was all made up would publish them without it.
def test_a_dataset_with_real_scores_carries_their_credit():
    from sheet_synth.pack import credit

    music, credits = credit({"real": 0.5, "scores": ["lieder@38c5db510224/e2f2c3c7b6ba", "pdmx@15571083/0123456789ab"]})
    assert "50%" in music and "No real music" not in music
    assert "Phillip Long" in credits and "CC BY 4.0" in credits and "zenodo.org/records/15571083" in credits and "OpenScore Lieder" in credits
    assert credit({"seed": 1}) == ("- **No real music.** The tunes are random walks with the notation of real music and not its habits.", "")
    assert "Phillip Long" not in credit({"real": 1.0, "scores": "lieder@38c5db510224/e2f2c3c7b6ba"})[1]


# A failed example was once only logged. Its shard never reached a thousand examples, and the training that waits
# for whole shards waited for good. Made on a later run, it must not stay marked as failed.
def test_a_failed_example_is_recorded_in_its_shard_until_it_is_made(tmp_path, monkeypatch):
    from sheet_synth import example, generate

    def fails(*_arguments, **_options):
        raise RuntimeError("the render does not show the page its truth describes: ink on 14% of its notes")

    monkeypatch.setenv("SHEET_SYNTH_DEVICE", "CPU")
    monkeypatch.setattr(sys, "argv", ["generate", "--out", str(tmp_path), "--count", "2"])
    monkeypatch.setattr(example, "make_example", fails)
    generate.main()
    shard = tmp_path / "00000"
    assert sorted(path.name for path in shard.iterdir()) == ["00000000.failed", "00000001.failed"]
    assert "ink on 14%" in (shard / "00000000.failed").read_text()
    monkeypatch.setattr(example, "make_example", lambda *_arguments, **_options: (b"photo", {}))
    generate.main()
    assert sorted(path.name for path in shard.iterdir()) == ["00000000.jpg", "00000000.json", "00000001.jpg", "00000001.json"]
