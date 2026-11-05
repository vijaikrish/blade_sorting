import csv
import math

import pytest

from data_io import Blade, export_schedule_csv, read_blades_csv, sample_blades, validate_blades


def test_validate_preserves_exact_names_and_returns_immutable_blades():
    result = validate_blades([Blade("  Rotor, α ", 1.25), Blade("B-2", 2)])
    assert result == (Blade("  Rotor, α ", 1.25), Blade("B-2", 2.0))


def test_validate_rejects_invalid_population():
    with pytest.raises(ValueError):
        validate_blades([Blade("a", 1)])
    with pytest.raises(ValueError):
        validate_blades([Blade("a", 1), Blade("a", 2)])
    with pytest.raises(ValueError):
        validate_blades([Blade("a", True), Blade("b", 2)])


def test_csv_reads_bom_semicolon_and_preserves_identifiers(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("\ufeffblade_name;weight\nRotor, α;101.5\nB-2;99\n", encoding="utf-8")
    assert read_blades_csv(p) == (Blade("Rotor, α", 101.5), Blade("B-2", 99.0))


def test_csv_rejects_bad_weight_with_row_number(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("blade_name,weight\na,1\nb,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 3"):
        read_blades_csv(p)


def test_sample_blades_is_deterministic_and_variable_size():
    assert sample_blades(3, 7) == sample_blades(3, 7)
    assert len(sample_blades(4)) == 4
    assert len(sample_blades(5)) == 5


def test_export_angles_avoid_binary_float_artifacts(tmp_path):
    p = tmp_path / "angle.csv"
    export_schedule_csv(p, (Blade("A", 10), Blade("B", 11)), (0, 1),
                        (math.radians(25), math.radians(39.400000000000006)))
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["angle_deg"] for row in rows] == ["25", "39.4"]


def test_export_exact_schema_roundtrip_and_no_overwrite(tmp_path):
    p = tmp_path / "schedule.csv"
    blades = (Blade("Rotor, α", 10), Blade("B-2", 11))
    export_schedule_csv(p, blades, (1, 0), (0.0, math.pi))
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["slot_number", "angle_deg", "blade_name", "weight_g"]
    assert rows[1][2] == "B-2"
    with pytest.raises(FileExistsError):
        export_schedule_csv(p, blades, (0, 1), (0.0, math.pi))


def test_export_guards_formula_ids_and_failed_export_keeps_original(tmp_path):
    p = tmp_path / "schedule.csv"
    p.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_schedule_csv(p, (Blade("=1+1", 10), Blade(" safe", 11)), (0, 1), (0.0, 1.0))
    assert p.read_text(encoding="utf-8") == "untouched"
    export_schedule_csv(p, (Blade("=1+1", 10), Blade(" safe", 11)), (0, 1), (0.0, 1.0), overwrite=True)
    assert "'=1+1" in p.read_text(encoding="utf-8")


def test_export_rejects_invalid_permutation_before_file_write(tmp_path):
    p = tmp_path / "invalid.csv"
    with pytest.raises(ValueError):
        export_schedule_csv(p, (Blade("A", 1), Blade("B", 2)), (0, 0), (0.0, 1.0))
    assert not p.exists()


@pytest.mark.parametrize(
    "raw",
    [
        'blade_name,weight\n"A"junk,100\nB,101\n',
        "blade_name,weight,weight\nA,100,100\nB,101,101\n",
        "blade_name,weight\nA,100\nB,0\n",
        "blade_name,weight\nA,100\nB,inf\n",
        "blade_name,weight\nA,100\nA,101\n",
    ],
)
def test_csv_rejects_malformed_records(tmp_path, raw):
    p = tmp_path / "bad.csv"
    p.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        read_blades_csv(p)


def test_failed_new_export_does_not_leave_partial_schedule(tmp_path, monkeypatch):
    import data_io

    original_writer = data_io.csv.writer

    class BrokenWriter:
        def __init__(self, stream):
            self.wrapped = original_writer(stream)

        def writerow(self, row):
            self.wrapped.writerow(row)

        def writerows(self, rows):
            raise OSError("simulated disk error")

    monkeypatch.setattr(data_io.csv, "writer", BrokenWriter)
    target = tmp_path / "schedule.csv"
    with pytest.raises(OSError, match="simulated disk error"):
        export_schedule_csv(target, (Blade("A", 100), Blade("B", 101)), (0, 1), (0.0, 1.0))
    assert not target.exists(), "a failed export must not look like a mounting schedule"
    assert list(tmp_path.iterdir()) == []


def test_export_accepts_numpy_integer_permutation(tmp_path):
    import numpy as np

    target = tmp_path / "numpy.csv"
    export_schedule_csv(
        target, (Blade("A", 100), Blade("B", 101)), np.array([1, 0]), np.array([0.0, math.pi])
    )
    assert target.exists()


@pytest.mark.parametrize("overwrite", [1, "yes", None])
def test_overwrite_requires_explicit_boolean(tmp_path, overwrite):
    target = tmp_path / "consent.csv"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        export_schedule_csv(
            target, (Blade("A", 100), Blade("B", 101)), (0, 1), (0.0, math.pi), overwrite=overwrite
        )
    assert target.read_text(encoding="utf-8") == "original"
