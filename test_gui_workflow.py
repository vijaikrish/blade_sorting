"""Real Qt workflow acceptance and state isolation checks."""

import csv
import pytest
from PySide6.QtCore import QThread
from data_io import sample_blades
from gui import MainWindow
from solver import SolverConfig, solve
from rotor_math import slot_geometry


def prepared(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_blades(sample_blades(8))
    return w


def present_result(w):
    blades = w.get_blades()
    rotor = w._input_model()
    geom = slot_geometry(len(blades), rotor)
    result = solve(
        [b.weight for b in blades],
        geom.x_mm,
        geom.y_mm,
        (rotor.disk_ux_gmm, rotor.disk_uy_gmm),
        SolverConfig(iterations=100, restarts=1),
    )
    w.show_result(result, blades, rotor, geom)
    return result


def test_background_result_rendering_runs_on_gui_thread(qtbot, qapp):
    w = prepared(qtbot)
    seen = []
    original = w._draw_result

    def draw(*args):
        seen.append(QThread.currentThread() == qapp.thread())
        original(*args)

    w._draw_result = draw
    w.start_optimization()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.result is not None
    assert seen == [True]


def test_cancel_discards_complete_or_partial_worker_result(qtbot):
    w = prepared(qtbot)
    w.start_optimization()
    w._cancel_work()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.result is None
    assert not w.export_button.isEnabled()
    assert "cancel" in w.status_label.text().lower()


def test_edits_while_running_discard_snapshot(qtbot):
    w = prepared(qtbot)
    w.start_optimization()
    w.seed_input.setValue(7)
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.result is None and not w.export_button.isEnabled()
    assert "stale" in w.status_label.text().lower()


def test_import_is_transactional_and_schedule_exports_real_ids(qtbot, tmp_path):
    w = prepared(qtbot)
    before = w.get_blades()
    present_result(w)
    p = tmp_path / "bad.csv"
    p.write_text("blade_name,weight\nA,100\nB,bad\n", encoding="utf-8")
    with pytest.raises(ValueError):
        w.import_csv_path(p)
    assert w.get_blades() == before and w.result is not None
    out = tmp_path / "mounting.csv"
    w.export_csv_path(out)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(before)
    assert {r["blade_name"] for r in rows} == {b.blade_name for b in before}
    assert [int(r["slot_number"]) for r in rows] == list(range(1, len(before) + 1))


@pytest.mark.parametrize(
    "name",
    ["diameter_input", "thickness_input", "density_input", "radius_input", "seed_input", "blade_count"],
)
def test_every_numeric_input_invalidates_schedule(qtbot, name):
    w = prepared(qtbot)
    present_result(w)
    field = getattr(w, name)
    field.setValue(field.value() + 1)
    assert w.result is None and not w.export_button.isEnabled()


def test_auto_radius_diameter_edit_invalidates_export(qtbot):
    w = prepared(qtbot)
    assert w.radius_auto.isChecked()
    present_result(w)
    assert w.export_button.isEnabled()
    w.diameter_input.setValue(w.diameter_input.value() + 2)
    assert w.radius_input.value() == w.diameter_input.value() / 2
    assert w.result is None and not w.export_button.isEnabled()
    with pytest.raises(ValueError, match="No current valid result"):
        w.export_csv_path("should-not-exist.csv")


def test_result_contains_heuristic_comparison_and_engineering_model_details(qtbot):
    w = prepared(qtbot)
    present_result(w)
    assert hasattr(w, "comparison_label")
    text = w.comparison_label.text().lower()
    assert all(term in text for term in ("initial", "greedy", "pairing", "floor", "g·mm"))
    assert hasattr(w, "rotor_details_label")
    assert "phase" in w.rotor_details_label.text().lower()


def test_worker_failure_clears_previous_export_and_restores_controls(qtbot, monkeypatch):
    import gui

    w = prepared(qtbot)
    present_result(w)

    def broken(*args, **kwargs):
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(gui, "solve", broken)
    w.start_optimization()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert "injected worker failure" in w.status_label.text()
    assert w.result is None and not w.export_button.isEnabled()
    assert w.optimize_button.isEnabled()
