"""Physical closing-slot configuration must reach the real GUI worker."""
import pytest
from data_io import closure_demo_blades
from gui import MainWindow


def make_window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_blades(closure_demo_blades())
    w.closure_enabled.setChecked(True)
    w.closure_slot_input.setValue(17)  # one-based physical slot
    w.closure_blade_combo.setCurrentIndex(w.closure_blade_combo.findData("Closing-100"))
    return w


def test_closing_blade_cannot_move_from_fixed_physical_slot(qtbot):
    w = make_window(qtbot)
    w.closure_mode_combo.setCurrentIndex(w.closure_mode_combo.findData("demonstration"))
    w.start_optimization()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.result is not None, w.status_label.text()
    assert w.result.permutation[16] == 99
    assert "demonstration" in w.rotor_details_label.text().lower()


def test_measured_total_bare_rotor_vector_replaces_modeled_disk(qtbot):
    w = make_window(qtbot)
    w.mode_combo.setCurrentIndex(1)
    w.closure_mode_combo.setCurrentIndex(w.closure_mode_combo.findData("measured"))
    w.closure_radius_input.setValue(270)
    w.measured_disk_x_input.setValue(12)
    w.measured_disk_y_input.setValue(-8)
    w.closure_verified.setChecked(True)
    w.start_optimization()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.result is not None, w.status_label.text()
    assert w._solved_snapshot[1].disk_ux_gmm == pytest.approx(12)
    assert w._solved_snapshot[1].disk_uy_gmm == pytest.approx(-8)
    assert w._solved_snapshot[2].radii_mm[16] == pytest.approx(270)


def test_pending_drawings_block_solver_and_export(qtbot):
    w = make_window(qtbot)
    w.closure_mode_combo.setCurrentIndex(w.closure_mode_combo.findData("pending"))
    w.start_optimization()
    assert w._thread is None
    assert w.result is None
    assert not w.export_button.isEnabled()
    assert "pending" in w.status_label.text().lower()


def test_demonstration_closure_is_visibly_flagged_before_solving(qtbot):
    w = make_window(qtbot)
    w.closure_mode_combo.setCurrentIndex(w.closure_mode_combo.findData("demonstration"))
    assert "DEMONSTRATION" in w.warning.text()
    assert "cutout" in w.warning.text().lower()


def test_change_to_closing_slot_invalidates_prior_schedule(qtbot):
    w = make_window(qtbot)
    w.closure_mode_combo.setCurrentIndex(w.closure_mode_combo.findData("demonstration"))
    w.start_optimization()
    qtbot.waitUntil(lambda: w._thread is None, timeout=120000)
    assert w.export_button.isEnabled()
    w.closure_slot_input.setValue(18)
    assert w.result is None and not w.export_button.isEnabled()
