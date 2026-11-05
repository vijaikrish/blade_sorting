"""Acceptance tests for the ordered setup and output exploration workflow."""

import pytest
from PySide6.QtWidgets import QScrollArea
from data_io import sample_blades
from gui import MainWindow
from rotor_math import slot_geometry
from solver import SolverConfig, solve


def test_inputs_are_sequential_and_outputs_are_separate(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert [w.workspace_tabs.tabText(i) for i in range(w.workspace_tabs.count())] == [
        "Inputs", "Outputs", "Help"
    ]
    assert [w.input_steps.tabText(i) for i in range(w.input_steps.count())] == [
        "1 Analysis", "2 Blades", "3 Runout", "4 Installation", "5 Closure", "6 Method"
    ]
    assert w.workspace_tabs.currentIndex() == 0
    assert w.input_steps.currentWidget() is w.analysis_tab
    assert w.input_tab.isAncestorOf(w.blade_table)
    assert w.runout_tab.isAncestorOf(w.runout_inputs[0])
    assert w.installation_tab.isAncestorOf(w.start_angle_input)
    assert w.method_tab.isAncestorOf(w.method_combo)


def test_analysis_and_method_steps_explain_assumptions(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert "solid" in w.analysis_note.text().lower()
    assert "blade-only" in w.analysis_note.text().lower()
    assert "pairing" in w.method_note.text().lower()
    assert "annealing" in w.method_note.text().lower()
    assert "slot 1" in w.installation_note.text().lower()


def test_step_navigation_explains_and_follows_order(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert "model" in w.step_explanation.text().lower()
    w.next_step_button.click()
    assert w.input_steps.currentIndex() == 1
    assert "blade" in w.step_explanation.text().lower()
    w.previous_step_button.click()
    assert w.input_steps.currentIndex() == 0
    assert not w.previous_step_button.isEnabled()
    w.input_steps.setCurrentIndex(5)
    assert not w.next_step_button.isEnabled()  # outputs require a current calculation


def test_slot_map_runout_profile_uses_installation_reference(qtbot):
    from data_io import sample_blades
    from rotor_math import slot_geometry
    from solver import SolverConfig, solve
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_blades(sample_blades(8))
    w.mode_combo.setCurrentIndex(1)
    w.start_angle_input.setValue(25)
    rotor = w._input_model()
    geom = slot_geometry(8, rotor)
    result = solve([b.weight for b in w.get_blades()], geom.x_mm, geom.y_mm,
                   (rotor.disk_ux_gmm, rotor.disk_uy_gmm),
                   SolverConfig(iterations=100, restarts=1))
    w.show_result(result, w.get_blades(), rotor, geom)
    line = w.figures[1].axes[1].lines[0]
    assert float(line.get_xdata()[0]) == pytest.approx(25)
    assert float(line.get_xdata()[-1]) == pytest.approx(385)


def test_output_overview_is_scrollable_on_small_windows(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert isinstance(w.tabs.widget(0), QScrollArea)
    w.resize(900, 600)
    w.show()
    qtbot.wait(50)
    assert w.tabs.widget(0).widget() is not None


def test_output_overview_opens_individual_graph_and_runout_is_disabled_in_blade_only(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.input_steps.setCurrentIndex(2)
    assert not w.runout_inputs[0].isEnabled()
    w.mode_combo.setCurrentIndex(1)
    assert w.runout_inputs[0].isEnabled()
    assert w.tabs.tabText(0) == "Overview"
    w.set_blades(sample_blades(8))
    blades = w.get_blades()
    rotor = w._input_model()
    geom = slot_geometry(len(blades), rotor)
    result = solve([b.weight for b in blades], geom.x_mm, geom.y_mm,
                   (rotor.disk_ux_gmm, rotor.disk_uy_gmm),
                   SolverConfig(iterations=100, restarts=1))
    w.show_result(result, blades, rotor, geom)
    assert all(label.pixmap() and not label.pixmap().isNull() for label in w.overview_previews)
    w.overview_buttons[2].click()
    assert w.workspace_tabs.currentIndex() == 1
    assert w.tabs.tabText(w.tabs.currentIndex()) == "Runout"
    assert w.export_button.isEnabled()
