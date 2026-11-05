def test_main_window_exposes_contract(qtbot):
    from gui import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.mode_combo.currentData() == "blade_only"
    assert len(window.runout_inputs) == 8
    assert window.result is None


def test_blade_only_disables_inactive_inputs(qtbot):
    from gui import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert not window.thickness_input.isEnabled()
    assert not window.density_input.isEnabled()
    assert all(not field.isEnabled() for field in window.runout_inputs)


def test_mode_switch_preserves_radius_and_shows_estimate_warning(qtbot):
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.radius_auto.setChecked(False)
    w.radius_input.setValue(181.5)
    w.mode_combo.setCurrentIndex(1)
    assert w.radius_input.value() == 181.5 and w.radius_input.isEnabled()
    assert all(x.isEnabled() for x in w.runout_inputs) and "ESTIMATED RUNOUT MODEL" in w.warning.text()


def test_input_edit_invalidates_exportable_snapshot(qtbot):
    from gui import MainWindow
    from rotor_math import slot_geometry
    from solver import SolverConfig, solve
    from data_io import sample_blades

    w = MainWindow()
    qtbot.addWidget(w)

    w.set_blades(sample_blades(4))
    blades = w.get_blades()
    rotor = w._input_model()
    geom = slot_geometry(len(blades), rotor)
    result = solve(
        [b.weight for b in blades], geom.x_mm, geom.y_mm, config=SolverConfig(iterations=2, restarts=1)
    )
    w.show_result(result, blades, rotor, geom)
    assert w.export_button.isEnabled()
    w.diameter_input.setValue(510)
    assert w.result is None and not w.export_button.isEnabled()


def test_runout_plot_uses_raw_samples_and_redraw_replaces_axes(qtbot):
    from gui import MainWindow
    from rotor_math import slot_geometry

    from data_io import sample_blades

    w = MainWindow()
    qtbot.addWidget(w)
    w.set_blades(sample_blades(4))
    w.mode_combo.setCurrentIndex(1)
    for field, value in zip(w.runout_inputs, [1, 0, -1, 0, 1, 0, -1, 0]):
        field.setValue(value)
    rotor = w._input_model()
    assert tuple(rotor.runout_samples_mm) == (1, 0, -1, 0, 1, 0, -1, 0)
    from solver import solve, SolverConfig

    blades = w.get_blades()
    geom = slot_geometry(len(blades), rotor)
    result = solve(
        [b.weight for b in blades],
        geom.x_mm,
        geom.y_mm,
        (rotor.disk_ux_gmm, rotor.disk_uy_gmm),
        SolverConfig(iterations=2, restarts=1),
    )
    w.show_result(result, blades, rotor, geom)
    w.show_result(result, blades, rotor, geom)
    assert len(w.figures[0].axes) == 1
    assert tuple(w.figures[2].axes[0].lines[1].get_ydata()) == tuple(rotor.runout_samples_mm)


def test_default_blade_masses_are_blank_and_layout_sidebar_is_bounded(qtbot):
    from PySide6.QtWidgets import QScrollArea
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert w.blade_table.item(0, 1).text() == ""
    assert any(isinstance(x, QScrollArea) for x in w.findChildren(QScrollArea))


def test_background_solver_completes_real_full_workload(qtbot):
    from gui import MainWindow
    from data_io import sample_blades

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_blades(sample_blades(8))
    window.show()
    window.start_optimization()
    assert not window.optimize_button.isEnabled()
    qtbot.waitUntil(lambda: window._thread is None, timeout=120_000)
    assert window.result is not None
    assert window.result.iterations_completed == 500_000
    assert "warm-up" in window.status_label.text()
