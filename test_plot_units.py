"""Plot units and engineering-help regressions."""

import math
from data_io import sample_blades
from gui import MainWindow
from rotor_math import slot_geometry
from solver import solve, SolverConfig


def test_polar_axis_has_real_unbalance_units_not_unlabelled_normalization(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_blades(sample_blades(8))
    blades = w.get_blades()
    rotor = w._input_model()
    geom = slot_geometry(len(blades), rotor)
    result = solve(
        [b.weight for b in blades], geom.x_mm, geom.y_mm, config=SolverConfig(iterations=10, restarts=1)
    )
    w.show_result(result, blades, rotor, geom)
    largest = max(
        math.hypot(result.initial_blade_ux_gmm, result.initial_blade_uy_gmm),
        math.hypot(result.blade_ux_gmm, result.blade_uy_gmm),
        result.residual_gmm,
    )
    ax = w.figures[0].axes[0]
    assert ax.get_ylim()[1] >= largest
    assert "g·mm" in (ax.get_title() + ax.get_ylabel())
    assert len(ax.get_yticks()) <= 4  # sparse rings stay readable beside the legend


def test_full_engineering_help_is_available_in_the_app(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert hasattr(w, "help_browser")
    text = w.help_browser.toPlainText().lower()
    assert all(word in text for word in ["solid", "projection", "cooling", "nist", "g·mm"])
    assert "phase_vector.png" in w.help_browser.toHtml()
    assert w.help_browser.document().baseUrl().isLocalFile()
