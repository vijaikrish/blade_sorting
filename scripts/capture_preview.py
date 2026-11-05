"""Exercise the real Qt worker and capture actual result tabs, not mockups."""
# ruff: noqa: E402 -- bootstrap repository root for direct script execution

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QTest
from gui import MainWindow
from data_io import closure_demo_blades

app = QApplication.instance() or QApplication([])
window = MainWindow()
window.mode_combo.setCurrentIndex(1)
window.radius_auto.setChecked(False)
window.radius_input.setValue(300)
window.start_angle_input.setValue(25)
window.set_blades(closure_demo_blades(42))
window.closure_enabled.setChecked(True)
window.closure_mode_combo.setCurrentIndex(window.closure_mode_combo.findData("demonstration"))
window.closure_slot_input.setValue(100)
for j, field in enumerate(window.runout_inputs):
    field.setValue(
        0.003 * math.cos(j * math.pi / 4)
        - 0.002 * math.sin(j * math.pi / 4)
        + 0.0006 * math.cos(j * math.pi / 2)
    )
window.show()
# Settle the initial font/canvas paints before timing an interactive solve.
app.processEvents()
QTest.qWait(200)
app.processEvents()
window.start_optimization()
loop = QEventLoop()
window._thread.finished.connect(loop.quit)
QTimer.singleShot(120000, loop.quit)
loop.exec()
if window._thread is not None:
    window.close()
    raise RuntimeError("worker did not complete within 120 seconds")
assert window.result is not None, window.status_label.text()
assert window.result.iterations_completed == 500000
out = ROOT / "artifacts"
for section, index, name in [
    (0, 0, "analysis"),
    (0, 1, "inventory"),
    (0, 2, "runout_input"),
    (0, 3, "installation"),
    (0, 4, "closure"),
    (0, 5, "method"),
    (1, 0, "overview"),
    (1, 1, "vectors"),
    (1, 2, "slot_map"),
    (1, 3, "runout"),
    (1, 4, "convergence"),
    (1, window.tabs.indexOf(window.schedule_tab), "mounting"),
    (2, 0, "help"),
]:
    window.workspace_tabs.setCurrentIndex(section)
    if section == 0:
        window.input_steps.setCurrentIndex(index)
    elif section == 1:
        window.tabs.setCurrentIndex(index)
    for canvas in window.canvases:
        canvas.draw()
    app.processEvents()
    assert window.grab().save(str(out / f"ui_{name}.png"))
window.resize(1280, 720)
window.workspace_tabs.setCurrentIndex(0)
window.input_steps.setCurrentIndex(0)
app.processEvents()
assert window.width() == 1280 and window.height() == 720, (window.width(), window.height())
assert window.grab().save(str(out / "ui_1280x720.png"))
window.resize(900, 600)
window.workspace_tabs.setCurrentIndex(0)
window.input_steps.setCurrentIndex(0)
app.processEvents()
assert window.grab().save(str(out / "ui_900x600_inputs.png"))
window.workspace_tabs.setCurrentIndex(1)
window.tabs.setCurrentIndex(0)
app.processEvents()
assert window.grab().save(str(out / "ui_900x600_outputs.png"))
window.resize(1280, 720)
window.export_csv_path(out / "demo_closure_mounting_schedule.csv", overwrite=True)
result = window.result
summary = {
    name: getattr(result, name)
    for name in (
        "initial_j_gmm",
        "greedy_j_gmm",
        "pairing_j_gmm",
        "residual_gmm",
        "blade_ux_gmm",
        "blade_uy_gmm",
        "net_ux_gmm",
        "net_uy_gmm",
        "lower_bound_gmm",
        "elapsed_ms",
        "iterations_completed",
        "seed",
    )
}
summary.update(
    {
        "window_size": [1400, 900],
        "small_window_size": [window.width(), window.height()],
        "mode": "combined + fixed closure demonstration",
        "data": "synthetic; closure_demo_blades(42): 99 × 30–40 g plus one 50 g closure; signed 1x plus 2x runout",
        "closure_slot": 100,
        "start_angle_deg": 25,
        "status": window.status_label.text(),
        "exported_rows": window.results_table.rowCount(),
    }
)
(out / "ui_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
window.close()
app.processEvents()
