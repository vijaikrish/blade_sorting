"""Blade sorting desktop application."""

from __future__ import annotations
import argparse
import sys
from PySide6.QtWidgets import QApplication
from data_io import sample_blades
from gui import MainWindow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="load a deterministic combined-mode sample (does not optimize)"
    )
    parser.add_argument("--screenshot", metavar="PATH", help="save a QWidget.grab() screenshot and exit")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    if args.demo:
        window.mode_combo.setCurrentIndex(1)
        window.set_blades(sample_blades(32))
        for field, value in zip(window.runout_inputs, (0.12, 0.08, -0.02, -0.09, -0.12, -0.08, 0.02, 0.09)):
            field.setValue(value)
        window.status_label.setText("Synthetic demonstration data · not measured rotor/blade data.")
    window.show()
    if args.screenshot:
        app.processEvents()
        if not window.grab().save(args.screenshot):
            raise OSError(f"Could not save screenshot to {args.screenshot}")
        window.close()
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
