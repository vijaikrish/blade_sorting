from __future__ import annotations

import math
import threading
from dataclasses import replace
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, QPropertyAnimation, QSequentialAnimationGroup, QEasingCurve, QUrl, Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QTableWidget,
    QTextBrowser,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QTabWidget,
    QGroupBox,
    QHeaderView,
    QProgressBar,
    QCheckBox,
    QScrollArea,
    QGraphicsOpacityEffect,
    QMessageBox,
    QFileDialog,
)
from data_io import Blade, validate_blades, read_blades_csv, export_schedule_csv, sample_blades
from rotor_math import RotorInputs, build_rotor, slot_geometry
from solver import SolverConfig, solve, solve_fixed_closure, warm_up, SolverCancelled

STYLE = "QMainWindow{background:#f3f6fa;color:#172b43} QWidget{font-size:12px} QGroupBox{background:white;border:1px solid #dbe3ec;border-radius:10px;margin-top:10px;padding:12px;font-weight:600} QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px;color:#254a68} QPushButton{background:#145a73;color:white;border:0;border-radius:6px;padding:9px 14px;font-weight:600} QPushButton:hover{background:#1c718c} QPushButton:pressed{background:#0d465b;padding-top:10px;padding-bottom:8px} QPushButton:disabled{background:#aab7c2} QTableWidget{background:white;border:1px solid #dbe3ec;gridline-color:#e5ebf0} QTabWidget::pane{background:white;border:1px solid #dbe3ec;border-radius:8px} QTabBar::tab{padding:9px 14px} QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox{padding:5px;border:1px solid #cbd6e0;border-radius:5px;background:white}"


class _Worker(QObject):
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, masses, geom, rotor, snapshot, seed, method, cancel, closure=None):
        super().__init__()
        self.masses = masses
        self.geom = geom
        self.rotor = rotor
        self.snapshot = snapshot
        self.seed = seed
        self.method = method
        self.cancel = cancel
        self.closure = closure
        self.warmup_ms = 0.0

    def run(self):
        try:
            self.warmup_ms = warm_up()
            disk = (self.rotor.disk_ux_gmm, self.rotor.disk_uy_gmm)
            config = SolverConfig(seed=self.seed, iterations=100000, restarts=5)
            if self.closure is None:
                result = solve(self.masses, self.geom.x_mm, self.geom.y_mm, disk,
                               config, method=self.method, cancel=self.cancel)
            else:
                result = solve_fixed_closure(self.masses, self.geom.x_mm, self.geom.y_mm,
                    disk=disk, config=config, method=self.method, cancel=self.cancel,
                    **self.closure)
            self.finished.emit(result, self.snapshot)
        except SolverCancelled:
            self.failed.emit("Optimization cancelled.")
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blade sorting")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLE)
        self._install_readable_font()
        self.result = None
        self._revision = 0
        self._worker = None
        self._thread = None
        self._cancel = None
        self._warmup_ms = 0.0
        self._running_revision = 0
        self._solved_snapshot = None
        self._button_animations = {}
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("BLADE SORTING")
        title.setStyleSheet("font-size:22px;font-weight:800;color:#123c56;letter-spacing:1px")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("ROTOR ENGINEERING · STATIC PRE-BALANCE"))
        layout.addLayout(header)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(
            "background:#fff1d7;color:#824d00;padding:10px;border-radius:6px;font-weight:600"
        )
        layout.addWidget(self.warning)
        self.workspace_tabs = QTabWidget()
        layout.addWidget(self.workspace_tabs, 1)
        self.input_page = QWidget()
        input_page_layout = QVBoxLayout(self.input_page)
        self.step_explanation = QLabel()
        self.step_explanation.setWordWrap(True)
        self.step_explanation.setStyleSheet("color:#254a68;padding:8px;font-size:13px")
        input_page_layout.addWidget(self.step_explanation)
        self.input_steps = QTabWidget()
        input_page_layout.addWidget(self.input_steps, 1)
        step_navigation = QHBoxLayout()
        self.previous_step_button = QPushButton("← Previous step")
        self.next_step_button = QPushButton("Next step →")
        step_navigation.addWidget(self.previous_step_button)
        step_navigation.addStretch()
        step_navigation.addWidget(self.next_step_button)
        input_page_layout.addLayout(step_navigation)
        self.workspace_tabs.addTab(self.input_page, "Inputs")
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        self.input_steps.addTab(self.analysis_tab, "1 Analysis")
        left = QWidget()
        ll = QVBoxLayout(left)
        analysis_layout.addWidget(left)
        cfg = QGroupBox("Configuration · mm / kg·m⁻³")
        cfg.setMaximumWidth(760)
        cfg.setMaximumHeight(360)
        form = QFormLayout(cfg)
        ll.addWidget(cfg)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Blade-only model", "blade_only")
        self.mode_combo.addItem("Combined disk + runout estimate", "combined")
        form.addRow("Model", self.mode_combo)
        self.diameter_input = self._double(1, 100000, 500)
        self.thickness_input = self._double(0.01, 10000, 50)
        self.density_input = self._double(0.01, 1e6, 7850)
        self.radius_input = self._double(0.01, 100000, 250)
        self.radius_auto = QCheckBox("Auto: D/2 + offset (0 mm)")
        self.radius_auto.setChecked(True)
        form.addRow("Rotor diameter · mm", self.diameter_input)
        form.addRow("Disk thickness · mm", self.thickness_input)
        form.addRow("Density · kg/m³", self.density_input)
        form.addRow("Blade CG radius · mm", self.radius_input)
        form.addRow("Radius mode", self.radius_auto)
        self.analysis_note = QLabel(
            "Blade-only sets the bare-rotor vector to zero. Combined mode estimates the mass "
            "of a uniform solid disk from diameter, thickness and density, then uses the 1× "
            "runout fit as an assumed eccentricity—not a measured mass-unbalance vector. "
            "Blade CG radius is from the rotation axis; auto uses D/2."
        )
        self.analysis_note.setWordWrap(True)
        ll.addWidget(self.analysis_note)
        ll.addStretch()
        self.runout_inputs = [self._double(-1e6, 1e6, 0, 4) for _ in range(8)]
        self.runout_angle_labels = []
        runout_grid = QWidget()
        run = QGridLayout(runout_grid)
        for i, field in enumerate(self.runout_inputs):
            row, col = divmod(i, 2)
            angle_label = QLabel(f"{i * 45}°")
            self.runout_angle_labels.append(angle_label)
            run.addWidget(angle_label, row, col * 2)
            run.addWidget(field, row, col * 2 + 1)
        self.runout_reference_label = QLabel("Signed radial samples · mm, relative to slot 1 · outward positive")
        self.runout_reference_label.setWordWrap(True)
        controls = QGroupBox("Search method")
        cf = QFormLayout(controls)
        self.method_combo = QComboBox()
        self.method_combo.addItem("Hybrid SA", "hybrid")
        self.method_combo.addItem("Opposite pairing", "pairing")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2147483647)
        self.seed_input.setValue(42)
        cf.addRow("Method", self.method_combo)
        self.blade_count = QSpinBox()
        self.blade_count.setRange(2, 120)
        self.blade_count.setValue(32)
        self.optimize_button = QPushButton("Optimize · 5 × 100k")
        self.import_button = QPushButton("Import CSV")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.status_label = QLabel("Ready · complete the steps, then calculate.")
        self.status_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        right = QWidget()
        rl = QVBoxLayout(right)
        self.workspace_tabs.addTab(right, "Outputs")
        self.cards = QLabel("RESIDUAL — g·mm  |  NET VECTOR —  |  RUN —")
        self.cards.setStyleSheet(
            "background:#123c56;color:white;padding:14px;border-radius:8px;font-size:15px;font-weight:700"
        )
        self.cards.setWordWrap(True)
        rl.addWidget(self.cards)
        self.comparison_label = QLabel("Initial / vector greedy / pairing / residual floor — g·mm")
        self.comparison_label.setWordWrap(True)
        self.rotor_details_label = QLabel(
            "Model details appear after a complete calculation. Slot 1: 0° +x; CCW."
        )
        self.rotor_details_label.setWordWrap(True)
        rl.addWidget(self.comparison_label)
        rl.addWidget(self.rotor_details_label)
        self.tabs = QTabWidget()
        rl.addWidget(self.tabs, 1)
        self.figures = []
        self.canvases = []
        self._make_tabs()
        self._make_overview()
        self.blade_table = QTableWidget(0, 2)
        self.blade_table.setHorizontalHeaderLabels(["Blade ID", "Mass · g"])
        self.blade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.input_tab = QWidget()
        input_layout = QVBoxLayout(self.input_tab)
        input_layout.addWidget(QLabel("Blade inventory · measured blade masses in g; choose manual, CSV or seeded demonstration"))
        inventory_form = QFormLayout()
        inventory_form.addRow("Number of blades", self.blade_count)
        inventory_form.addRow("Synthetic sample / search seed", self.seed_input)
        input_layout.addLayout(inventory_form)
        inventory_actions = QHBoxLayout()
        self.blank_inventory_button = QPushButton("Generate blank inventory")
        self.random_samples_button = QPushButton("Load random samples")
        inventory_actions.addWidget(self.blank_inventory_button)
        inventory_actions.addWidget(self.random_samples_button)
        inventory_actions.addWidget(self.import_button)
        input_layout.addLayout(inventory_actions)
        input_layout.addWidget(self.blade_table)
        self.input_steps.addTab(self.input_tab, "2 Blades")
        self.runout_tab = QWidget()
        runout_layout = QVBoxLayout(self.runout_tab)
        runout_layout.addWidget(QLabel("Eight signed radial readings every 45° from slot 1; positive means outward. Not peak-to-peak TIR."))
        runout_box = QGroupBox("Runout readings · mm")
        runout_box.setMaximumWidth(840)
        runout_form = QVBoxLayout(runout_box)
        runout_form.addWidget(self.runout_reference_label)
        runout_form.addWidget(runout_grid)
        runout_layout.addWidget(runout_box)
        runout_layout.addStretch()
        self.input_steps.addTab(self.runout_tab, "3 Runout")
        self.installation_tab = QWidget()
        self.installation_tab_index = self.input_steps.addTab(self.installation_tab, "4 Installation")
        install_form = QFormLayout(self.installation_tab)
        self.start_angle_input = self._double(0, 359.99, 0, 2)
        install_form.addRow("Slot 1 global angle · °", self.start_angle_input)
        self.reduce_motion = QCheckBox("Reduce motion (disable animated button feedback)")
        install_form.addRow(self.reduce_motion)
        self.installation_note = QLabel(
            "Slot 1 defines the first runout measurement at relative 0°. Enter its angle "
            "from global +x here (default 0°); the remaining runout samples advance "
            "45° from slot 1 and blade slots advance 360°/N. Use the same viewing face "
            "and positive counterclockwise direction throughout."
        )
        self.installation_note.setWordWrap(True)
        install_form.addRow(self.installation_note)
        self.closure_tab = QWidget()
        closure_form = QFormLayout(self.closure_tab)
        self.closure_enabled = QCheckBox("Use fixed closing blade / locking piece")
        closure_form.addRow(self.closure_enabled)
        self.closure_blade_combo = QComboBox()
        closure_form.addRow("Closing blade ID", self.closure_blade_combo)
        self.closure_slot_input = QSpinBox()
        self.closure_slot_input.setRange(1, self.blade_count.value())
        self.closure_slot_input.setValue(self.blade_count.value())
        closure_form.addRow("Fixed physical slot (1-based)", self.closure_slot_input)
        self.closure_mode_combo = QComboBox()
        self.closure_mode_combo.addItem("Drawings / measurements pending — block calculation", "pending")
        self.closure_mode_combo.addItem("Demonstration — nominal radius, cutout omitted", "demonstration")
        self.closure_mode_combo.addItem("Measured — effective closure CG + total bare rotor vector", "measured")
        closure_form.addRow("Interpretation", self.closure_mode_combo)
        self.closure_radius_input = self._double(0.01, 100000, 250, 3)
        closure_form.addRow("Closing blade effective CG radius · mm", self.closure_radius_input)
        self.measured_disk_x_input = self._double(-1e12, 1e12, 0, 4)
        self.measured_disk_y_input = self._double(-1e12, 1e12, 0, 4)
        closure_form.addRow("Measured TOTAL bare rotor Ux · g·mm", self.measured_disk_x_input)
        closure_form.addRow("Measured TOTAL bare rotor Uy · g·mm", self.measured_disk_y_input)
        self.closure_verified = QCheckBox("Verified: bare-rotor vector excludes closure blade; same angular reference")
        closure_form.addRow(self.closure_verified)
        closure_note = QLabel(
            "Measured mode replaces the runout-derived disk vector; do not add the cutout twice. "
            "Demonstration mode omits the cutout and is NOT a physical balancing prediction. "
            "Pending mode prevents a calculation until the drawing or measurement is available."
        )
        closure_note.setWordWrap(True)
        closure_form.addRow(closure_note)
        self.input_steps.addTab(self.closure_tab, "5 Closure")
        self.method_tab = QWidget()
        method_layout = QVBoxLayout(self.method_tab)
        method_layout.addWidget(QLabel("Choose the calculation method after checking all inputs. The seed above also reproduces the hybrid search."))
        method_layout.addWidget(controls)
        self.method_note = QLabel(
            "Hybrid simulated annealing: vector-greedy start, five seeded runs × 100,000 "
            "swap proposals; keeps the best result but cannot guarantee the global optimum. "
            "Opposite pairing: fast deterministic baseline, no annealing. Both report g·mm."
        )
        self.method_note.setWordWrap(True)
        method_layout.addWidget(self.method_note)
        method_actions = QHBoxLayout()
        method_actions.addWidget(self.optimize_button)
        method_actions.addWidget(self.cancel_button)
        method_layout.addLayout(method_actions)
        method_layout.addWidget(self.status_label)
        method_layout.addWidget(self.progress)
        method_layout.addStretch()
        self.input_steps.addTab(self.method_tab, "6 Method")
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Slot", "Angle · °", "Blade", "Mass · g"])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_tab = QWidget()
        schedule_layout = QVBoxLayout(self.schedule_tab)
        schedule_layout.addWidget(self.results_table)
        self.tabs.addTab(self.schedule_tab, "Schedule")
        self.export_button = QPushButton("Export mounting schedule…")
        self.export_button.setEnabled(False)
        schedule_layout.addWidget(self.export_button)
        for w in [
            self.mode_combo,
            self.method_combo,
            self.seed_input,
            self.diameter_input,
            self.thickness_input,
            self.density_input,
            self.radius_input,
            self.start_angle_input,
            self.closure_enabled,
            self.closure_blade_combo,
            self.closure_slot_input,
            self.closure_mode_combo,
            self.closure_radius_input,
            self.measured_disk_x_input,
            self.measured_disk_y_input,
            self.closure_verified,
            self.radius_auto,
            *self.runout_inputs,
        ]:
            sig = (
                getattr(w, "currentIndexChanged", None)
                or getattr(w, "valueChanged", None)
                or getattr(w, "toggled", None)
            )
            if sig:
                sig.connect(self._edited)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.closure_enabled.toggled.connect(self._mode_changed)
        self.closure_mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.radius_auto.toggled.connect(self._radius_changed)
        self.diameter_input.valueChanged.connect(self._radius_changed)
        self.blade_count.valueChanged.connect(self._resize_blades)
        self.blade_table.itemChanged.connect(self.invalidate_results)
        self.blade_table.itemChanged.connect(self._refresh_closure_blades)
        self.optimize_button.clicked.connect(self.start_optimization)
        self.blank_inventory_button.clicked.connect(self.generate_blank_inventory)
        self.random_samples_button.clicked.connect(self.load_random_samples)
        self.import_button.clicked.connect(self._import_dialog)
        self.export_button.clicked.connect(self._export_dialog)
        self.cancel_button.clicked.connect(self._cancel_work)
        for button in (self.optimize_button, self.cancel_button, self.import_button,
                       self.blank_inventory_button, self.random_samples_button, self.export_button):
            button.clicked.connect(lambda _checked=False, b=button: self._animate_button(b))
        self.input_steps.currentChanged.connect(self._update_step_header)
        self.previous_step_button.clicked.connect(
            lambda: self.input_steps.setCurrentIndex(max(0, self.input_steps.currentIndex() - 1))
        )
        self.next_step_button.clicked.connect(self._advance_step)
        self._mode_changed()
        self._radius_changed()
        self._resize_blades()
        self._update_step_header()

    @staticmethod
    def _double(lo, hi, val, dec=2):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setValue(val)
        w.setKeyboardTracking(False)
        return w

    @staticmethod
    def _install_readable_font():
        from PySide6.QtGui import QFont, QFontDatabase
        from pathlib import Path

        candidates = [Path("C:/Windows/Fonts/segoeui.ttf")]
        try:
            from matplotlib import get_data_path

            candidates.append(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
        except Exception:
            pass
        for path in candidates:
            if path.is_file() and QFontDatabase.addApplicationFont(str(path)) >= 0:
                break
        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Segoe UI", 9))

    def _make_tabs(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        for name in ["Vectors", "Slot map", "Runout", "Convergence"]:
            f = Figure(figsize=(6, 4), tight_layout=True)
            c = FigureCanvasQTAgg(f)
            pane = QWidget()
            v = QVBoxLayout(pane)
            v.addWidget(NavigationToolbar2QT(c, pane))
            v.addWidget(c, 1)
            self.tabs.addTab(pane, name)
            self.figures.append(f)
            self.canvases.append(c)
        from pathlib import Path
        pane = QWidget()
        v = QVBoxLayout(pane)
        self.help_browser = QTextBrowser()
        self.help_browser.setOpenExternalLinks(True)
        doc = Path(__file__).resolve().parent / "docs" / "USER_GUIDE.html"
        if doc.is_file():
            self.help_browser.setSource(QUrl.fromLocalFile(str(doc)))
        else:
            self.help_browser.setPlainText("Operator guide missing. See docs/ENGINEERING_MODEL.md before using results.")
        v.addWidget(self.help_browser)
        self.workspace_tabs.addTab(pane, "Help")

    def _make_overview(self):
        """Four thumbnails provide a quick comparison and open full interactive plots."""
        overview = QWidget()
        grid = QGridLayout(overview)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(overview)
        self.overview_previews = []
        self.overview_buttons = []
        for index, name in enumerate(("Vectors", "Slot map", "Runout", "Convergence")):
            cell = QWidget()
            column = QVBoxLayout(cell)
            label = QLabel("Run calculation to view this graph")
            label.setFixedSize(350, 155)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background:#f5f8fb;border:1px solid #dbe3ec;border-radius:6px")
            button = QPushButton(f"Open {name} view")
            button.clicked.connect(lambda _checked=False, i=index: self.open_graph_view(i))
            column.addWidget(label)
            column.addWidget(button)
            grid.addWidget(cell, index // 2, index % 2)
            self.overview_previews.append(label)
            self.overview_buttons.append(button)
        self.tabs.insertTab(0, scroller, "Overview")

    def open_graph_view(self, index):
        self.workspace_tabs.setCurrentIndex(1)
        self.tabs.setCurrentIndex(index + 1)

    def _update_step_header(self, *_):
        descriptions = (
            "Select the analysis model and enter rotor geometry and blade CG radius (mm).",
            "Enter blade count and measured masses (g), or import CSV; seeded samples are synthetic.",
            "Enter eight signed runout measurements (mm), starting at installation slot 1.",
            "Locate installation slot 1 relative to global +x; runout 0° is at that slot.",
            "If applicable, fix the closing blade to its physical slot and choose a defensible interpretation.",
            "Select the calculation methodology and run; inspect results in Outputs.",
        )
        index = self.input_steps.currentIndex()
        self.step_explanation.setText(descriptions[index])
        self.previous_step_button.setEnabled(index > 0)
        self.next_step_button.setEnabled(index < len(descriptions) - 1 or self.result is not None)
        self.next_step_button.setText("View outputs →" if index == len(descriptions) - 1 else "Next step →")

    def _advance_step(self):
        index = self.input_steps.currentIndex()
        if index < self.input_steps.count() - 1:
            self.input_steps.setCurrentIndex(index + 1)
        elif self.result is not None:
            self.workspace_tabs.setCurrentIndex(1)

    def _animate_button(self, button):
        """Brief opacity pulse gives click feedback without delaying the action."""
        effect = button.graphicsEffect()
        if effect is None:
            effect = QGraphicsOpacityEffect(button)
            button.setGraphicsEffect(effect)
        old = self._button_animations.get(button)
        if old is not None:
            old.stop()
        effect.setOpacity(1.0)
        if self.reduce_motion.isChecked():
            return
        group = QSequentialAnimationGroup(button)
        for start, end, duration in ((1.0, .68, 70), (.68, 1.0, 130)):
            pulse = QPropertyAnimation(effect, b"opacity", group)
            pulse.setDuration(duration)
            pulse.setStartValue(start)
            pulse.setEndValue(end)
            pulse.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(pulse)
        self._button_animations[button] = group
        group.start()

    def _mode_changed(self, *_):
        active = self.mode_combo.currentData() == "combined"
        for w in [self.thickness_input, self.density_input, *self.runout_inputs]:
            w.setEnabled(active)
        warnings = []
        if active:
            warnings.append("ESTIMATED RUNOUT MODEL — geometry-derived 1× fit is not a measured bare-rotor unbalance.")
        closure_mode = self.closure_mode_combo.currentData()
        if self.closure_enabled.isChecked():
            if closure_mode == "demonstration":
                warnings.append("DEMONSTRATION ONLY — closing-blade CG is assumed nominal; rotor cutout mass moment is omitted. Not a physical balance prediction.")
            elif closure_mode == "pending":
                warnings.append("CLOSURE DATA PENDING — calculation blocked until measurements are entered or demonstration is explicitly selected.")
            else:
                warnings.append("MEASURED CLOSURE — use the TOTAL measured bare-rotor vector; it replaces rather than adds to the runout estimate.")
        self.warning.setText("  ".join(warnings))
        self.warning.setVisible(bool(warnings))
        self.invalidate_results()

    def _radius_changed(self, *_):
        auto = self.radius_auto.isChecked()
        self.radius_input.setEnabled(not auto)
        if auto:
            self.radius_input.setValue(self.diameter_input.value() / 2)

    def _edited(self, *_):
        self.invalidate_results()

    def invalidate_results(self, *_):
        self._revision += 1
        self.result = None
        self._solved_snapshot = None
        self.export_button.setEnabled(False)
        self.results_table.setRowCount(0)
        self.cards.setText("RESIDUAL — g·mm  |  NET VECTOR —  |  RUN —")
        self.comparison_label.setText("Initial / vector greedy / pairing / residual floor — g·mm")
        self.rotor_details_label.setText(
            "Model details appear after a complete calculation. Slot 1: 0° +x; CCW."
        )
        for f, c in zip(getattr(self, "figures", []), getattr(self, "canvases", [])):
            f.clear()
            c.draw_idle()
        for preview in getattr(self, "overview_previews", []):
            preview.clear()
            preview.setText("Run calculation to view this graph")
        if hasattr(self, "input_steps") and self.input_steps.count() == 6:
            self._update_step_header()

    def _refresh_closure_blades(self, *_):
        n = self.blade_table.rowCount()
        old_max = self.closure_slot_input.maximum()
        was_last = self.closure_slot_input.value() == old_max
        self.closure_slot_input.setMaximum(max(1, n))
        if was_last:
            self.closure_slot_input.setValue(max(1, n))
        selected = self.closure_blade_combo.currentData()
        self.closure_blade_combo.blockSignals(True)
        self.closure_blade_combo.clear()
        for i in range(n):
            item = self.blade_table.item(i, 0)
            name = item.text() if item is not None else f"B{i + 1:03d}"
            self.closure_blade_combo.addItem(name, name)
        match = self.closure_blade_combo.findData(selected)
        self.closure_blade_combo.setCurrentIndex(match if match >= 0 else max(0, n - 1))
        self.closure_blade_combo.blockSignals(False)

    def _closure_config(self, blades, rotor):
        if not self.closure_enabled.isChecked():
            return None
        if len(blades) < 3:
            raise ValueError("Closing blade mode requires at least three blades")
        chosen = self.closure_blade_combo.currentData()
        indices = [i for i, blade in enumerate(blades) if blade.blade_name == chosen]
        if len(indices) != 1:
            raise ValueError("Select an existing unique closing blade ID")
        mode = self.closure_mode_combo.currentData()
        if mode == "pending":
            raise ValueError("Drawings/measurements pending: enter closure CG and bare-rotor vector, or explicitly choose demonstration")
        config = {"fixed_blade_index": indices[0], "fixed_slot": self.closure_slot_input.value() - 1, "mode": mode}
        if mode == "measured":
            if rotor.mode != "combined":
                raise ValueError("Measured bare-rotor closure requires Rotor + blade mode")
            if not self.closure_verified.isChecked():
                raise ValueError("Confirm the measured closure CG and total bare-rotor vector reference before calculating")
            config.update(closure_radius_mm=self.closure_radius_input.value(),
                          measured_disk=(self.measured_disk_x_input.value(), self.measured_disk_y_input.value()))
        return config

    def _resize_blades(self, *_):
        n = self.blade_count.value()
        old = self.blade_table.rowCount()
        self.blade_table.setRowCount(n)
        for r in range(n):
            if r >= old:
                self.blade_table.setItem(r, 0, QTableWidgetItem(f"B{r + 1:03d}"))
                self.blade_table.setItem(r, 1, QTableWidgetItem(""))
        self._refresh_closure_blades()
        self.invalidate_results()

    def set_blades(self, blades):
        valid = validate_blades(blades)
        self.blade_table.blockSignals(True)
        self.blade_table.setRowCount(len(valid))
        self.blade_count.setValue(len(valid))
        for r, b in enumerate(valid):
            self.blade_table.setItem(r, 0, QTableWidgetItem(b.blade_name))
            self.blade_table.setItem(r, 1, QTableWidgetItem(str(b.weight)))
        self.blade_table.blockSignals(False)
        self._refresh_closure_blades()
        self.invalidate_results()

    def generate_blank_inventory(self):
        n = self.blade_count.value()
        self.blade_table.blockSignals(True)
        self.blade_table.setRowCount(n)
        for row in range(n):
            self.blade_table.setItem(row, 0, QTableWidgetItem(f"B{row + 1:03d}"))
            self.blade_table.setItem(row, 1, QTableWidgetItem(""))
        self.blade_table.blockSignals(False)
        self._refresh_closure_blades()
        self.invalidate_results()
        self.workspace_tabs.setCurrentIndex(0)
        self.input_steps.setCurrentWidget(self.input_tab)
        QMessageBox.information(self, "Blank inventory", "Enter blade masses manually, or use the seed to generate synthetic masses.")

    def load_random_samples(self):
        self.set_blades(sample_blades(self.blade_count.value(), seed=self.seed_input.value()))
        self.workspace_tabs.setCurrentIndex(0)
        self.input_steps.setCurrentWidget(self.input_tab)
        QMessageBox.information(self, "Synthetic sample inventory", f"Loaded {self.blade_count.value()} synthetic blade masses using seed {self.seed_input.value()}.")

    def get_blades(self):
        rows = []
        for r in range(self.blade_table.rowCount()):
            a = self.blade_table.item(r, 0)
            b = self.blade_table.item(r, 1)
            try:
                rows.append(Blade(a.text() if a else "", float(b.text()) if b else float("nan")))
            except ValueError as exc:
                raise ValueError(f"Row {r + 1}: mass must be numeric") from exc
        return validate_blades(rows)

    def _load_sample(self):
        self.load_random_samples()

    def import_csv_path(self, path):
        self.set_blades(read_blades_csv(path))

    def _import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import blades", "", "CSV files (*.csv)")
        if path:
            try:
                self.import_csv_path(path)
            except (ValueError, OSError) as exc:
                QMessageBox.critical(self, "Import failed", str(exc))

    def _input_model(self):
        readings = (
            tuple(w.value() for w in self.runout_inputs)
            if self.mode_combo.currentData() == "combined"
            else (0.0,) * 8
        )
        return build_rotor(
            RotorInputs(
                mode=self.mode_combo.currentData(),
                diameter_mm=self.diameter_input.value(),
                thickness_mm=self.thickness_input.value(),
                density_kg_m3=self.density_input.value(),
                cg_radius_mm=None if self.radius_auto.isChecked() else self.radius_input.value(),
                runout_mm=readings,
                start_angle_deg=self.start_angle_input.value(),
            )
        )

    def start_optimization(self):
        if self._thread is not None:
            return
        try:
            blades = self.get_blades()
            rotor = self._input_model()
            closure = self._closure_config(blades, rotor)
        except (ValueError, TypeError) as exc:
            self.status_label.setText(f"Input error: {exc}")
            return
        if closure and closure["mode"] == "measured":
            ux, uy = closure["measured_disk"]
            rotor = replace(rotor, disk_ux_gmm=ux, disk_uy_gmm=uy,
                warnings=("Measured TOTAL bare-rotor vector replaces the geometry-derived disk estimate.",))
        geom = slot_geometry(len(blades), rotor)
        if closure and closure["mode"] == "measured":
            k = closure["fixed_slot"]
            angle = float(geom.angles_rad[k])
            radius = closure["closure_radius_mm"]
            radii, gx, gy = geom.radii_mm.copy(), geom.x_mm.copy(), geom.y_mm.copy()
            radii[k], gx[k], gy[k] = radius, radius * math.cos(angle), radius * math.sin(angle)
            geom = replace(geom, radii_mm=radii, x_mm=gx, y_mm=gy)
        masses = np.array([b.weight for b in blades], dtype=float)
        self.invalidate_results()
        self._running_revision = self._revision
        snapshot = (blades, rotor, geom)
        self._cancel = threading.Event()
        self._worker = _Worker(
            masses,
            geom,
            rotor,
            snapshot,
            self.seed_input.value(),
            self.method_combo.currentData(),
            self._cancel,
            closure,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._complete)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._restore)
        self.optimize_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.show()
        self.status_label.setText("Compiling / optimizing in background…")
        self._thread.start()

    @Slot(object, object)
    def _complete(self, result, data):
        if self._cancel is not None and self._cancel.is_set():
            self.status_label.setText("Optimization cancelled; result discarded.")
            return
        if self._running_revision != self._revision:
            self.status_label.setText("Inputs changed; discarded stale calculation.")
            return
        self.show_result(result, *data)
        warmup_ms = self._worker.warmup_ms if self._worker is not None else self._warmup_ms
        self.status_label.setText(
            f"Complete · warm-up {warmup_ms:.0f} ms · solve {result.elapsed_ms:.0f} ms · "
            f"{result.iterations_completed:,} proposals · seed {result.seed} · {result.method}. "
            "Finite search; no global-optimum guarantee."
        )

    def _failed(self, message):
        self.invalidate_results()
        self.status_label.setText(message)

    def _restore(self):
        if self._worker is not None:
            self._warmup_ms = self._worker.warmup_ms
        self.optimize_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.hide()
        self._worker = None
        self._thread = None

    def _cancel_work(self):
        if self._cancel:
            self._cancel.set()
            self.status_label.setText("Cancellation requested…")

    def show_result(self, result, blades, rotor, geometry):
        self.result = result
        self._solved_snapshot = (blades, rotor, geometry)
        self.export_button.setEnabled(True)
        self.results_table.setRowCount(len(result.permutation))
        for slot, index in enumerate(result.permutation):
            vals = [
                str(slot + 1),
                f"{math.degrees(float(geometry.angles_rad[slot])):.3f}",
                blades[index].blade_name,
                f"{blades[index].weight:.6g}",
            ]
            for col, value in enumerate(vals):
                self.results_table.setItem(slot, col, QTableWidgetItem(value))
        self.cards.setText(
            f"RESIDUAL {result.residual_gmm:.5g} g·mm | NET ({result.net_ux_gmm:.5g}, {result.net_uy_gmm:.5g}) g·mm | {result.elapsed_ms:.0f} ms"
        )
        self.comparison_label.setText(
            f"Initial {result.initial_j_gmm:.6g} | Vector greedy {result.greedy_j_gmm:.6g} | "
            f"Pairing {result.pairing_j_gmm:.6g} | Conservative floor {result.lower_bound_gmm:.6g} g·mm"
        )
        phase = (
            f"{math.degrees(math.atan2(rotor.ey_mm, rotor.ex_mm)) % 360:.3f}°"
            if math.hypot(rotor.ex_mm, rotor.ey_mm) > 0
            else "undefined"
        )
        self.rotor_details_label.setText(
            f"Disk mass estimate {rotor.disk_mass_g:.6g} g | e=({rotor.ex_mm:.6g}, {rotor.ey_mm:.6g}) mm | "
            f"1× phase {phase} | Omitted RMS {rotor.runout_residual_rms_mm:.6g} mm\n"
            + " ".join((*rotor.warnings, *result.warnings))
        )
        self.status_label.setText(
            f"Complete · {result.iterations_completed:,} proposals · seed {result.seed} · {result.method}. Finite search; no global-optimum guarantee."
        )
        self._draw_result(result, blades, rotor, geometry)
        for canvas, preview in zip(self.canvases, self.overview_previews):
            canvas.draw()
            preview.setPixmap(canvas.grab().scaled(
                preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        self._update_step_header()
        self.tabs.setCurrentIndex(0)
        self.workspace_tabs.setCurrentIndex(1)

    def _draw_result(self, result, blades, rotor, geometry):
        for figure in self.figures:
            figure.clear()
        f = self.figures[0]
        ax = f.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        vectors = [
            ("Disk", rotor.disk_ux_gmm, rotor.disk_uy_gmm, "#d97706"),
            ("Initial blades", result.initial_blade_ux_gmm, result.initial_blade_uy_gmm, "#64748b"),
            ("Optimized blades", result.blade_ux_gmm, result.blade_uy_gmm, "#0f766e"),
            ("Net", result.net_ux_gmm, result.net_uy_gmm, "#be123c"),
        ]
        scale = max([math.hypot(x, y) for _, x, y, _ in vectors] + [1e-9])
        for name, x, y, color in vectors:
            mag = math.hypot(x, y)
            phase = math.degrees(math.atan2(y, x)) % 360 if mag else float("nan")
            ax.annotate(
                "",
                xy=(math.atan2(y, x), mag),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
            )
            ax.plot(
                [],
                [],
                color=color,
                label=f"{name}: {mag:.4g} g·mm, {phase:.1f}°" if mag else f"{name}: 0 g·mm, phase undefined",
            )
        ax.set_ylim(0, scale * 1.1)
        ax.set_rgrids([scale * fraction for fraction in (.25, .5, .75)], angle=115, fmt="%.0f")
        ax.set_title("Static unbalance vectors · g·mm", pad=22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=8)
        weights = [blades[i].weight for i in result.permutation]
        ax2 = self.figures[1].add_subplot(111)
        ax2.scatter(np.degrees(geometry.angles_rad), weights, c=weights, cmap="viridis")
        ax2.set(xlabel="Slot angle · ° (CCW)", ylabel="Blade mass · g", title="Mounting mass map")
        profile_ax = ax2.twinx()
        smooth = np.linspace(float(geometry.angles_rad[0]),
                             float(geometry.angles_rad[0]) + 2*np.pi, 361)
        profile_ax.plot(np.degrees(smooth), rotor.ex_mm*np.cos(smooth)+rotor.ey_mm*np.sin(smooth),
                        color="#a855f7", alpha=.7, label="1× radial projection")
        profile_ax.set_ylabel("1× radial deviation · mm", color="#9333ea")
        profile_ax.legend(loc="lower right", fontsize=8)
        ax3 = self.figures[2].add_subplot(111)
        ang = self.start_angle_input.value() + np.arange(8) * 45
        readings = rotor.runout_fit_mm if rotor.mode == "combined" else [0] * 8
        samples = rotor.runout_samples_mm if rotor.mode == "combined" else [0] * 8
        ax3.plot(ang, readings, "o-", label="Fitted 1×")
        ax3.plot(ang, samples, ".--", label="Measured samples")
        ax3.set(xlabel="Rotor angle · °", ylabel="Radial runout · mm", title="Runout harmonic fit")
        ax3.legend()
        ax4 = self.figures[3].add_subplot(111)
        if result.trace:
            ax4.plot([x for x, _ in result.trace], [y for _, y in result.trace])
        else:
            ax4.plot([0], [result.residual_gmm], "o")
        ax4.set(xlabel="Proposed swaps", ylabel="Best residual · g·mm",
                title="Search convergence" if result.trace else "Pairing baseline · no annealing")
        for canvas in self.canvases:
            canvas.draw_idle()

    def export_csv_path(self, path, overwrite=False):
        if self.result is None or self._solved_snapshot is None:
            raise ValueError("No current valid result to export")
        blades, rotor, geometry = self._solved_snapshot
        export_schedule_csv(
            path,
            blades,
            self.result.permutation,
            geometry.angles_rad,
            metadata={"mode": rotor.mode},
            overwrite=overwrite,
        )

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export mounting schedule", "mounting_schedule.csv", "CSV files (*.csv)"
        )
        if path:
            try:
                overwrite = False
                if __import__("os").path.exists(path):
                    answer = QMessageBox.question(
                        self,
                        "Replace existing file?",
                        f"{path} already exists. Replace it?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    overwrite = True
                self.export_csv_path(path, overwrite)
            except (ValueError, OSError) as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._cancel_work()
            self._thread.quit()
            self._thread.wait()
        event.accept()
