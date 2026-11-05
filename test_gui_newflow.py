from PySide6.QtWidgets import QMessageBox


def test_inventory_actions_create_blank_or_seeded_rows(qtbot, monkeypatch):
    from gui import MainWindow

    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    w = MainWindow()
    qtbot.addWidget(w)
    w.blade_count.setValue(5)
    w.generate_blank_inventory()
    assert w.workspace_tabs.currentIndex() == 0
    assert w.input_steps.currentWidget() is w.input_tab
    assert w.blade_table.rowCount() == 5
    assert [w.blade_table.item(i, 0).text() for i in range(5)] == [f"B{i:03}" for i in range(1, 6)]
    assert all(w.blade_table.item(i, 1).text() == "" for i in range(5))
    assert "manually" in notices[-1].lower()
    w.seed_input.setValue(123)
    w.load_random_samples()
    assert w.workspace_tabs.currentIndex() == 0
    assert w.input_steps.currentWidget() is w.input_tab
    assert w.blade_table.rowCount() == 5
    masses = [w.blade_table.item(i, 1).text() for i in range(5)]
    assert all(masses)
    assert "seed 123" in notices[-1].lower()


def test_installation_reference_tab_and_input_model(qtbot):
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert w.start_angle_input.value() == 0
    assert w.input_steps.tabText(w.installation_tab_index) == "4 Installation"
    before = w._revision
    w.start_angle_input.setValue(37.5)
    assert w._revision > before
    assert w._input_model().start_angle_deg == 37.5


def test_buttons_pulse_on_click_and_can_disable_motion(qtbot):
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.optimize_button.click()  # invalid blank masses, but click feedback must still work
    assert isinstance(w.optimize_button.graphicsEffect(), QGraphicsOpacityEffect)
    assert w._button_animations
    w.reduce_motion.setChecked(True)
    w.optimize_button.click()
    assert w.optimize_button.graphicsEffect().opacity() == 1.0


def test_tabs_have_compact_labels_and_hover_press_feedback(qtbot):
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert all(len(w.tabs.tabText(i)) <= 12 for i in range(w.tabs.count()))
    assert "QPushButton:hover" in w.styleSheet()
    assert "QPushButton:pressed" in w.styleSheet()


def test_runout_labels_are_relative_to_slot_one(qtbot):
    from gui import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert [w.runout_angle_labels[i].text() for i in range(8)] == [
        f"{i * 45}°" for i in range(8)
    ]
    assert "relative to slot 1" in w.runout_reference_label.text().lower()
