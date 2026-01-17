"""
Menu Actions

Handles menu bar action callbacks for the main window.
"""

from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
import qtawesome as qta


def capture_iq(main_window):
    """Start capturing IQ data to file."""
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(
        main_window,
        "Save IQ Data",
        "",
        "Binary Files (*.bin);;All Files (*)",
        options=options
    )
    if not file_path:
        return

    # Use system radio if available
    radio = getattr(main_window.system, 'radio', None)
    if radio is None:
        main_window.statusBar().showMessage("No radio hardware available", 2000)
        return
    
    radio.open_file(file_path)
    main_window.statusBar().showMessage(f"Capture IQ Started: {file_path}")
    main_window.capture_button.setIcon(qta.icon("fa5s.stop-circle"))

    # Disable UI controls during capture
    if hasattr(main_window, 'menu_widget'):
        main_window.menu_widget.center_freq_slider.setEnabled(False)
        main_window.menu_widget.center_freq_input.setEnabled(False)
        main_window.menu_widget.sampling_rate_slider.setEnabled(False)
        main_window.menu_widget.sampling_rate_input.setEnabled(False)

    # Reconnect button to stop
    main_window.capture_button.triggered.disconnect()
    main_window.capture_button.triggered.connect(
        lambda: stop_capture(main_window)
    )


def stop_capture(main_window):
    """Stop capturing IQ data."""
    radio = getattr(main_window.system, 'radio', None)
    if radio is not None:
        radio.close_file()

    main_window.statusBar().showMessage("Capture IQ Stopped")
    main_window.capture_button.setIcon(qta.icon("fa5s.camera"))

    # Re-enable UI controls
    if hasattr(main_window, 'menu_widget'):
        main_window.menu_widget.center_freq_slider.setEnabled(True)
        main_window.menu_widget.center_freq_input.setEnabled(True)
        main_window.menu_widget.sampling_rate_slider.setEnabled(True)
        main_window.menu_widget.sampling_rate_input.setEnabled(True)

    main_window.capture_button.triggered.disconnect()
    main_window.capture_button.triggered.connect(
        lambda: capture_iq(main_window)
    )


def close_app(main_window):
    """Exit the application."""
    main_window.statusBar().showMessage("Exiting application")
    QApplication.quit()


def about_app(main_window):
    """Show about dialog."""
    copyright_info = "© 2025 TechSprint 6G. All rights reserved."
    msg_box = QMessageBox(main_window)
    msg_box.setWindowTitle("About")
    msg_box.setText(copyright_info)
    msg_box.exec()
