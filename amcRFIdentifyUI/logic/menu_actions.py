# Import the simulation window class
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
import qtawesome as qta
import time

capture_running = False


def capture_iq(main_window):
    global capture_running
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

    flowgraph = main_window.plot_constellation_widget.flowgraph
    # Open the file sink for writing
    flowgraph.open_file(file_path)
    capture_running = True

    main_window.statusBar().showMessage(f"Capture IQ Started: {file_path}")
    main_window.capture_button.setIcon(qta.icon("fa5s.stop-circle"))

    # Disable UI controls
    main_window.menu_widget.center_freq_slider.setEnabled(False)
    main_window.menu_widget.center_freq_input.setEnabled(False)
    main_window.menu_widget.sampling_rate_slider.setEnabled(False)
    main_window.menu_widget.sampling_rate_input.setEnabled(False)

    # Reconnect the capture button to stop
    main_window.capture_button.triggered.disconnect()
    main_window.capture_button.triggered.connect(
        lambda: stop_capture(main_window)
    )
    print(f"Capture IQ started, saving to {file_path}")


def stop_capture(main_window):
    global capture_running
    flowgraph = main_window.plot_constellation_widget.flowgraph

    flowgraph.close_file()
    capture_running = False

    main_window.statusBar().showMessage("Capture IQ Stopped")
    main_window.capture_button.setIcon(qta.icon("fa5s.camera"))

    # Re-enable UI controls
    main_window.menu_widget.center_freq_slider.setEnabled(True)
    main_window.menu_widget.center_freq_input.setEnabled(True)
    main_window.menu_widget.sampling_rate_slider.setEnabled(True)
    main_window.menu_widget.sampling_rate_input.setEnabled(True)

    main_window.capture_button.triggered.disconnect()
    main_window.capture_button.triggered.connect(
        lambda: capture_iq(main_window)
    )
    print("Capture IQ stopped and file closed.")


def simulate(main_window):
    main_window.statusBar().showMessage("Loading IQ Data for Simulation", 2000)

    # Update the main window's widgets for simulation
    main_window.plot_constellation_widget.setVisible(False)
    main_window.simulation_widget.setVisible(True)

    # Update other UI elements as needed
    main_window.menu_widget.center_freq_slider.setEnabled(False)
    main_window.menu_widget.center_freq_input.setEnabled(False)
    main_window.menu_widget.sampling_rate_slider.setEnabled(False)
    main_window.menu_widget.sampling_rate_input.setEnabled(False)

    main_window.statusBar().showMessage("Simulation Mode Activated", 2000)
    print("Simulation Mode Activated")


def close_app(main_window):
    main_window.statusBar().showMessage("Exiting application")
    QApplication.quit()


def about_app(main_window):
    copyright_info = "© 2025 Your Company. All rights reserved."
    main_window.statusBar().showMessage("About Dialog")
    msg_box = QMessageBox(main_window)
    msg_box.setWindowTitle("About")
    msg_box.setText(copyright_info)
    msg_box.exec()
    print("About Dialog triggered")
    print(copyright_info)
