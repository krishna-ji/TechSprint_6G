from PyQt6.QtWidgets import QDialog, QVBoxLayout, QSlider, QLabel, QLineEdit, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from config import VARS


# def play_radio(main_window):
#     main_window.statusBar().showMessage("FM radio unmuted", 2000)
#     if hasattr(main_window, "plot_constellation_widget"):
#         flowgraph = main_window.plot_constellation_widget.flowgraph
#         flowgraph.unmute_fm()


# def stop_radio(main_window):
#     main_window.statusBar().showMessage("FM radio muted", 2000)
#     if hasattr(main_window, "plot_constellation_widget"):
#         flowgraph = main_window.plot_constellation_widget.flowgraph
#         flowgraph.mute_fm()


# from PyQt6.QtWidgets import QMessageBox


def play_radio(main_window):
    # Assume that bar_graph_widget contains the probability data
    fm_threshold = 0.5
    prob_data = None
    # Check if bar_graph_widget exists and has probability data
    if hasattr(main_window, "bar_graph_widget"):
        prob_data = main_window.bar_graph_widget.prob_mod.data

    if len(prob_data) == 0:
        main_window.statusBar().showMessage("No FM detected", 2000)
        return

    fm_prob = prob_data[7]
    # Ensure FM probability is the maximum and above threshold
    if fm_prob < fm_threshold or fm_prob != max(prob_data):
        main_window.statusBar().showMessage("No FM detected", 2000)
        return

    main_window.statusBar().showMessage("FM radio unmuted", 2000)
    if hasattr(main_window, "plot_constellation_widget"):
        flowgraph = main_window.plot_constellation_widget.flowgraph
        flowgraph.unmute_fm()


def stop_radio(main_window):
    main_window.statusBar().showMessage("FM radio muted", 2000)
    if hasattr(main_window, "plot_constellation_widget"):
        flowgraph = main_window.plot_constellation_widget.flowgraph
        flowgraph.mute_fm()
