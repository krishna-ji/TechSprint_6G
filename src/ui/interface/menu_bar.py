from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction
import qtawesome as qta
from logic.menu_actions import *
from logic.toolbar_actions import *
from logic.radio_actions import *


def create_menu_bar(parent):
    menu_bar = QMenuBar(parent)

    # Create menus and add them to the menu bar
    # Create menus
    file_menu = QMenu("File", parent)
    view_menu = QMenu("View", parent)
    help_menu = QMenu("Help", parent)
    radio_menu = QMenu("Radio", parent)
    # with icons

    # Add menus to the menu bar
    menu_bar.addMenu(file_menu)
    menu_bar.addMenu(radio_menu)
    menu_bar.addMenu(view_menu)
    menu_bar.addMenu(help_menu)

    # FILE->MENU SUBMENU
    capture_iq_action = QAction(qta.icon(
        'fa5s.camera'), "Capture IQ", parent, triggered=lambda: capture_iq(parent))
    simulate_action = QAction(qta.icon(
        'fa5s.folder-open'), "Simulation Window", parent, triggered=lambda: simulate(parent))
    exit_amc_action = QAction(qta.icon(
        'mdi.location-exit'), "Exit AMC", parent, triggered=lambda: close_app(parent))

    file_menu.addAction(capture_iq_action)
    file_menu.addAction(simulate_action)
    file_menu.addSeparator()
    file_menu.addAction(exit_amc_action)

    # VIEW->MENU SUBMENU
    maximize_action = QAction(
        qta.icon('fa5s.expand'), "Maximize", parent, triggered=parent.showMaximized)
    minimize_action = QAction(
        qta.icon('fa5s.compress'), "Minimize", parent, triggered=parent.showMinimized)

    view_menu.addAction(maximize_action)
    view_menu.addAction(minimize_action)

    # RADIO->MENU SUBMENU
    radio1_action = QAction(qta.icon(
        'fa5s.play'), "Start Playback", parent, triggered=lambda: play_radio(parent))
    radio2_action = QAction(qta.icon(
        'fa5s.stop'), "Stop Playback", parent, triggered=lambda: stop_radio(parent))

    radio_menu.addAction(radio1_action)
    radio_menu.addAction(radio2_action)

    # HELP->MENU SUBMENU
    about_action = QAction(qta.icon('fa5s.info-circle'),
                           "About", parent, triggered=lambda: about_app(parent))
    help_menu.addAction(about_action)

    return menu_bar
