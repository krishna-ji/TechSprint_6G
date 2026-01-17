from PyQt6.QtWidgets import QApplication


def capture_iq2(main_window):
    main_window.statusBar().showMessage("Capture IQ Started")
    print("Capture IQ tOOLBAR triggered")


def simulate2(main_window):
    main_window.statusBar().showMessage("Loading IQ Started")
    print("Load IQ TOOLBAR triggered")


def close_app2(main_window):
    main_window.statusBar().showMessage("Exiting application TOOLBAR ")
    QApplication.quit()
