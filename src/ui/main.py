from widgets.charts.waterfall import PlotWaterfallDiagram
from widgets.charts.timedomain import PlotTimeDomain
from widgets.charts.freqdomain import PlotFrequencyDomain
from widgets.charts.constellation import PlotConstellation
import qtawesome as qta
from widgets.menu import MenuWidget
from widgets.charts.probability import BarGraphWidget
from interface.toolbar import create_tool_bar
from interface.menu_bar import create_menu_bar
from config import *
from core.system import SystemController
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QStatusBar, QDockWidget, QLabel, QFrame, QHBoxLayout, QGridLayout
import signal
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulation_playground_window = None
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(X_OFFSET, Y_OFFSET, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.setWindowIcon(
            QIcon(qta.icon('mdi.signal-5g').pixmap(32, 32)))        # set Icon
        # self.setWindowIcon(qta.icon("icon.png"))

        # Create the status bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Create the menu bar
        self.setMenuBar(create_menu_bar(self))

        # Create the Tool Bar
        tool_bar = create_tool_bar(self)
        tool_bar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tool_bar)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # --- System Controller (The Brain) ---
        self.system = SystemController()
        
        # --- Heads-Up Display (Status Panel) ---
        # Instead of a dock, we place this at the top of the central layout ("The Dashboard")
        self.status_frame = QFrame()
        self.status_frame.setObjectName("statusFrame") # For QSS styling
        self.status_layout = QGridLayout(self.status_frame)
        
        # Titles
        lbl_mod_title = QLabel("DETECTED MODULATION")
        lbl_mod_title.setObjectName("statusTitle")
        lbl_chan_title = QLabel("ACTIVE CHANNEL")
        lbl_chan_title.setObjectName("statusTitle")
        
        # Values (Styled as badged/LCD)
        self.lbl_mod_val = QLabel("WAITING...")
        self.lbl_mod_val.setObjectName("modLabel") 
        self.lbl_chan_val = QLabel("SCANNING...")
        self.lbl_chan_val.setObjectName("chanLabel")
        
        # Add to grid
        self.status_layout.addWidget(lbl_mod_title, 0, 0)
        self.status_layout.addWidget(self.lbl_mod_val, 1, 0)
        self.status_layout.addWidget(lbl_chan_title, 0, 1)
        self.status_layout.addWidget(self.lbl_chan_val, 1, 1)
        
        # Add status frame to top of main layout
        self.layout.addWidget(self.status_frame)

        # Create and add the dockable plot widgets
        # Enable external driving for constellation to use SystemController data
        self.plot_constellation_widget = PlotConstellation(external_drive=True)
        dock_widget1 = QDockWidget("Constellation Plot", self)
        dock_widget1.setWidget(self.plot_constellation_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_widget1)

        self.plot_widget2 = PlotFrequencyDomain()
        dock_widget2 = QDockWidget("Frequency Domain Plot", self)
        dock_widget2.setWidget(self.plot_widget2)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget2)

        self.plot_widget3 = PlotTimeDomain()
        dock_widget3 = QDockWidget("Time Domain Plot", self)
        dock_widget3.setWidget(self.plot_widget3)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_widget3)

        self.plot_widget4 = PlotWaterfallDiagram(
            VARS["FFT_SIZE"], VARS["NUM_ROWS"])
        dock_widget4 = QDockWidget("Waterfall Plot", self)
        dock_widget4.setWidget(self.plot_widget4)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget4)
        
        # --- Removed Old Status Dock (Now in HUD Frame ) ---

        # Create and add the dockable bar graph widget
        self.bar_graph_widget = BarGraphWidget()
        dock_widget_bar = QDockWidget("Bar Graph", self)
        dock_widget_bar.setWidget(self.bar_graph_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, dock_widget_bar)

        # Create and add another dockable menu on the left side
        another_dock_widget = QDockWidget("Another Menu", self)
        self.menu_widget = MenuWidget(self)  # The menu widget reference
        another_dock_widget.setWidget(self.menu_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, another_dock_widget)

        # Dock arrangement - Improved Tiled Layout
        # Stack Secondary plots and Tabify similar ones
        self.splitDockWidget(dock_widget1, dock_widget3, Qt.Orientation.Vertical) # Constellation over Time
        self.splitDockWidget(dock_widget2, dock_widget4, Qt.Orientation.Vertical) # Freq over Waterfall
        
        # Tabify Time and Freq Domain (Secondary views)
        self.tabifyDockWidget(dock_widget3, dock_widget2)
        
        # Ensure Waterfall is prominent
        self.resizeDocks([dock_widget4], [400], Qt.Orientation.Vertical)
        
        # --- Connect System Controller Signals ---
        # All data flows from SystemController (single source of truth)
        
        # IQ data to constellation and other plots
        self.system.update_plots.connect(self.plot_constellation_widget.update_from_external)
        self.system.update_plots.connect(self.plot_widget2.data_gen.generate_frequency_domain_data)
        self.system.update_plots.connect(
            lambda iq_data: self.plot_widget3.data_gen.generate_time_domain_data(
                iq_data, VARS["SAMPLE_RATE"])
        )
        self.system.update_plots.connect(
            lambda iq_data: self.plot_widget4.update_plot(iq_data)
        )
        
        # Probabilities to bar graph (no duplicate inference!)
        self.system.update_probs.connect(self.bar_graph_widget.update_probabilities)
        
        # Status updates
        self.system.update_status.connect(self.update_status_display)
        
        # Start the system
        self.system.start_system()

    def update_status_display(self, mod_text, chan_text):
        self.lbl_mod_val.setText(mod_text)
        self.lbl_chan_val.setText(chan_text)
        self.status_bar.showMessage(f"{mod_text} | {chan_text}")

    def closeEvent(self, event):
        print("Closing application...")
        # Stop system
        self.system.stop_system()
        # Stop the constellation flowgraph on exit
        self.plot_constellation_widget.close()
        event.accept()


# class SimulationPlaygroundWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Simulation Playground")
#         self.setGeometry(100, 100, 800, 600)
#         self.setWindowIcon(QIcon(qta.icon('mdi.play-circle').pixmap(32, 32)))

#         # Add your simulation playground widgets and layout here
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.layout = QVBoxLayout(self.central_widget)
#         # Example widget
#         self.label = QLabel("Welcome to the Simulation Playground!")
#         self.layout.addWidget(self.label)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and apply the QSS file
    with open("style.qss", "r") as file:
        app.setStyleSheet(file.read())

    main_window = MainWindow()
    main_window.show()

    def signal_handler(sig, frame):
        print("Signal received, closing application...")
        main_window.close()

    signal.signal(signal.SIGINT, signal_handler)
    sys.exit(app.exec())
