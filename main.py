#!/usr/bin/env python3
"""
TechSprint 6G - Cognitive Radio System

Main entry point for the application.
Launches the PyQt6 UI for spectrum sensing and modulation classification.

Usage:
    python main.py [--hardware]

Options:
    --hardware    Use RTL-SDR hardware instead of simulation mode
"""

import sys
import signal
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon


def main():
    """Launch the Cognitive Radio UI application."""
    import os
    
    # Set up paths
    project_root = Path(__file__).parent
    ui_dir = project_root / "src" / "ui"
    
    # Add ui directory to path for its internal imports (widgets, logic, etc.)
    sys.path.insert(0, str(ui_dir))
    
    # Change to UI directory for relative paths (style.qss, icons)
    os.chdir(ui_dir)
    
    # Import after path setup
    from main import MainWindow
    
    app = QApplication(sys.argv)
    
    # Load stylesheet
    style_path = ui_dir / "style.qss"
    if style_path.exists():
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        window.close()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
