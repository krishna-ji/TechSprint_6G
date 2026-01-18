#!/usr/bin/env python3
"""
TechSprint 6G - Cognitive Radio System

Main entry point for the application.
Launches the PyQt6 UI for spectrum sensing and intelligent channel allocation.

Usage:
    python main.py                    # Simulation mode
    python main.py --use-hardware     # Use RTL-SDR hardware
    python main.py --help             # Show all options

Options:
    --use-hardware     Use RTL-SDR hardware for real spectrum sensing
    --simulate         Force simulation mode (default)
"""

import sys
import signal
import argparse
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ui"))


def main():
    """Launch the Cognitive Radio UI application."""
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="TechSprint 6G - Cognitive Radio with AI-based Spectrum Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run in simulation mode
  python main.py --use-hardware     # Run with RTL-SDR hardware
  
Hardware Requirements:
  - RTL-SDR dongle (RTL2832U based)
  - SoapySDR + rtl-sdr drivers installed
        """
    )
    parser.add_argument('--use-hardware', action='store_true', 
                       help='Use RTL-SDR hardware for real spectrum sensing')
    parser.add_argument('--simulate', action='store_true',
                       help='Force simulation mode (default)')
    parser.add_argument('--hardware', action='store_true', 
                       help='Alias for --use-hardware')
    args = parser.parse_args()
    
    use_hardware = (args.use_hardware or args.hardware) and not args.simulate
    
    # Change to UI directory for relative paths (style.qss, icons)
    ui_dir = PROJECT_ROOT / "src" / "ui"
    os.chdir(ui_dir)
    
    # Import UI components - use new dashboard
    from PyQt6.QtWidgets import QApplication
    from main_dashboard import CognitiveRadioDashboard
    
    # Print startup mode
    if use_hardware:
        print("🔌 Starting with HARDWARE mode (RTL-SDR)")
    else:
        print("🖥️  Starting with SIMULATION mode")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Load stylesheet
    style_path = ui_dir / "style.qss"
    if style_path.exists():
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
    
    # Create and show main window
    window = CognitiveRadioDashboard(use_hardware=use_hardware)
    window.show()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nSignal received, closing...")
        window.close()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
