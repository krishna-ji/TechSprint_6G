from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer


# Track simulated transmission state
_tx_state = {
    "active": False,
    "channel": None,
    "frequency": None,
    "start_time": None,
    "packets_sent": 0
}


def simulate_transmit(main_window):
    """
    Simulate data transmission on the best available spectrum hole.
    Demonstrates cognitive radio's opportunistic spectrum access.
    """
    global _tx_state
    
    # Get system controller
    controller = getattr(main_window, 'system_controller', None)
    if controller is None or controller.sweeper is None:
        QMessageBox.warning(main_window, "Not Ready", 
                           "System not initialized. Please wait for spectrum sweep.")
        return
    
    sweeper = controller.sweeper
    
    if _tx_state["active"]:
        # Stop transmission
        _tx_state["active"] = False
        msg = f"""
📡 TRANSMISSION STOPPED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Channel: {_tx_state['channel']} ({_tx_state['frequency']/1e6:.1f} MHz)
Packets Sent: {_tx_state['packets_sent']}
Status: Secondary User vacated channel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(msg)
        main_window.statusBar().showMessage(f"TX Stopped - {_tx_state['packets_sent']} packets sent")
        QMessageBox.information(main_window, "Transmission Stopped", msg)
        return
    
    # Find best channel from latest sweep
    holes = sweeper.find_spectrum_holes()
    if not holes:
        QMessageBox.warning(main_window, "No Spectrum Holes",
                           "No free channels available!\nAll spectrum is occupied by Primary Users.")
        return
    
    # Select best channel (center of largest hole)
    best_channel = sweeper._select_best_channel(holes)
    best_freq = sweeper.get_channel_frequency(best_channel)
    occupancy = sweeper.channels[best_channel].occupancy
    
    # Start "transmission"
    import time
    _tx_state = {
        "active": True,
        "channel": best_channel,
        "frequency": best_freq,
        "start_time": time.time(),
        "packets_sent": 0
    }
    
    msg = f"""
📡 SIMULATED TRANSMISSION STARTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Channel: {best_channel}
Frequency: {best_freq/1e6:.1f} MHz
Occupancy: {occupancy:.1%} (threshold < 30%)
Status: 🟢 Secondary User accessing spectrum hole
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This simulates opportunistic spectrum access:
• Cognitive Radio detected a spectrum hole
• RL agent selected optimal channel
• Secondary User can transmit without interfering

Click "Simulate TX" again to stop transmission.
"""
    print(msg)
    main_window.statusBar().showMessage(f"📡 TX Active on Ch {best_channel} @ {best_freq/1e6:.1f} MHz")
    
    # Start packet counter
    def update_packets():
        if _tx_state["active"]:
            _tx_state["packets_sent"] += 10
            main_window.statusBar().showMessage(
                f"📡 TX Ch {best_channel} | {_tx_state['packets_sent']} packets | "
                f"Occ: {sweeper.channels[best_channel].occupancy:.1%}"
            )
    
    # Update every 500ms
    if not hasattr(main_window, '_tx_timer'):
        main_window._tx_timer = QTimer(main_window)
        main_window._tx_timer.timeout.connect(update_packets)
    main_window._tx_timer.start(500)
    
    QMessageBox.information(main_window, "Transmission Started", msg)


def show_spectrum_info(main_window):
    """Show detailed spectrum analysis dialog."""
    controller = getattr(main_window, 'system_controller', None)
    if controller is None or controller.sweeper is None:
        QMessageBox.warning(main_window, "Not Ready", 
                           "System not initialized. Please wait for spectrum sweep.")
        return
    
    sweeper = controller.sweeper
    
    # Build spectrum report
    report = """
╔══════════════════════════════════════════════════════════════╗
║                  SPECTRUM ANALYSIS REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
"""
    
    # Channel status
    report += f"║ Sweep Range: {sweeper.start_freq/1e6:.1f} - {sweeper.end_freq/1e6:.1f} MHz\n"
    report += f"║ Total Channels: {sweeper.n_channels}\n"
    report += f"║ Channel Spacing: {sweeper.channel_spacing/1e6:.2f} MHz\n"
    report += "║\n"
    
    # Spectrum map
    report += "║ SPECTRUM MAP:\n║ "
    for i, ch in enumerate(sweeper.channels):
        if ch.occupancy < 0.3:
            report += "🟢"
        elif ch.occupancy < 0.6:
            report += "🟡"
        else:
            report += "🔴"
    report += "\n║ " + "".join([str(i % 10) for i in range(sweeper.n_channels)]) + "\n"
    report += "║\n"
    
    # Detailed channel list
    report += "║ CHANNEL DETAILS:\n"
    for ch in sweeper.channels:
        status = "🟢 FREE" if ch.occupancy < 0.3 else ("🟡 WEAK" if ch.occupancy < 0.6 else "🔴 BUSY")
        report += f"║   Ch {ch.index:2d}: {ch.frequency/1e6:6.1f} MHz | {status} | {ch.modulation:8s} | {ch.power_db:+5.1f} dB\n"
    
    # Holes
    holes = sweeper.find_spectrum_holes()
    report += "║\n║ SPECTRUM HOLES (contiguous free channels):\n"
    if holes:
        for start, end in holes:
            start_freq = sweeper.get_channel_frequency(start)
            end_freq = sweeper.get_channel_frequency(end)
            report += f"║   Ch {start}-{end}: {start_freq/1e6:.1f} - {end_freq/1e6:.1f} MHz ({end-start+1} channels)\n"
    else:
        report += "║   No spectrum holes available!\n"
    
    report += "╚══════════════════════════════════════════════════════════════╝"
    
    print(report)
    QMessageBox.information(main_window, "Spectrum Analysis", report)


def capture_iq2(main_window):
    main_window.statusBar().showMessage("Capture IQ Started")
    print("Capture IQ tOOLBAR triggered")


def simulate2(main_window):
    main_window.statusBar().showMessage("Loading IQ Started")
    print("Load IQ TOOLBAR triggered")


def close_app2(main_window):
    main_window.statusBar().showMessage("Exiting application TOOLBAR ")
    QApplication.quit()
