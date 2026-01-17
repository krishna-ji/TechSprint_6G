# config.py
"""
UI Configuration and Constants

Centralized configuration for the 6G Cognitive Radio UI application.
All magic numbers and configuration values should be defined here.

Sections
--------
- Window Configuration: Main window dimensions and title
- Signal Processing: Sample sizes, FFT parameters, channel count
- UI Colors: Color scheme for charts and UI elements
- Refresh Rates: Update intervals for charts and probability display
- Hardware Configuration: RTL-SDR default parameters

Usage
-----
>>> from config import SAMPLE_SIZE, N_CHANNELS, SAMPLE_RATE
>>> fft_result = np.fft.fft(iq_data, SAMPLE_SIZE)
"""

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================
WINDOW_TITLE = "Intelligent 6G Cognitive Radio"
X_OFFSET = 0
Y_OFFSET = 0
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

# =============================================================================
# SIGNAL PROCESSING CONSTANTS
# =============================================================================
SAMPLE_SIZE = 1024          # IQ samples per frame
N_CHANNELS = 20             # Number of spectrum channels for RL
FFT_SIZE = 1024             # FFT size for frequency analysis
SAMPLE_RATE = 1.024e6       # Sample rate in Hz
CENTER_FREQUENCY = 95.6e6   # Default center frequency in Hz

# =============================================================================
# UI COLORS
# =============================================================================
SIDEBAR_COLOR = "#2c3e50"
BUTTON_COLOR = "#34495e"
APP_BACKGROUND_COLOR = '121212'

# Chart colors
CONSTELLATION_PLOT_COLOR = '#F9C74F'
FREQUENCY_DOMAIN_PLOT_COLOR = '#FF6F91'
TIME_DOMAIN_PLOT_COLOR = '#A2C2E0'
WATERFALL_PLOT_COLOR = '#ff6f91'
PROBABILITY_BAR_COLOR = '#6A82FB'

# =============================================================================
# REFRESH RATES (milliseconds)
# =============================================================================
FIGURE_REFRESH_RATE = 200
PROBABILITY_REFRESH_RATE = 1000

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
VARS = {
    "SAMPLE_RATE": SAMPLE_RATE,
    "FFT_SIZE": FFT_SIZE,
    "NUM_ROWS": 100,
    "CENTER_FREQUENCY": CENTER_FREQUENCY,
    "GAIN": 10,
    "BANDWIDTH": 1e6,
}
