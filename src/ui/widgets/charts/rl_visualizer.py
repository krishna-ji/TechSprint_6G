"""
RL Decision Visualizer

Displays the RL agent's decision-making process including:
- Current recommendation
- Action probabilities
- Decision reasoning
- Historical performance
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QProgressBar, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QColor
import numpy as np
from collections import deque


class RLDecisionPanel(QWidget):
    """
    Panel showing RL agent's decision-making visualization.
    """
    
    def __init__(self, n_channels: int = 20):
        super().__init__()
        self.n_channels = n_channels
        
        # Decision history
        self.decision_history = deque(maxlen=50)
        self.switch_count = 0
        self.stay_count = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("RL DECISION ENGINE")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Main recommendation card
        self.rec_frame = QFrame()
        self.rec_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 2px solid #3498DB;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        rec_layout = QVBoxLayout(self.rec_frame)
        rec_layout.setSpacing(5)
        
        # Recommended channel
        self.rec_label = QLabel("RECOMMENDED: CH --")
        self.rec_label.setStyleSheet("""
            color: #3498DB;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
        """)
        self.rec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rec_layout.addWidget(self.rec_label)
        
        # Frequency
        self.freq_label = QLabel("@ ---.-- MHz")
        self.freq_label.setStyleSheet("color: #888; font-size: 12px;")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rec_layout.addWidget(self.freq_label)
        
        layout.addWidget(self.rec_frame)
        
        # Decision details grid
        details_grid = QGridLayout()
        details_grid.setSpacing(5)
        
        # Action
        details_grid.addWidget(QLabel("Action:"), 0, 0)
        self.action_label = QLabel("--")
        self.action_label.setStyleSheet("color: #2ECC71; font-weight: bold;")
        details_grid.addWidget(self.action_label, 0, 1)
        
        # Confidence
        details_grid.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 3px;
                background-color: #1E1E2E;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 2px;
            }
        """)
        details_grid.addWidget(self.confidence_bar, 1, 1)
        
        # Occupancy
        details_grid.addWidget(QLabel("Occupancy:"), 2, 0)
        self.occupancy_label = QLabel("--%")
        self.occupancy_label.setStyleSheet("color: #2ECC71;")
        details_grid.addWidget(self.occupancy_label, 2, 1)
        
        # Reason
        details_grid.addWidget(QLabel("Reason:"), 3, 0)
        self.reason_label = QLabel("--")
        self.reason_label.setStyleSheet("color: #888; font-style: italic;")
        self.reason_label.setWordWrap(True)
        details_grid.addWidget(self.reason_label, 3, 1)
        
        layout.addLayout(details_grid)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #333;")
        layout.addWidget(sep)
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.switches_label = QLabel("Switches: 0")
        self.switches_label.setStyleSheet("color: #E74C3C; font-size: 11px;")
        stats_layout.addWidget(self.switches_label)
        
        self.stays_label = QLabel("Stays: 0")
        self.stays_label.setStyleSheet("color: #2ECC71; font-size: 11px;")
        stats_layout.addWidget(self.stays_label)
        
        self.efficiency_label = QLabel("Efficiency: --%")
        self.efficiency_label.setStyleSheet("color: #3498DB; font-size: 11px;")
        stats_layout.addWidget(self.efficiency_label)
        
        layout.addLayout(stats_layout)
        
        # Action probabilities visualization
        probs_label = QLabel("Channel Scores:")
        probs_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(probs_label)
        
        self.probs_layout = QHBoxLayout()
        self.probs_layout.setSpacing(1)
        
        self.prob_bars = []
        for i in range(n_channels):
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setRange(0, 100)
            bar.setValue(50)
            bar.setFixedWidth(15)
            bar.setTextVisible(False)
            bar.setStyleSheet("""
                QProgressBar {
                    border: none;
                    background-color: #1E1E2E;
                }
                QProgressBar::chunk {
                    background-color: #3498DB;
                }
            """)
            self.prob_bars.append(bar)
            self.probs_layout.addWidget(bar)
        
        layout.addLayout(self.probs_layout)
        
        layout.addStretch()
    
    @pyqtSlot(dict)
    def update_decision(self, decision: dict):
        """
        Update with new RL decision.
        
        Parameters
        ----------
        decision : dict
            Contains: channel, freq, action, confidence, occupancy, 
                     reason, action_probs, prev_channel
        """
        channel = decision.get('channel', 0)
        freq = decision.get('freq', 0) / 1e6
        action = decision.get('action', 'STAY')
        confidence = decision.get('confidence', 0)
        occupancy = decision.get('occupancy', 0)
        reason = decision.get('reason', '')
        action_probs = decision.get('action_probs', np.zeros(self.n_channels))
        prev_channel = decision.get('prev_channel', channel)
        
        # Update recommendation
        self.rec_label.setText(f"RECOMMENDED: CH {channel:02d}")
        self.freq_label.setText(f"@ {freq:.2f} MHz")
        
        # Update action
        if action == 'SWITCH':
            self.action_label.setText(f"⇢ SWITCH ({prev_channel}→{channel})")
            self.action_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
            self.switch_count += 1
            self.rec_frame.setStyleSheet("""
                QFrame {
                    background-color: #252526;
                    border: 2px solid #E74C3C;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        else:
            self.action_label.setText("● STAY")
            self.action_label.setStyleSheet("color: #2ECC71; font-weight: bold;")
            self.stay_count += 1
            self.rec_frame.setStyleSheet("""
                QFrame {
                    background-color: #252526;
                    border: 2px solid #2ECC71;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        
        # Update confidence
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update occupancy
        occ_pct = occupancy * 100
        self.occupancy_label.setText(f"{occ_pct:.0f}%")
        if occ_pct < 30:
            self.occupancy_label.setStyleSheet("color: #2ECC71;")
        elif occ_pct < 60:
            self.occupancy_label.setStyleSheet("color: #F1C40F;")
        else:
            self.occupancy_label.setStyleSheet("color: #E74C3C;")
        
        # Update reason
        self.reason_label.setText(reason or "Optimal channel selection")
        
        # Update statistics
        self.switches_label.setText(f"Switches: {self.switch_count}")
        self.stays_label.setText(f"Stays: {self.stay_count}")
        
        total = self.switch_count + self.stay_count
        if total > 0:
            efficiency = (self.stay_count / total) * 100
            self.efficiency_label.setText(f"Stability: {efficiency:.0f}%")
        
        # Update probability bars
        if len(action_probs) == self.n_channels:
            max_prob = max(action_probs) if max(action_probs) > 0 else 1
            for i, (bar, prob) in enumerate(zip(self.prob_bars, action_probs)):
                normalized = int((prob / max_prob) * 100) if max_prob > 0 else 0
                bar.setValue(normalized)
                
                if i == channel:
                    bar.setStyleSheet("""
                        QProgressBar {
                            border: none;
                            background-color: #1E1E2E;
                        }
                        QProgressBar::chunk {
                            background-color: #2ECC71;
                        }
                    """)
                else:
                    bar.setStyleSheet("""
                        QProgressBar {
                            border: none;
                            background-color: #1E1E2E;
                        }
                        QProgressBar::chunk {
                            background-color: #3498DB;
                        }
                    """)
        
        # Record decision
        self.decision_history.append({
            'channel': channel,
            'action': action,
            'occupancy': occupancy
        })
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.switch_count = 0
        self.stay_count = 0
        self.decision_history.clear()


class AllocationIntelligencePanel(QWidget):
    """
    Panel showing spectrum allocation intelligence metrics.
    """
    
    def __init__(self, n_channels: int = 20):
        super().__init__()
        self.n_channels = n_channels
        
        # History tracking
        self.occupancy_history = deque(maxlen=100)
        self.hole_sizes = deque(maxlen=50)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("ALLOCATION INTELLIGENCE")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(5)
        
        # Spectrum efficiency
        metrics_grid.addWidget(QLabel("Spectrum Efficiency:"), 0, 0)
        self.efficiency_label = QLabel("--%")
        self.efficiency_label.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 14px;")
        metrics_grid.addWidget(self.efficiency_label, 0, 1)
        
        # Average hole size
        metrics_grid.addWidget(QLabel("Avg Hole Size:"), 1, 0)
        self.hole_size_label = QLabel("-- ch")
        self.hole_size_label.setStyleSheet("color: #3498DB; font-weight: bold;")
        metrics_grid.addWidget(self.hole_size_label, 1, 1)
        
        # Free channels
        metrics_grid.addWidget(QLabel("Free Channels:"), 2, 0)
        self.free_label = QLabel("-- / 20")
        self.free_label.setStyleSheet("color: #2ECC71;")
        metrics_grid.addWidget(self.free_label, 2, 1)
        
        # Occupied channels
        metrics_grid.addWidget(QLabel("Occupied (PU):"), 3, 0)
        self.occupied_label = QLabel("-- / 20")
        self.occupied_label.setStyleSheet("color: #E74C3C;")
        metrics_grid.addWidget(self.occupied_label, 3, 1)
        
        layout.addLayout(metrics_grid)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #333;")
        layout.addWidget(sep)
        
        # Spectrum holes visualization
        holes_label = QLabel("Spectrum Holes (contiguous free):")
        holes_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(holes_label)
        
        self.holes_text = QLabel("Scanning...")
        self.holes_text.setStyleSheet("""
            color: #2ECC71; 
            font-family: monospace; 
            font-size: 11px;
            background-color: #1E1E2E;
            padding: 5px;
            border-radius: 4px;
        """)
        self.holes_text.setWordWrap(True)
        layout.addWidget(self.holes_text)
        
        # Trend indicator
        trend_layout = QHBoxLayout()
        trend_label = QLabel("Spectrum Trend:")
        trend_label.setStyleSheet("color: #888; font-size: 10px;")
        trend_layout.addWidget(trend_label)
        
        self.trend_indicator = QLabel("━━━")
        self.trend_indicator.setStyleSheet("color: #888; font-size: 14px;")
        trend_layout.addWidget(self.trend_indicator)
        trend_layout.addStretch()
        
        layout.addLayout(trend_layout)
        
        layout.addStretch()
    
    @pyqtSlot(dict)
    def update_intelligence(self, data: dict):
        """
        Update intelligence metrics.
        
        Parameters
        ----------
        data : dict
            Contains: occupancy_array, spectrum_holes, free_count, 
                     occupied_count, efficiency
        """
        occupancy = data.get('occupancy_array', np.zeros(self.n_channels))
        holes = data.get('spectrum_holes', [])
        free_count = data.get('free_count', 0)
        occupied_count = data.get('occupied_count', 0)
        
        # Calculate efficiency (free channels / total)
        efficiency = (free_count / self.n_channels) * 100
        self.efficiency_label.setText(f"{efficiency:.0f}%")
        
        if efficiency > 50:
            self.efficiency_label.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 14px;")
        elif efficiency > 25:
            self.efficiency_label.setStyleSheet("color: #F1C40F; font-weight: bold; font-size: 14px;")
        else:
            self.efficiency_label.setStyleSheet("color: #E74C3C; font-weight: bold; font-size: 14px;")
        
        # Convert holes from list of channel indices to contiguous ranges
        hole_ranges = []
        if holes and len(holes) > 0:
            # holes is a list of free channel indices like [2, 3, 4, 7, 8]
            sorted_holes = sorted(holes)
            if len(sorted_holes) > 0:
                start = sorted_holes[0]
                end = sorted_holes[0]
                
                for ch in sorted_holes[1:]:
                    if ch == end + 1:
                        end = ch
                    else:
                        hole_ranges.append((start, end))
                        start = ch
                        end = ch
                hole_ranges.append((start, end))
        
        # Average hole size
        if hole_ranges:
            avg_hole = np.mean([end - start + 1 for start, end in hole_ranges])
            self.hole_sizes.append(avg_hole)
            self.hole_size_label.setText(f"{avg_hole:.1f} ch")
        else:
            self.hole_size_label.setText("0 ch")
        
        # Free/occupied counts
        self.free_label.setText(f"{free_count} / {self.n_channels}")
        self.occupied_label.setText(f"{occupied_count} / {self.n_channels}")
        
        # Holes text
        if hole_ranges:
            holes_str = "  ".join([f"[{s}-{e}]" for s, e in hole_ranges])
            self.holes_text.setText(holes_str)
            self.holes_text.setStyleSheet("""
                color: #2ECC71; 
                font-family: monospace; 
                font-size: 11px;
                background-color: #1E1E2E;
                padding: 5px;
                border-radius: 4px;
            """)
        else:
            self.holes_text.setText("No spectrum holes available")
            self.holes_text.setStyleSheet("""
                color: #E74C3C; 
                font-family: monospace; 
                font-size: 11px;
                background-color: #1E1E2E;
                padding: 5px;
                border-radius: 4px;
            """)
        
        # Update trend
        self.occupancy_history.append(np.mean(occupancy))
        
        if len(self.occupancy_history) >= 5:
            recent = list(self.occupancy_history)[-5:]
            trend = recent[-1] - recent[0]
            
            if trend > 0.05:
                self.trend_indicator.setText("↗ Increasing")
                self.trend_indicator.setStyleSheet("color: #E74C3C; font-size: 12px;")
            elif trend < -0.05:
                self.trend_indicator.setText("↘ Decreasing")
                self.trend_indicator.setStyleSheet("color: #2ECC71; font-size: 12px;")
            else:
                self.trend_indicator.setText("→ Stable")
                self.trend_indicator.setStyleSheet("color: #F1C40F; font-size: 12px;")
