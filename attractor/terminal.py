import sys
from datetime import datetime
import os


class TerminalCounter:
    """This handles the interaction with the terminal for feedback"""
    def __init__(self, max_i, bar_length=40):
        self.max_i = max_i
        self.bar_length = bar_length
        self.i = 0
        self.timestamp_start = datetime.now()

    def start(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        self._update_terminal()

    def count_up(self):
        if self.i < self.max_i:
            self.i += 1
            self._update_terminal()
            if self.i == self.max_i:
                self.end()

    def eta(self):
        if self.i == 0:
            return "Estimating..."
        elapsed = datetime.now() - self.timestamp_start
        estimated_total = elapsed / self.i * self.max_i
        remaining = estimated_total - elapsed
        return str(remaining).split('.')[0]

    def _update_terminal(self):
        progress = self.i / self.max_i
        filled_len = int(self.bar_length * progress)
        bar = '█' * filled_len + '-' * (self.bar_length - filled_len)
        fps_val = f"{self.fps():.1f}"
        line = f"[{bar}] {self.i}/{self.max_i} ETA: {self.eta()} ({fps_val} fps)"

        sys.stdout.write("\r" + line + "\033[K")
        sys.stdout.flush()

    def fps(self):
        elapsed = (datetime.now() - self.timestamp_start).total_seconds()
        if elapsed == 0:
            return 0
        return self.i / elapsed
    
    def end(self):
        total_seconds = (datetime.now() - self.timestamp_start).total_seconds()
        avg_fps = self.i / total_seconds if total_seconds > 0 else 0.0

        # Clear progress line and print final stats
        sys.stdout.write(
            f"\rRendered {self.i} Frames "
            f"in {str(datetime.now() - self.timestamp_start).split('.')[0]} "
            f"({avg_fps:.1f} fps)\033[K\n"
        )
        sys.stdout.flush()
