import cv2
import numpy as np
import os

class AlphaChannel:
    def __init__(self, filename: str, fps: int = 30):
        self.filename = filename
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.frame_size: tuple[int, int] | None = None
        self.is_color: bool | None = None
        self.initialized = False

    def _init_writer(self, frame_shape):
        self.frame_size = (frame_shape[1], frame_shape[0])
        # Treat single-channel (H,W,1) as grayscale
        self.is_color = not (len(frame_shape) == 3 and frame_shape[2] == 1)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size, isColor=self.is_color)
        self.initialized = True

    def add_frame(self, frame: np.ndarray, a: float | None = None, b: float | None = None):
        if not self.initialized:
            self._init_writer(frame.shape)

        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            return

        # Convert RGBA to grayscale if input has 4 channels
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

        # Convert (H,W,1) -> (H,W) for grayscale VideoWriter
        if len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        # Add text overlay
        if a is not None and b is not None:
            # For grayscale, white text; for color, BGR white
            color = 255 if len(frame.shape) == 2 else (255, 255, 255)
            cv2.putText(frame, f"a = {a:.4f}, b = {b:.4f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        if self.writer is not None:
            self.writer.write(frame)

    def save(self):
        if self.writer is not None:
            self.writer.release()
            print(f"save to => '{os.path.abspath(self.filename)}'")