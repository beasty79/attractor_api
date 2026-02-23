from dataclasses import dataclass
from .colormap import ColorMap
from typing import Optional

@dataclass
class Option:
    fps: int
    frames: int
    resolution: int
    colormap: Optional[ColorMap] = None
    alphaThreshold: int = 0

    def __post_init__(self):
        self.total_time: float = round(self.frames / self.fps, 1)
        self.colormap = ColorMap("viridis") if self.colormap is None else self.colormap

    @staticmethod
    def from_time(
            seconds: float,
            fps: int,
            resolution: int = 1000,
            colormap: Optional[ColorMap] = None,
            alphaThreshold = 0
        ) -> "Option":

        return Option(
            fps=fps,
            frames=round(seconds * fps),
            resolution=resolution,
            colormap=colormap,
            alphaThreshold=alphaThreshold
        )
