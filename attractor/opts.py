from dataclasses import dataclass
from .colormap import ColorMap

@dataclass
class Option:
    fps: int
    frames: int
    resolution: int
    iterations: int
    colormap: ColorMap = ColorMap("viridis")

    def __post_init__(self):
        self.total_time: float = round(self.frames / self.fps, 1)
    
    @staticmethod
    def from_time(
          seconds: float, 
          fps: int,
          iterations: int = 1_000_000,
          resolution: int = 1000,
          colormap: ColorMap = ColorMap("viridis"),
        ) -> "Option":

        return Option(
            fps=fps, 
            frames=round(seconds * fps), 
            iterations=iterations,
            resolution=resolution,
            colormap=colormap,
        )
