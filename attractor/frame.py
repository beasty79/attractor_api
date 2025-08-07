from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional
from .api import apply_color
import numpy as np


@dataclass
class Frame:
    """
    This Class Represents one render instance
    Identifiers like, a, b, c, ... are initial value for the strange attractors
    """
    resolution: int
    percentile: float
    colors: NDArray
    n: int

    def __post_init__(self):
        # attributes only available after render
        self.img_: Optional[NDArray] = None
        self.raw_: Optional[NDArray] = None
        self.collapsed: bool = False

    @property
    def img(self) -> NDArray:
        if self.img_ is not None:
            return self.img_

        if self.raw is None:
            raise RuntimeError("Frame isn't rendered yet!")

        self.img_ = apply_color(self.raw, self.colors)
        return self.img_

    @img.setter
    def img(self, value: NDArray):
        self.img_ = value

    @property
    def raw(self) -> Optional[NDArray]:
        return self.raw_

    @raw.setter
    def raw(self, new_raw):
        self.raw_ = new_raw
        self.is_collapsed()

    def render(self, only_raw = False):
        if not only_raw and self.raw is not None:
            self.img = apply_color(self.raw, self.colors)

    def scatter_to_normalized(self, x_raw, y_raw):
        points_per_pixel = np.histogram2d(x_raw, y_raw, bins=self.resolution)[0]

        # clip outliers
        max_value = np.percentile(points_per_pixel, self.percentile)
        max_value = max_value if np.isfinite(max_value) and max_value > 0 else 1.0
        points_per_pixel = np.clip(points_per_pixel, 0, max_value)

        # normalize to [0,1]
        normalized = (points_per_pixel / np.max(points_per_pixel)).astype(np.float32)
        return normalized

    def is_collapsed(self):
        assert self.raw is not None, "first render"
        non_zero = np.count_nonzero(self.raw)
        thresh = self.resolution ** 2 * 0.05
        self.collapsed = non_zero < thresh


@dataclass
class SimonFrame(Frame):
    a: float
    b: float

    def render(self, only_raw = False):
        from .attractor import simon

        x, y = simon(self.a, self.b, self.n)
        self.raw = self.scatter_to_normalized(x, y)
        super().render(only_raw=only_raw)



@dataclass
class CliffordFrame(Frame):
    a: float
    b: float
    c: float
    d: float

    def init_args(self) -> tuple:
        return (self.a, self.b, self.c, self.d)

    def render(self, only_raw = False):
        from .attractor import clifford

        x, y = clifford(self.a, self.b, self.c, self.d, self.n)
        self.raw = self.scatter_to_normalized(x, y)
        super().render(only_raw=only_raw)

