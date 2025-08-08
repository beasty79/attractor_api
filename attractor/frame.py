from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional, Iterable
from .api import apply_color
import numpy as np


@dataclass
class Frame:
    """
    This Class Represents one render instance
    Identifiers like, a, b, c, ... are initial value for the strange attractors
    """
    resolution: int | list[int]
    percentile: float | list[float]
    colors: NDArray | list[NDArray]
    n: int | list[int]

    def __post_init__(self):
        # attributes only available after render
        self.img_: Optional[NDArray] = None
        self.raw_: Optional[NDArray] = None
        self.collapsed: bool = False

    def set_static(self, resolution: bool, percentile: bool, colors: bool, n: bool) -> None:
        return

    def __len__(self) -> int:
        return -1

    @property
    def img(self) -> NDArray:
        if self.img_ is not None:
            return self.img_

        if self.raw is None:
            raise RuntimeError("Frame isn't rendered yet!")

        self.check_multiple()
        # self.colors: NDArray
        if isinstance(self.colors, list):
            raise Exception()
        self.img_ = apply_color(self.raw, self.colors)
        return self.img_

    def check_multiple(self):
        assert not isinstance(self.resolution, (list, np.ndarray))
        assert not isinstance(self.n, (list, np.ndarray))
        assert not isinstance(self.percentile, (list, np.ndarray))
        assert not isinstance(self.colors[0][0], (list, np.ndarray))
        return True

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
        self.check_multiple()
        if not only_raw and self.raw is not None:
            if isinstance(self.colors, list):
                raise Exception()
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
        assert not isinstance(self.resolution, list)
        non_zero = np.count_nonzero(self.raw)
        thresh = self.resolution ** 2 * 0.05
        self.collapsed = non_zero < thresh

# resolution: int
# percentile: float
# colors: NDArray
# n: int

@dataclass
class SimonFrame(Frame):
    a: float | list
    b: float | list

    def render(self, only_raw = False):
        from .attractor import simon

        if isinstance(self.a, list) or isinstance(self.b, list):
            raise ValueError("a or b are a list, when using lists for assignment call .toFrames() first")

        x, y = simon(self.a, self.b, self.n)
        self.raw = self.scatter_to_normalized(x, y)
        super().render(only_raw=only_raw)

    def set_static(self, resolution: bool, percentile: bool, colors: bool, n: bool) -> None:
        if self.resolution:
            # self.resolution = [self.resolution] * len(self)




    def toFrames(self) -> list[Frame]:

        return []

    def __len__(self) -> int:
        if isinstance(self.a, list):
            return len(self.a)
        if isinstance(self.b, list):
            return len(self.b)
        if isinstance(self.n, list):
            return len(self.n)
        if isinstance(self.percentile, list):
            return len(self.percentile)
        return 1




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

        self.check_multiple()
        x, y = clifford(self.a, self.b, self.c, self.d, self.n)
        self.raw = self.scatter_to_normalized(x, y)
        super().render(only_raw=only_raw)

