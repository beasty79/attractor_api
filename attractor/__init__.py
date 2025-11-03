from .space import (
    Performance_Renderer,
    ColorMap,
    sinspace,
    cosspace,
    bpmspace,
    map_area,
    linspace,
)
from .frame import Frame, SimonFrame
from .utils import apply_colormap
from .opts import Option

__version__ = "0.1.0"

__all__ = [
    "Performance_Renderer",
    "apply_colormap",
    "SimonFrame",
    "ColorMap",
    "sinspace",
    "cosspace",
    "bpmspace",
    "linspace",
    "map_area",
    "Option",
    "Frame"
]
