from .api import (
    Performance_Renderer,
    ColorMap,
    sinspace,
    cosspace,
    bpmspace,
    map_area,
    linspace,
)

from .frame import Frame, SimonFrame

from .attractor import render_frame
from .utils import apply_colormap

__version__ = "0.1.0"

__all__ = [
    "ColorMap",
    "Performance_Renderer",
    "sinspace",
    "cosspace",
    "bpmspace",
    "linspace",
    "map_area",
    "render_frame",
    "apply_colormap",
    "Frame",
    "SimonFrame"
]
