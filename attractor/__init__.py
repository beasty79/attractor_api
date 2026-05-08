from .space import (
    Performance_Renderer,
    ColorMap,
    sinspace,
    cosspace,
    bpmspace,
    map_area,
    linspace,
    sawspace,
    squarespace,
    Waveform
)
from .config import Config
from .frame import Frame, SimonFrame
from .utils import apply_colormap
from .opts import Option
from .complex_path import KeyframeInterpolator, Point, resample_by_frames
from .generic import color_generic
from .map_collapse import render_collapse_map, CollapseMap
from .attractor import simon
from .alphaWriter import AlphaChannel

__version__ = "0.1.0"

__all__ = [
    "simon",
    "AlphaChannel",
    "resample_by_frames",
    "Performance_Renderer",
    "KeyframeInterpolator",
    "apply_colormap",
    "color_generic",
    "render_collapse_map",
    "squarespace",
    "SimonFrame",
    "ColorMap",
    "sawspace",
    "sinspace",
    "cosspace",
    "bpmspace",
    "linspace",
    "map_area",
    "Option",
    "Frame",
    "Point",
    "Waveform",
    "Config",
    "CollapseMap"
]
