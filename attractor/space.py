from numpy.typing import NDArray
import numpy as np
import os

from .colormap import ColorMap
from .render_class import Performance_Renderer


def linspace(lower: float, upper: float, n: int):
    """
    [equals np.linspace]
    Parameters: 
    - lower (float): The minimum value.
    - upper (float): The maximum value.
    - n (int): Number of points in the output array.

    Returns:
    - np.ndarray: An array of values from between lower and upper evenly spaced
    """
    return np.linspace(lower, upper, n)

def bpmspace(lower: float, upper: float, n: int, bpm: int, fps: int):
    """
    Parameters:
    - lower (float): The minimum value.
    - upper (float): The maximum value.
    - n (int): Number of points in the output array.
    - p (float): Number of sine periods to span across the interval.

    Returns:
    - np.ndarray: An array of values shaped by a sine wave between lower and upper.
    """
    total_time = n / fps
    minutes = total_time / 60
    periods_needed = minutes * bpm
    return sinspace(lower, upper, n, p=periods_needed)


def sinspace(lower: float, upper: float, n: int, p: float = 1.0):
    """
    Parameters:
    - lower (float): The minimum value.
    - upper (float): The maximum value.
    - n (int): Number of points in the output array.
    - p (float): Number of sine periods to span across the interval.

    Returns:
    - np.ndarray: An array of values shaped by a sine wave between lower and upper.
    """
    phase = np.linspace(0, 2 * np.pi * p, n)
    sin_wave = (np.sin(phase) + 1) / 2
    return lower + (upper - lower) * sin_wave


def cosspace(lower: float, upper: float, n: int, p: float = 1.0):
    """
    Parameters:
    - lower (float): The minimum value.
    - upper (float): The maximum value.
    - n (int): Number of points in the output array.
    - p (float): Number of sine periods to span across the interval.

    Returns:
    - np.ndarray: An array of values shaped by a cos wave between lower and upper.
    """
    phase = np.linspace(0, 2 * np.pi * p, n)
    cos_wave = (np.cos(phase) + 1) / 2
    return lower + (upper - lower) * cos_wave


def map_area(a: NDArray, b: NDArray, fname: str, colormap: ColorMap, skip_empty: bool = True, fps: int = 15, n=1_000_000, percentile=99, resolution=1000):
    """Generates a animation over a whole area. a, b are the axis (uses np.meshgrid)"""
    assert len(a) == len(b), "a & b dont match in length"
    A, B = np.meshgrid(a, b)

    for i in range(A.shape[0]):
        if i % 2 == 1:
            A[i] = A[i][::-1]
    A = A.flatten()

    # A = A.ravel()
    B = B.ravel()
    process = Performance_Renderer(
        a=A,
        b=B,
        colormap=colormap,
        frames=len(A),
        fps=fps,
        percentile=percentile,
        n=n,
        resolution=resolution
    )
    process.set_static("a", False)
    process.set_static("b", False)
    process.start_render_process(fname, verbose_image=True, threads=4, chunksize=8, skip_empty_frames=skip_empty)


def from_generic(path: str, colormap: ColorMap):
    # new filepath with colormap suffix
    suffix = "_inv" if colormap.inverted else ""
    fname = os.path.basename(path).replace(".mp4", "")
    fname = f"{fname}_{colormap.name}{suffix}.mp4"
    new_path = os.path.join(os.path.dirname(path), fname)
    print("export_path: ", new_path)


    # writer = VideoFileWriter()
    ...