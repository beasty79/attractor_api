from numpy.typing import NDArray
from numba import njit, prange
from typing import Optional
from functools import wraps
# from math import sin, cos
# from math import sin, cos
import math
from numpy import ndarray
from typing import Any
import numpy as np
import timeit


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@njit
def iterate(a: float, b: float, n: int) -> tuple[ndarray, ndarray]:
    x, y = a, b

    arr_x = np.zeros(shape=(n,), dtype=np.float64)
    arr_y = np.zeros(shape=(n,), dtype=np.float64)
    for i in range(n):
        x_new = math.sin(x**2 - y**2 + a)
        y_new = math.cos(2 * x * y + b)

        x, y = x_new, y_new
        arr_x[i] = x
        arr_y[i] = y

    return arr_x, arr_y


def render_frame(
    resolution: int,
    a: float,
    b: float,
    n: int,
    percentile: float,
    colors: Optional[NDArray[np.float32]] = None,
    raw: bool = True
) -> NDArray:
    """
    Computes the Simon Attractor and returns either a normalized histogram or a color-mapped image.

    Args:
        resolution (int): Resolution of the output grid (res x res). Runtime ~ O(n^2).
        a (float): Parameter 'a' for the Simon Attractor.
        b (float): Parameter 'b' for the Simon Attractor.
        n (int): Number of iterations. Higher values yield smoother output; usually n > 1_000_000.
        percentile (float): Clipping percentile for histogram normalization (e.g., 95-99.9).
        colors (NDArray[np.float32] | None): Colormap values in range [0, 1]. Required if raw is False.
        raw (bool): If True, returns raw normalized histogram. If False, returns color-mapped image.

    Returns:
        NDArray[np.float32] if raw=True, otherwise NDArray[np.uint8] (RGB image).
    """
    x_raw, y_raw = iterate(a, b, n)
    histogram, _, _ = np.histogram2d(x_raw, y_raw, bins=resolution)

    clip_max = np.percentile(histogram, percentile)
    if clip_max == 0 or np.isnan(clip_max):
        clip_max = 1.0

    h_normalized = np.clip(histogram / clip_max, 0, 1).astype(np.float32)

    if raw:
        return h_normalized

    if colors is None:
        raise ValueError("`colors` must be provided when raw=False.")

    values = (h_normalized * 255).astype(int)
    img = (colors[values] * 255).astype(np.uint8)
    return img


def render(colors: np.typing.NDArray[np.float32], resolution: int, a: float, b: float, n: int, percentile: float):
    """This Calcultes the image using a, b be the inital value for the iterations the Simon Attractor

    Args:
        colors (np.typing.NDArray[np.float32]): colormap_values in range [0;1]
        resolution (int): how many pixels are generated (resxres) scales O(n**2) so be carefull
        a_initial (float):
        b_initial (float):
        n (int): higher values will improve the output scales O(n)
        percentile (float): This makes sure to the colormaps range is properly utilized usually between (95-99.9)

    Returns:
        image as a numpy array in shape
    """
    x_raw, y_raw = iterate(a, b, n)
    histogramm, _, _ = np.histogram2d(x_raw, y_raw, bins=resolution)
    if percentile is None:
        percentile = 95
    clip_max = np.percentile(histogramm, percentile)
    if clip_max == 0 or np.isnan(clip_max):
        clip_max = 1
    h_normalized = histogramm / clip_max
    h_normalized = np.clip(h_normalized, 0, 1)
    values = (h_normalized * 255).astype(int)
    img = (colors[values] * 255).astype(np.uint8)
    return img


def render_raw(resolution: int, a: float, b: float, n: int, percentile: float) -> NDArray[np.float32]:
    """
    same as render but it doesnt apply the colormaps yet
    Computes a normalized 2D histogram of the Simon Attractor iterations.
    Args:
        resolution (int): The resolution of the output grid (res x res). Be cautious: runtime ~ O(n^2).
        a (float): Parameter 'a' for the Simon Attractor.
        b (float): Parameter 'b' for the Simon Attractor.
        n (int): Number of iterations. Higher values yield smoother output.
        percentile (float | None): Upper clipping percentile for normalization. Typical values: 95–99.9.

    Returns:
        NDArray[np.float32]: A normalized (0.0–1.0) 2D array representing the histogram density.
    """
    x_raw, y_raw = iterate(a, b, n)
    histogram, _, _ = np.histogram2d(x_raw, y_raw, bins=resolution)

    clip_max = np.percentile(histogram, percentile)
    if clip_max == 0 or np.isnan(clip_max):
        clip_max = 1.0
    h_normalized = histogram / clip_max
    h_normalized = np.clip(h_normalized, 0, 1)
    return h_normalized.astype(np.float32)


def to_img(h_normalized: NDArray[np.float32], colors: NDArray[np.float32]) -> NDArray[np.uint8]:
    values = (h_normalized * 255).astype(int)
    values = np.clip(values, 0, 255)
    img = (colors[values] * 255).astype(np.uint8)
    return img