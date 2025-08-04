from numpy.typing import NDArray
from typing import Optional
from numpy import ndarray
from numba import njit
import numpy as np
import math


@njit
def iterate(a: float, b: float, n: int) -> tuple[ndarray, ndarray]:
    """calculates the simon attractor

    Args:
        a (float): _description_
        b (float): _description_
        n (int): _description_

    Returns:
        tuple[ndarray, ndarray]: arr_x, arr_y
        # arr_x[i], arr_y[i] => x, y at iteration i
    """
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
) -> NDArray:
    """
    Computes the Simon Attractor and returns either a normalized histogram or a color-mapped image.

    Args:
        resolution (int): Resolution of the output grid (res x res). Runtime ~ O(n^2).
        a (float): Parameter 'a' for the Simon Attractor.
        b (float): Parameter 'b' for the Simon Attractor.
        n (int): Number of iterations. Higher values yield smoother output; usually n > 1_000_000.
        percentile (float): Clipping percentile for histogram normalization (e.g., 95-99.9). u dont see much without

    Returns:
        NDArray[np.float32] if raw=True, otherwise NDArray[np.uint8] (RGB image).
    """
    # calculate
    x_raw, y_raw = iterate(a, b, n)
    points_per_pixel = np.histogram2d(x_raw, y_raw, bins=resolution)[0]

    # clip outliers
    max_value = np.percentile(points_per_pixel, percentile)
    max_value = max_value if np.isfinite(max_value) and max_value > 0 else 1.0
    points_per_pixel = np.clip(points_per_pixel, 0, max_value)

    # normalize to [0,1]
    h_normalized = (points_per_pixel / np.max(points_per_pixel)).astype(np.float32)
    return h_normalized