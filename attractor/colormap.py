from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


class ColorMap:
    def __init__(self, name: str, inverted: bool = False) -> None:
        self.name = name
        self.color = self.get_colors_array(name)
        self.inverted = inverted

    def set_inverted(self, state: bool):
        self.inverted = state

    def get_colors_array(self, cmap: str) -> NDArray:
        color_map = plt.get_cmap(cmap)
        linear = np.linspace(0, 1, 256)
        return color_map(linear)

    def greyscale(self, inverted: bool = False) -> NDArray:
        linear = np.linspace(1.0, 0.0, 256)
        rgb = np.stack([linear, linear, linear], axis=1)
        rgba = np.concatenate([rgb, np.ones((256, 1))], axis=1)
        return rgba if not inverted else rgba[::-1]

    def get(self) -> NDArray:
        return self.color[::-1] if self.inverted else self.color

    def __repr__(self) -> str:
        return f"Colormap['{self.name}', {self.inverted=}]"

    @staticmethod
    def colormaps():
        return list(plt.colormaps)