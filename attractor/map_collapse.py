from dataclasses import dataclass
from typing import Optional
from matplotlib.backend_bases import MouseEvent
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import multiprocessing
from time import sleep
import numpy as np
import pickle
import os

from .terminal import TerminalCounter
from .frame import Frame, SimonFrame
from .utils import apply_colormap
from .colormap import ColorMap
from .config import Config

@dataclass
class CollapseContainer:
    img: np.ndarray
    a: np.ndarray
    b: np.ndarray

class CollapseMap:
    _data: Optional[CollapseContainer] = None

    def __init__(self, a_bounds: tuple[float, float], b_bounds: tuple[float, float], delta: float, data=None) -> None:
        self._a_bounds = a_bounds
        self._b_bounds = b_bounds
        self._delta = delta
        self._data = data

    def save(self, filename: str):
        """Save the CollapseMap object as a pickle file."""
        assert self._data is not None, "Render before saving it!"

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self._data, f)

    @classmethod
    def load(cls, filename: str) -> "CollapseMap":
        """Load a data object from a pickle file."""
        if not os.path.exists(filename):
            filename = filename + ".pkl"
        if not os.path.exists(filename):
            raise FileNotFoundError()

        with open(filename, 'rb') as f:
            data: CollapseContainer = pickle.load(f)

        delta = abs(data.a[0] - data.a[1])
        a_bounds = data.a[0], data.a[-1]
        b_bounds = data.b[0], data.b[-1]

        return cls(a_bounds, b_bounds, delta, data)

    def show(self):
        """Show the map with matplotlib (only works afer render())"""
        assert self._data is not None, "render first"
        show_collapse_map(self._data)

    def render(self):
        """Render all frames with current paramters"""
        self._data = render_collapse_map(
            a_bounds=self._a_bounds,
            b_bounds=self._b_bounds,
            delta=self._delta
        )



def _render_frames_collapse(frames: list[tuple[Frame, tuple[int, int]]], shape: tuple[int, int], use_counter: bool = True):
    if use_counter:
        counter = TerminalCounter(len(frames))
        counter.start()

    collapseMap = np.zeros(dtype=np.float16, shape=shape)

    return_value: tuple[int, tuple]
    with multiprocessing.Pool(Config().threads) as pool:
        for return_value in pool.imap(_render_frame_collapse_wrapper, frames, chunksize=Config().chunksize):
            is_collapsed: int = return_value[0]
            x, y = return_value[1]

            if use_counter and counter is not None:
                counter.count_up()

            collapseMap[y, x] = is_collapsed
    return (collapseMap - collapseMap.min()) / (collapseMap.max() - collapseMap.min())


def _render_frame_collapse_wrapper(args: tuple[Frame, tuple[int, int]]):
    frame, (x, y) = args
    frame.render(only_raw=True)
    is_collapsed = frame.is_collapsed()
    frame.clear()
    return is_collapsed, (x, y)


def render_collapse_map(a_bounds: tuple[float, float], b_bounds: tuple[float, float], delta: float=0.01):
    a_diff = abs(a_bounds[0] - a_bounds[1])
    b_diff = abs(b_bounds[0] - b_bounds[1])
    na: int = round(a_diff / delta)
    nb: int = round(b_diff / delta)
    A = np.linspace(a_bounds[0], a_bounds[1], na)
    B = np.linspace(b_bounds[0], b_bounds[1], nb)

    print(f"{len(A)}x{len(B)}   frames: {len(A)*len(B)}")
    sleep(1.5)

    frames: list[tuple[Frame, tuple]] = []
    colomap = ColorMap("viridis")
    for x, a in enumerate(A):
        for y, b in enumerate(B):
            frame = SimonFrame(
                a=a,
                b=b,
                colors=colomap,
                n=3_000,
                resolution=150
            )
            frames.append((frame, (x, y)))

    collapseMap = _render_frames_collapse(frames, shape=(len(B), len(A)))
    img = apply_colormap(collapseMap, ColorMap("viridis"))
    return CollapseContainer(img, A, B)

def show_collapse_map(map: CollapseContainer):
    from .frame import SimonFrame
    currentPath: list[tuple[float, float]] = []
    # Create two subplots side by side
    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.tight_layout()

    # First image
    im1 = ax1.imshow(map.img, origin='lower')
    ax1.set_title("Original Image")

    # Text box for coordinates on hover
    coord_text = ax1.text(0.02, 0.98, '', color='white', transform=ax1.transAxes, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, pad=2))

    # Placeholder for second image
    im2 = ax2.imshow(np.zeros_like(map.img), origin='lower')
    ax2.set_title("Rendered Frame")

    # Mouse hover event for first image
    def on_mouse_move(event: MouseEvent):
        if event.inaxes != ax1:
            return

        assert event.xdata
        assert event.ydata
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        try:
            x_val = map.a[x]
            y_val = map.b[y]
            coord_text.set_text(f"a: {x_val:.4f}, b: {y_val:.4f}")
            fig.canvas.draw_idle()
        except IndexError:
            return

        # Render SimonFrame
        frame = SimonFrame(
            a=x_val,
            b=y_val,
            n=500_000,
            resolution=500,
            colors=ColorMap("viridis", True)
        )
        frame.render()
        im2.set_data(frame.img)
        im2.set_extent((0, frame.img.shape[1], 0, frame.img.shape[0]))
        ax2.set_aspect('auto')
        ax2.set_title(f"Rendered Frame (a={x_val:.4f}, b={y_val:.4f})")
        ax2.set_xlim(0, frame.img.shape[1])
        ax2.set_ylim(0, frame.img.shape[0])
        fig.canvas.draw_idle()

    # Click event to render SimonFrame and update second panel
    def on_click(event: MouseEvent):
        toolbar = plt.get_current_fig_manager()
        if toolbar is not None:
            toolbar = toolbar.toolbar
            if toolbar.mode != '': # type: ignore
                return

        if event.inaxes == ax1:
            assert event.xdata
            assert event.ydata
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            try:
                x_val = map.a[x]
                y_val = map.b[y]
            except IndexError:
                return

            map.img[y, x] = [255, 0, 0, 255]
            im1.set_data(map.img)
            fig.canvas.draw_idle()
            currentPath.append((round(float(x_val), 4), round(float(y_val), 4)))

            os.system("cls")
            for i, (x, y) in enumerate(currentPath):
                print(f"{i:02d}: {x:2f}, {y:2f}")
            print(currentPath)


    def on_key(event: MouseEvent):
        assert event.xdata
        assert event.ydata
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        try:
            x_val = map.a[x]
            y_val = map.b[y]
        except IndexError:
            return

        if event.key == "enter":
            frame = SimonFrame(
                a=x_val,
                b=y_val,
                n=3_000_000,
                resolution=1000,
                colors=ColorMap("viridis")
            )
            frame.render()
            frame.show()

    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move) # type: ignore
    fig.canvas.mpl_connect('button_press_event', on_click) # type: ignore
    fig.canvas.mpl_connect('key_press_event', on_key) # type: ignore

    # Hide axes
    ax1.axis('off')
    ax2.axis('off')

    plt.show()