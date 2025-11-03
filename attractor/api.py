from numpy.typing import NDArray
import matplotlib.pyplot as plt
import multiprocessing
from typing import Any
from time import time
import numpy as np
import os
from typing import Optional
from functools import partial


# internal
from .VideoWriter import VideoFileWriter
from .terminal import TerminalCounter
from .utils import promt
from .view import play_video
from .utils import render_frame
from .frame import Frame, SimonFrame

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


class Performance_Renderer:
    """This is an api wrapper class for rendering simon attractors"""
    def __init__(
        self,
        a: float | NDArray,
        b: float | NDArray,
        colormap: ColorMap,
        frames: int,
        fps: int = 30,
        n: int | list[int] = 1_000_000,
        resolution: int | list[int] = 1000,
        percentile: float | NDArray = 99
    ) -> None:
        self.a = a
        self.b = b
        self.n = n
        self.resolution = resolution
        self.percentile = percentile
        self.frames = frames
        self.value = {
            'a': a,
            'b': b,
            'n': n,
            'resolution': resolution,
            'percentile': percentile
        }
        self.static = {
            'a': True,
            'b': True,
            'n': True,
            'resolution': True,
            'percentile': True
        }
        self.fps = fps
        self.writer = None
        self.color = None
        self.counter: TerminalCounter | None = None
        self.colormap: ColorMap = colormap
        self.hook: None = None
        self._demo = False

    def set_static(self, argument: Any, is_static: bool):
        """
        argument: {'a', 'b', 'n', 'resolution', 'percentile'}
        """
        if argument not in self.static:
            raise ValueError(f"arg: {argument} is invalid, should be: ['a', 'b', 'n', 'resolution', 'percentile']")
        self.static[argument] = is_static

    def addHook(self, signal):
        self.hook = signal

    def get_iter_value(self, arg: str) -> list[Any]:
        if arg not in self.static:
            raise ValueError("arg not in static")
        is_static: bool = self.static[arg]

        if is_static:
            return [self.value[arg]] * self.frames
        else:
            return self.value[arg]

    def get_unique_fname(self, fname: str) -> str:
        base_path = os.path.dirname(fname)
        full_name = os.path.basename(fname)
        name_only, ext = os.path.splitext(full_name)

        new_name = fname
        i_ = 0
        while os.path.exists(new_name):
            i_ += 1
            name_comp = f"{name_only}({i_}){ext}"
            new_name = os.path.join(base_path, name_comp)
        return new_name

    def show_demo(self, 
                  nth_frame: int = 10, 
                  real_time: bool = False, 
                  resolution: int = 750, 
                  iterations: int = 500_000,
                  fps: Optional[int] = None
        ):
        self._demo_var = nth_frame
        if os.path.exists("./tmp.mp4"):
            os.remove("./tmp.mp4")

        # cache class vars and change them
        fps_cache = self.fps
        self._demo = True
        self._demo_res = resolution
        self._demo_iterations = iterations
        self.fps = round(self.fps / self._demo_var)

        # render demo video
        self.start_render_process("./tmp.mp4", verbose_image=True, bypass_confirm=True)

        fps_ = fps if fps is not None else 10
        play_video("./tmp.mp4", self.fps if real_time else fps_)

        # rechange variables
        self.fps = fps_cache
        self._demo = False

    def get_frames(self, res, percentile, color, n, a, b) -> list[SimonFrame]:
        """Helper function"""
        return [
            SimonFrame(
                resolution=res[i],
                percentile=percentile[i],
                colors=color[i],
                n=n[i],
                a=a[i],
                b=b[i]
            )
            for i in range(len(res))
        ]

    def start_render_process(
            self,
            fname: str,
            verbose_image: bool    = False,
            threads: Optional[int] = 4,
            chunksize: int         = 4,
            skip_empty_frames: bool= True,
            bypass_confirm: bool   = False,
            save_as_generic: bool  = False,
            use_counter: bool      = True
        ):
        """starts the render Process

        Args:
            fname (str): filename / filepath
            verbose_image (bool, optional): adds a small text with the parameter per frmae. Defaults to False.
            threads (Optional[int], optional): cpu cores to use. Defaults to 4.
            chunksize (int, optional): the higher the chunksize the more efficient but it needs more memory. Defaults to 4.
            skip_empty_frames (bool, optional): skips frames wheree the fractal collapses. Defaults to True.
            bypass_confirm (bool, optional): bypass the terminal confirmation. Defaults to False.
            save_as_generic (bool, optional): saves as grey space image so it can be loaded and colored again without the need to render it again. Defaults to False.
        """
        res: list[int] = self.get_iter_value("resolution")
        a: list[int] = self.get_iter_value("a")
        b: list[int] = self.get_iter_value("b")
        n: list[int] = self.get_iter_value("n")
        percentile: list[int] = self.get_iter_value("percentile")
        
        if save_as_generic:
            self.color = self.colormap.greyscale()
        else:
            self.color = self.colormap.get()

        col = [self.color] * len(a)

        # checks and promting
        assert all(len(lst) == len(res) for lst in [a, b, n, percentile, col]), "Mismatched lengths in input lists"

        # Create Frame dataclass for every frame
        if self._demo:
            frames = self.get_frames([self._demo_res] * len(n), percentile, col, [self._demo_iterations] * len(n), a, b)
            frames = frames[::self._demo_var]
        else:
            frames = self.get_frames(res, percentile, col, n, a, b)

        if not bypass_confirm:
            promt(len(frames), self.fps)
        

        if not fname.lower().endswith('.mp4'):
            fname += '.mp4'

        # File Writer
        self.writer = VideoFileWriter(
            filename=self.get_unique_fname(fname),
            fps=self.fps
        )

        # Terminal Feedback
        tstart = time()

        if use_counter:
            self.counter = TerminalCounter(len(frames))
            if self.hook is None:
                self.counter.start()

        # render_func = lambda frame: render_frame(frame, only_raw=save_as_generic)

        # func = render_frame if not save_as_generic else render_frame_raw
        # func = partial(render_frame, only_raw=save_as_generic)
        # Multiproccessing
        try:
            with multiprocessing.Pool(threads) as pool:
                frame: Frame
                for i, frame in enumerate(pool.imap(render_frame, frames, chunksize=chunksize)):

                    # Either emit a Signal (pyqt6 hook) or show the progress-bar in the Terminal
                    if self.hook is not None:
                        self.hook.emit(i)
                    else:
                        if self.counter is not None:
                            self.counter.count_up()

                    # filter if frame is collapsed
                    if frame.collapsed and skip_empty_frames:
                        continue

                    # write a, b
                    if verbose_image:
                        self.writer.add_frame(frame.img, a=a[i], b=b[i])
                    else:
                        self.writer.add_frame(frame.img)
        except Exception as e:
            raise e

        # Process Finished
        total = time() - tstart
        min_ = int(total // 60)
        sec_ = int(total % 60)
        print(f"Finished render process in {min_:02d}:{sec_:02d}")
        print(f"Average: {self.frames / total:.2f} fps")
        self.writer.save()

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
