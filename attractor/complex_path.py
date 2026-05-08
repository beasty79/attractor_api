from dataclasses import dataclass
from .opts import Option
import numpy as np

@dataclass
class Point:
    frame: int
    value: float | int


@dataclass
class KeyframeInterpolator:
    opts: Option
    first_value: float
    last_value: float

    def __post_init__(self):
        self.points: dict[int, float] = {}
        self.points[0] = self.first_value
        self.points[self.opts.frames - 1] = self.last_value


    def add_keyframe(self, point: Point):
        assert 0 < point.frame <= self.opts.frames, "Points is out of bound [0 is already set]"
        self.points[point.frame] = point.value

    def to_array(self) -> np.typing.NDArray:
        spaces = []
        keys = sorted(self.points.keys())

        # interpolate using linsapce
        for i, (p1, p2) in enumerate(zip(keys, keys[1:])):
            v1, v2 = self.points[p1], self.points[p2]
            frame_diff = p2 - p1
            lin = np.linspace(v1, v2, frame_diff + 1)
            if i > 0:
                lin = lin[1:]  # remove overlap
            spaces.append(lin)

        # Merge arrays together
        length = sum([len(arr) for arr in spaces])
        empty_arr = np.zeros(shape=(length,), dtype=np.float32)
        i = 0
        for arr in spaces:
            for entry in arr:
                empty_arr[i] = entry
                i += 1

        return empty_arr


def resample_by_frames(points: np.ndarray, frames: int):
    """
    Resample a 2D polyline so it contains exactly `frames`
    equally spaced points along arc length.
    """

    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        raise ValueError("Need at least two points")

    # Compute cumulative arc length
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0.0)

    total_length = cumulative_length[-1]

    if total_length == 0:
        # All points identical
        return np.repeat(points[:1], frames, axis=0)

    # Evenly spaced target arc lengths
    target_distances = np.linspace(0, total_length, frames)

    # Interpolate x and y independently over arc length
    x = np.interp(target_distances, cumulative_length, points[:, 0])
    y = np.interp(target_distances, cumulative_length, points[:, 1])

    return np.column_stack((x, y))