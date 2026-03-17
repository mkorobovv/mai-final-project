from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


EARTH_GRAVITY: float = 9.81

CONTROL_LIMITS = np.array(
    [
        [-math.pi / 12.0, math.pi / 12.0],  # крен (roll)
        [-math.pi,        math.pi],          # тангаж (pitch)
        [-math.pi / 12.0, math.pi / 12.0],  # рыскание (yaw)
        [0.0,             12.0],             # тяга (thrust)
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Cylinder:
    x: float
    z: float
    radius: float


@dataclass(frozen=True)
class Window:
    x: float
    z: float
    radius: float


@dataclass(frozen=True)
class CostConfig:
    num_intervals: int = 15
    horizon: float = 5.6
    terminal_state: Tuple[float, ...] = (5.0, 5.0, 10.0, 0.0, 0.0, 0.0)
    terminal_penalty: float = 0.9
    cylinders: Tuple[Cylinder, ...] = (
        Cylinder(x=1.5, z=2.5, radius=2.5),
        Cylinder(x=6.5, z=7.5, radius=2.5),
    )
    cylinder_penalty: float = 0.9
    windows: Tuple[Window, ...] = (Window(x=4.0, z=5.0, radius=0.5),)
    window_penalty: float = 1.6

    @property
    def dt(self) -> float:
        return self.horizon / float(self.num_intervals)
