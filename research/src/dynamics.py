from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import ast
import math

import numpy as np
import pandas as pd


EARTH_GRAVITY = 9.81
CONTROL_LIMITS = np.array(
    [
        [-math.pi / 12.0, math.pi / 12.0],  # roll
        [-math.pi, math.pi],                # pitch
        [-math.pi / 12.0, math.pi / 12.0],  # yaw
        [0.0, 12.0],                        # thrust
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


def parse_pg_array(value: str) -> np.ndarray:
    """Parse Postgres array text into a float numpy vector."""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    cleaned = str(value).replace("{", "[").replace("}", "]")
    return np.asarray(ast.literal_eval(cleaned), dtype=float)


def _load_csvs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectories = pd.read_csv(data_dir / "trajectories.csv", header=None, names=["trajectory_id", "created_at"])
    states = pd.read_csv(
        data_dir / "trajectory_states.csv",
        header=None,
        names=["state_id", "trajectory_id", "step", "state", "control", "created_at"],
    )
    scores = pd.read_csv(data_dir / "scores.csv", header=None, names=["score_id", "trajectory_id", "score"])
    return trajectories, states, scores


def load_training_samples(data_dir: str | Path) -> pd.DataFrame:
    """
    Return flat sample table:
    trajectory_id, step, score, x1..x6, u1..u4.
    """
    data_path = Path(data_dir)
    _, states, scores = _load_csvs(data_path)

    states = states.copy()
    states["step"] = states["step"].astype(int)
    states["x"] = states["state"].map(parse_pg_array)
    states["u"] = states["control"].map(parse_pg_array)

    x_cols = [f"x{i}" for i in range(1, 7)]
    u_cols = [f"u{i}" for i in range(1, 5)]

    x_frame = pd.DataFrame(np.vstack(states["x"].to_numpy()), columns=x_cols, index=states.index)
    u_frame = pd.DataFrame(np.vstack(states["u"].to_numpy()), columns=u_cols, index=states.index)

    table = pd.concat([states[["trajectory_id", "step"]], x_frame, u_frame], axis=1)
    table = table.merge(scores[["trajectory_id", "score"]], how="inner", on="trajectory_id")
    table = table.sort_values(["trajectory_id", "step"]).reset_index(drop=True)

    return table


def build_trajectory_bundle(samples: pd.DataFrame) -> Dict[int, Dict[str, np.ndarray | float]]:
    """
    Convert flat samples into per-trajectory arrays:
    bundle[trajectory_id] = {"X": (N,6), "U": (N,4), "score": float}
    """
    bundle: Dict[int, Dict[str, np.ndarray | float]] = {}
    x_cols = [f"x{i}" for i in range(1, 7)]
    u_cols = [f"u{i}" for i in range(1, 5)]

    for tid, group in samples.groupby("trajectory_id", sort=True):
        g = group.sort_values("step")
        bundle[int(tid)] = {
            "X": g[x_cols].to_numpy(dtype=float),
            "U": g[u_cols].to_numpy(dtype=float),
            "score": float(g["score"].iloc[0]),
        }
    return bundle


def model(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """
    Quadrotor translational dynamics.
    state: [x1, x2, x3, v1, v2, v3]
    control: [phi, theta, psi, thrust]
    """
    phi, theta, psi, thrust = control
    return np.array(
        [
            state[3],
            state[4],
            state[5],
            (math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)) * thrust,
            (math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)) * thrust,
            thrust * math.cos(theta) * math.cos(phi) - EARTH_GRAVITY,
        ],
        dtype=float,
    )


def rk4_step(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Single fixed-step classical RK4 integration step."""
    k1 = model(state, control)
    k2 = model(state + 0.5 * dt * k1, control)
    k3 = model(state + 0.5 * dt * k2, control)
    k4 = model(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(initial_state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    """Integrate trajectory for full control sequence. Returns (len(controls)+1, 6)."""
    states = np.zeros((len(controls) + 1, len(initial_state)), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for k, u in enumerate(controls):
        states[k + 1] = rk4_step(states[k], np.asarray(u, dtype=float), dt)
    return states


def _euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    return float(np.linalg.norm(aa - bb))


def _heaviside(value: float) -> float:
    return 1.0 if value >= 0 else 0.0


def bolza_cost(initial_state: np.ndarray, controls: np.ndarray, cfg: CostConfig) -> float:
    """
    Solver-consistent functional:
    dt * N + terminal + cylinder penalties + window penalties.
    """
    states = rollout(initial_state, controls, cfg.dt)

    cylinder_penalty = 0.0
    for st in states:
        x1, x3 = st[0], st[2]
        for cyl in cfg.cylinders:
            signed_distance = cyl.radius - math.hypot(x1 - cyl.x, x3 - cyl.z)
            cylinder_penalty += cfg.cylinder_penalty * _heaviside(signed_distance)

    window_penalty = 0.0
    for wnd in cfg.windows:
        distances = [math.hypot(st[0] - wnd.x, st[2] - wnd.z) - wnd.radius for st in states]
        min_distance = min(distances)
        if min_distance > 0:
            window_penalty += cfg.window_penalty * (min_distance ** 2)

    terminal_penalty = cfg.terminal_penalty * _euclidean(states[cfg.num_intervals], cfg.terminal_state)
    return cfg.dt * cfg.num_intervals + terminal_penalty + cylinder_penalty + window_penalty


def clamp_controls(controls: np.ndarray) -> np.ndarray:
    """Clip controls to physically valid ranges used by solver."""
    lower = CONTROL_LIMITS[:, 0]
    upper = CONTROL_LIMITS[:, 1]
    return np.clip(np.asarray(controls, dtype=float), lower, upper)
