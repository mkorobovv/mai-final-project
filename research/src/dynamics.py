from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import ast
import math

import numpy as np
from numba import njit
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


def select_top_k_trajectories(
    samples: pd.DataFrame,
    bundle: Mapping[int, Mapping[str, np.ndarray | float]],
    top_k: int,
) -> Tuple[List[int], Dict[int, Dict[str, np.ndarray | float]], pd.DataFrame]:
    """Select top-k trajectories with the smallest score."""
    score_table = (
        samples[["trajectory_id", "score"]]
        .drop_duplicates()
        .sort_values("score")
        .reset_index(drop=True)
    )
    selected_ids = score_table.head(int(top_k))["trajectory_id"].astype(int).tolist()
    selected = {tid: dict(bundle[tid]) for tid in selected_ids}
    return selected_ids, selected, score_table


def split_trajectory_ids(
    trajectory_ids: Sequence[int],
    train_ratio: float = 0.8,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle trajectory IDs and split into train/test sets."""
    ids = np.asarray(trajectory_ids, dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    train_cut = int(float(train_ratio) * len(shuffled))
    return shuffled[:train_cut], shuffled[train_cut:]


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

def anneal(initial_state, start_controls, cfg, n_iter=200000, step_size=0.01, t0=200.0):
    initial_state = np.asarray(initial_state, dtype=np.float64)
    start_controls = np.asarray(start_controls, dtype=np.float64)

    terminal_state = np.asarray(cfg.terminal_state, dtype=np.float64)
    cylinders = np.asarray([(c.x, c.z, c.radius) for c in cfg.cylinders], dtype=np.float64)
    windows = np.asarray([(w.x, w.z, w.radius) for w in cfg.windows], dtype=np.float64)

    return anneal_nb(
        initial_state=initial_state,
        start_controls=start_controls,
        dt=cfg.dt,
        num_intervals=cfg.num_intervals,
        terminal_state=terminal_state,
        terminal_penalty_weight=cfg.terminal_penalty,
        cylinders=cylinders,
        cylinder_penalty_weight=cfg.cylinder_penalty,
        windows=windows,
        window_penalty_weight=cfg.window_penalty,
        n_iter=n_iter,
        step_size=step_size,
        t0=t0,
    )

def clamp_controls(controls: np.ndarray) -> np.ndarray:
    """Clip controls to physically valid ranges used by solver."""
    lower = CONTROL_LIMITS[:, 0]
    upper = CONTROL_LIMITS[:, 1]
    return np.clip(np.asarray(controls, dtype=float), lower, upper)


class QuadraticBundleController:
    """Per-step ridge-regression controller on quadratic state features."""

    def __init__(self, m_features: int = 6, ridge_lambda: float = 1e-3) -> None:
        self.m = int(m_features)
        self.ridge_lambda = float(ridge_lambda)
        self.models: Dict[int, np.ndarray] = {}
        self.control_dim = CONTROL_LIMITS.shape[0]

    def _basis(self, x_batch: np.ndarray) -> np.ndarray:
        x = np.asarray(x_batch[:, : self.m], dtype=float)
        quad_terms = []
        for i in range(self.m):
            for j in range(i, self.m):
                term = x[:, i] * x[:, j]
                if i == j:
                    term = 0.5 * term
                quad_terms.append(term)
        return np.column_stack(quad_terms + [x, np.ones(len(x), dtype=float)])

    def fit(
        self,
        bundle_dict: Mapping[int, Mapping[str, np.ndarray | float]],
        trajectory_ids: Sequence[int],
        n_steps: int,
    ) -> "QuadraticBundleController":
        self.models = {}
        for step in range(1, int(n_steps) + 1):
            x_rows: List[np.ndarray] = []
            u_rows: List[np.ndarray] = []
            for tid in trajectory_ids:
                row = bundle_dict[int(tid)]
                x_arr = np.asarray(row["X"], dtype=float)
                u_arr = np.asarray(row["U"], dtype=float)
                if step <= len(x_arr) and step <= len(u_arr):
                    x_rows.append(x_arr[step - 1])
                    u_rows.append(u_arr[step - 1])
            if not x_rows:
                continue

            xs = np.vstack(x_rows)
            us = np.vstack(u_rows)
            self.control_dim = us.shape[1]
            g = self._basis(xs)
            reg = self.ridge_lambda * np.eye(g.shape[1], dtype=float)
            self.models[step] = np.linalg.solve(g.T @ g + reg, g.T @ us)
        return self

    def predict(self, x_state: np.ndarray, step: int) -> np.ndarray:
        coeff = self.models.get(int(step))
        if coeff is None:
            return np.zeros(self.control_dim, dtype=float)

        x = np.asarray(x_state[: self.m], dtype=float)
        basis = []
        for i in range(self.m):
            for j in range(i, self.m):
                basis.append(0.5 * x[i] * x[i] if i == j else x[i] * x[j])
        basis.extend(x.tolist())
        basis.append(1.0)
        u = np.asarray(basis, dtype=float) @ coeff
        return clamp_controls(u)


def train_quadratic_controllers(
    bundle_dict: Mapping[int, Mapping[str, np.ndarray | float]],
    train_ids: Sequence[int],
    n_steps: int,
    feature_dims: Sequence[int],
    ridge_lambda: float = 2e-3,
) -> Dict[int, QuadraticBundleController]:
    """Train multiple quadratic controllers keyed by number of state features."""
    controllers: Dict[int, QuadraticBundleController] = {}
    for m in feature_dims:
        ctrl = QuadraticBundleController(m_features=int(m), ridge_lambda=ridge_lambda)
        ctrl.fit(bundle_dict, train_ids, n_steps)
        controllers[int(m)] = ctrl
    return controllers


def evaluate_pointwise_rmse(
    bundle_dict: Mapping[int, Mapping[str, np.ndarray | float]],
    test_ids: Sequence[int],
    controller: QuadraticBundleController,
    n_steps: int,
) -> Tuple[np.ndarray, float]:
    """Evaluate control RMSE on held-out trajectory steps."""
    errors = []
    for tid in test_ids:
        row = bundle_dict[int(tid)]
        x_arr = np.asarray(row["X"], dtype=float)
        u_arr = np.asarray(row["U"], dtype=float)
        upper = min(len(x_arr), len(u_arr), int(n_steps))
        for step in range(1, upper + 1):
            u_hat = controller.predict(x_arr[step - 1], step)
            errors.append((u_hat - u_arr[step - 1]) ** 2)

    if not errors:
        return np.zeros(controller.control_dim, dtype=float), 0.0

    err = np.asarray(errors, dtype=float)
    rmse_per_control = np.sqrt(err.mean(axis=0))
    global_rmse = float(np.sqrt(err.mean()))
    return rmse_per_control, global_rmse


def synthesize_with_controller(
    initial_state: np.ndarray,
    controller: QuadraticBundleController,
    cfg: CostConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Roll out closed-loop dynamics and compute Bolza score."""
    x = np.asarray(initial_state, dtype=float)
    controls = []
    for step in range(1, cfg.num_intervals + 1):
        u = controller.predict(x, step)
        controls.append(u)
        x = rollout(x, np.asarray([u]), cfg.dt)[-1]

    controls_arr = np.asarray(controls, dtype=float)
    states = rollout(initial_state, controls_arr, cfg.dt)
    score = bolza_cost(initial_state, controls_arr, cfg)
    return states, controls_arr, score


def evaluate_closed_loop(
    bundle_dict: Mapping[int, Mapping[str, np.ndarray | float]],
    trajectory_ids: Sequence[int],
    controller: QuadraticBundleController,
    cfg: CostConfig,
    max_cases: int | None = None,
) -> Tuple[List[Dict[str, np.ndarray | float | int]], pd.DataFrame]:
    """Run closed-loop synthesis from bundle initial states."""
    results: List[Dict[str, np.ndarray | float | int]] = []
    ids = list(trajectory_ids)
    if max_cases is not None:
        ids = ids[: int(max_cases)]

    for tid in ids:
        row = bundle_dict[int(tid)]
        x0 = np.asarray(row["X"], dtype=float)[0]
        states, controls, score = synthesize_with_controller(x0, controller, cfg)
        results.append(
            {
                "trajectory_id": int(tid),
                "pred_score": float(score),
                "true_score": float(row["score"]),
                "states": states,
                "controls": controls,
            }
        )

    summary = pd.DataFrame(
        [{k: v for k, v in record.items() if k not in ("states", "controls")} for record in results]
    )
    return results, summary


def terminal_errors(
    results: Sequence[Mapping[str, np.ndarray | float | int]],
    cfg: CostConfig,
    max_cases: int | None = None,
) -> np.ndarray:
    """Compute terminal position errors for synthesized trajectories."""
    rows = list(results)
    if max_cases is not None:
        rows = rows[: int(max_cases)]

    errors = []
    terminal = np.asarray(cfg.terminal_state[:3], dtype=float)
    for row in rows:
        states = np.asarray(row["states"], dtype=float)
        err = np.linalg.norm(states[-1][:3] - terminal)
        errors.append(float(err))
    return np.asarray(errors, dtype=float)

@njit(cache=True)
def model_nb(state, control):
    phi, theta, psi, thrust = control

    out = np.empty(6, dtype=np.float64)
    out[0] = state[3]
    out[1] = state[4]
    out[2] = state[5]
    out[3] = (math.cos(psi) * math.sin(theta) * math.cos(phi) +
              math.sin(psi) * math.sin(phi)) * thrust
    out[4] = (math.sin(psi) * math.sin(theta) * math.cos(phi) -
              math.cos(psi) * math.sin(phi)) * thrust
    out[5] = thrust * math.cos(theta) * math.cos(phi) - EARTH_GRAVITY
    return out

@njit(cache=True)
def rk4_step_nb(state, control, dt):
    k1 = model_nb(state, control)
    k2 = model_nb(state + 0.5 * dt * k1, control)
    k3 = model_nb(state + 0.5 * dt * k2, control)
    k4 = model_nb(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=True)
def rollout_nb(initial_state, controls, dt):
    n = controls.shape[0]
    states = np.empty((n + 1, 6), dtype=np.float64)
    states[0] = initial_state
    for k in range(n):
        states[k + 1] = rk4_step_nb(states[k], controls[k], dt)
    return states

@njit(cache=True)
def clamp_controls_nb(controls):
    lower = CONTROL_LIMITS[:, 0]
    upper = CONTROL_LIMITS[:, 1]
    return np.clip(controls, lower, upper)

@njit(cache=True)
def euclidean_nb(a, b):
    acc = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        acc += d * d
    return math.sqrt(acc)

@njit(cache=True)
def bolza_cost_nb(
    initial_state,
    controls,
    dt,
    num_intervals,
    terminal_state,
    terminal_penalty_weight,
    cylinders,          # shape (n_cyl, 3): x, z, radius
    cylinder_penalty_weight,
    windows,            # shape (n_win, 3): x, z, radius
    window_penalty_weight,
):
    states = rollout_nb(initial_state, controls, dt)

    cylinder_penalty = 0.0
    for i in range(states.shape[0]):
        x1 = states[i, 0]
        x3 = states[i, 2]
        for j in range(cylinders.shape[0]):
            cx = cylinders[j, 0]
            cz = cylinders[j, 1]
            cr = cylinders[j, 2]
            signed_distance = cr - math.hypot(x1 - cx, x3 - cz)
            if signed_distance >= 0.0:
                cylinder_penalty += cylinder_penalty_weight

    window_penalty = 0.0
    for j in range(windows.shape[0]):
        wx = windows[j, 0]
        wz = windows[j, 1]
        wr = windows[j, 2]

        min_distance = 1e300
        for i in range(states.shape[0]):
            d = math.hypot(states[i, 0] - wx, states[i, 2] - wz) - wr
            if d < min_distance:
                min_distance = d

        if min_distance > 0.0:
            window_penalty += window_penalty_weight * (min_distance * min_distance)

    terminal_penalty = terminal_penalty_weight * euclidean_nb(states[num_intervals], terminal_state)

    return dt * num_intervals + terminal_penalty + cylinder_penalty + window_penalty

@njit(cache=True)
def anneal_nb(
    initial_state,
    start_controls,
    dt,
    num_intervals,
    terminal_state,
    terminal_penalty_weight,
    cylinders,
    cylinder_penalty_weight,
    windows,
    window_penalty_weight,
    n_iter=200000,
    step_size=0.01,
    t0=200.0,
):
    current_u = clamp_controls_nb(start_controls.copy())
    current_j = bolza_cost_nb(
        initial_state, current_u, dt, num_intervals,
        terminal_state, terminal_penalty_weight,
        cylinders, cylinder_penalty_weight,
        windows, window_penalty_weight,
    )

    best_u = current_u.copy()
    best_j = current_j

    trace_len = n_iter // 100 + 1
    trace = np.empty(trace_len, dtype=np.float64)
    trace[0] = current_j
    trace_idx = 1

    for i in range(n_iter):
        temp = t0 / (1.0 + 0.01 * i)

        noise = np.random.normal(0.0, step_size, current_u.shape)
        cand_u = clamp_controls_nb(current_u + noise)

        cand_j = bolza_cost_nb(
            initial_state, cand_u, dt, num_intervals,
            terminal_state, terminal_penalty_weight,
            cylinders, cylinder_penalty_weight,
            windows, window_penalty_weight,
        )

        accept = False
        if cand_j < current_j:
            accept = True
        else:
            p = math.exp((current_j - cand_j) / max(temp, 1e-9))
            if np.random.random() < p:
                accept = True

        if accept:
            current_u = cand_u
            current_j = cand_j
            if cand_j < best_j:
                best_j = cand_j
                best_u = cand_u.copy()

        if i % 100 == 0:
            trace[trace_idx] = best_j
            trace_idx += 1

    return best_u, best_j, trace
