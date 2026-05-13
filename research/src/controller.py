from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CONTROL_LIMITS, CostConfig
from physics import bolza_cost, clamp_controls, rk4_step, rollout


class QuadraticBundleController:
    _FEATURE_INDEX_MAP: Dict[int, List[int]] = {
        1: [0],
        2: [0, 3],
        3: [0, 2, 5],
        4: [0, 2, 3, 5],
        5: [0, 2, 3, 4, 5],
        6: [0, 1, 2, 3, 4, 5],
    }

    def __init__(self, m_features: int = 6, ridge_lambda: float = 1e-3) -> None:
        self.m = int(m_features)
        self.ridge_lambda = float(ridge_lambda)
        self.models: Dict[int, np.ndarray] = {}
        self.control_dim = CONTROL_LIMITS.shape[0]

    def _feature_indices(self) -> List[int]:
        return self._FEATURE_INDEX_MAP.get(self.m, list(range(self.m)))

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        idx = self._feature_indices()
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return arr[idx]
        if arr.ndim == 2:
            return arr[:, idx]
        raise ValueError(f"Неподдерживаемая форма входных данных: {arr.shape}")

    def _basis_batch(self, x_batch: np.ndarray) -> np.ndarray:
        x = self._select_features(x_batch)
        m_eff = x.shape[1]

        quad_terms = []
        for i in range(m_eff):
            for j in range(i, m_eff):
                term = x[:, i] * x[:, j]
                if i == j:
                    term = 0.5 * term
                quad_terms.append(term)

        return np.column_stack(quad_terms + [x, np.ones(len(x), dtype=float)])

    def _basis_single(self, x: np.ndarray) -> np.ndarray:
        x = self._select_features(x)
        m_eff = x.shape[0]

        basis = []
        for i in range(m_eff):
            for j in range(i, m_eff):
                basis.append(0.5 * x[i] * x[i] if i == j else x[i] * x[j])
        basis.extend(x.tolist())
        basis.append(1.0)
        return np.asarray(basis, dtype=float)

    def _collect_step_samples(
        self,
        bundle_dict: Mapping,
        trajectory_ids: Sequence[int],
        step: int,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        states: List[np.ndarray] = []
        controls: List[np.ndarray] = []

        for trajectory_id in trajectory_ids:
            row = bundle_dict[int(trajectory_id)]
            state_sequence = np.asarray(row["X"], dtype=float)
            control_sequence = np.asarray(row["U"], dtype=float)

            if step > len(state_sequence) or step > len(control_sequence):
                continue

            states.append(state_sequence[step - 1])
            controls.append(control_sequence[step - 1])

        if not states:
            return None, None

        return np.vstack(states), np.vstack(controls)

    def fit(
        self,
        bundle_dict: Mapping,
        trajectory_ids: Sequence[int],
        n_steps: int,
    ) -> QuadraticBundleController:
        self.models = {}

        for step in range(1, int(n_steps) + 1):
            states, controls = self._collect_step_samples(bundle_dict, trajectory_ids, step)
            if states is None or controls is None:
                continue

            self.control_dim = controls.shape[1]

            design_matrix = self._basis_batch(states)
            ridge_matrix = self.ridge_lambda * np.eye(design_matrix.shape[1], dtype=float)
            self.models[step] = np.linalg.solve(
                design_matrix.T @ design_matrix + ridge_matrix,
                design_matrix.T @ controls,
            )

        return self

    def predict(self, x_state: np.ndarray, step: int) -> np.ndarray:
        coeff = self.models.get(int(step))
        if coeff is None:
            return np.zeros(self.control_dim, dtype=float)
        return clamp_controls(self._basis_single(x_state) @ coeff)


def train_quadratic_controllers(
    bundle_dict: Mapping,
    train_ids: Sequence[int],
    n_steps: int,
    feature_dims: Sequence[int],
    ridge_lambda: float = 2e-3,
) -> Dict[int, QuadraticBundleController]:
    controllers: Dict[int, QuadraticBundleController] = {}
    for m in feature_dims:
        ctrl = QuadraticBundleController(m_features=int(m), ridge_lambda=ridge_lambda)
        ctrl.fit(bundle_dict, train_ids, n_steps)
        controllers[int(m)] = ctrl
    return controllers

def synthesize_with_controller(
    initial_state: np.ndarray,
    controller: QuadraticBundleController,
    cfg: CostConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    state = np.asarray(initial_state, dtype=float)
    controls = []

    for step in range(1, cfg.num_intervals + 1):
        control = controller.predict(state, step)
        controls.append(control)
        state = rk4_step(state, control, cfg.dt)

    controls_arr = np.asarray(controls, dtype=float)
    states = rollout(initial_state, controls_arr, cfg.dt)
    score = bolza_cost(initial_state, controls_arr, cfg)
    return states, controls_arr, score


def evaluate_closed_loop(
    bundle_dict: Mapping,
    trajectory_ids: Sequence[int],
    controller: QuadraticBundleController,
    cfg: CostConfig,
    max_cases: int | None = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    ids = list(trajectory_ids)
    if max_cases is not None:
        ids = ids[:int(max_cases)]

    results = []
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

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("states", "controls")}
        for r in results
    ])
    return results, summary


def evaluate_pointwise_rmse(
    bundle_dict: Mapping,
    test_ids: Sequence[int],
    controller: QuadraticBundleController,
    n_steps: int,
) -> Tuple[np.ndarray, float]:
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
    return np.sqrt(err.mean(axis=0)), float(np.sqrt(err.mean()))


def terminal_errors(
    results: Sequence[Mapping],
    cfg: CostConfig,
    max_cases: int | None = None,
) -> np.ndarray:
    rows = list(results)
    if max_cases is not None:
        rows = rows[:int(max_cases)]

    terminal = np.asarray(cfg.terminal_state[:3], dtype=float)
    errors = []

    for row in rows:
        terminal_state = np.asarray(row["states"], dtype=float)[-1][:3]
        errors.append(float(np.linalg.norm(terminal_state - terminal)))

    return np.asarray(errors, dtype=float)
