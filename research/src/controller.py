from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CONTROL_LIMITS, CostConfig
from physics import bolza_cost, clamp_controls, rollout


class QuadraticBundleController:
    """
    Пошаговый регулятор на основе гребневой регрессии на квадратичных признаках состояния.

    Для каждого временного шага обучается отдельная линейная модель:
        u = W^T * phi(x),
    где phi(x) — вектор квадратичных, линейных и константного признаков.
    """

    # Карта: количество признаков → индексы координат из 6-мерного состояния
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
        """Строит матрицу признаков для батча состояний (N, m_eff) → (N, d_basis)."""
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
        """Строит вектор признаков для одного состояния (m_eff,) → (d_basis,)."""
        x = self._select_features(x)
        m_eff = x.shape[0]

        basis = []
        for i in range(m_eff):
            for j in range(i, m_eff):
                basis.append(0.5 * x[i] * x[i] if i == j else x[i] * x[j])
        basis.extend(x.tolist())
        basis.append(1.0)
        return np.asarray(basis, dtype=float)

    def fit(
        self,
        bundle_dict: Mapping,
        trajectory_ids: Sequence[int],
        n_steps: int,
    ) -> QuadraticBundleController:
        """Обучает регрессионные модели для каждого шага по обучающей выборке траекторий."""
        self.models = {}

        for step in range(1, int(n_steps) + 1):
            x_rows, u_rows = [], []

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

            G = self._basis_batch(xs)
            reg = self.ridge_lambda * np.eye(G.shape[1], dtype=float)
            # Гребневая регрессия: W = (G^T G + λI)^{-1} G^T U
            self.models[step] = np.linalg.solve(G.T @ G + reg, G.T @ us)

        return self

    def predict(self, x_state: np.ndarray, step: int) -> np.ndarray:
        """Вычисляет управление по текущему состоянию и номеру шага."""
        coeff = self.models.get(int(step))
        if coeff is None:
            return np.zeros(self.control_dim, dtype=float)
        u = self._basis_single(x_state) @ coeff
        return clamp_controls(u)


# ---------------------------------------------------------------------------
# Фабрика регуляторов
# ---------------------------------------------------------------------------

def train_quadratic_controllers(
    bundle_dict: Mapping,
    train_ids: Sequence[int],
    n_steps: int,
    feature_dims: Sequence[int],
    ridge_lambda: float = 2e-3,
) -> Dict[int, QuadraticBundleController]:
    """Обучает набор квадратичных регуляторов для разных размерностей признакового пространства."""
    controllers: Dict[int, QuadraticBundleController] = {}
    for m in feature_dims:
        ctrl = QuadraticBundleController(m_features=int(m), ridge_lambda=ridge_lambda)
        ctrl.fit(bundle_dict, train_ids, n_steps)
        controllers[int(m)] = ctrl
    return controllers


# ---------------------------------------------------------------------------
# Оценка качества
# ---------------------------------------------------------------------------

def synthesize_with_controller(
    initial_state: np.ndarray,
    controller: QuadraticBundleController,
    cfg: CostConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Разворачивает замкнутую траекторию с синтезированным управлением."""
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
    bundle_dict: Mapping,
    trajectory_ids: Sequence[int],
    controller: QuadraticBundleController,
    cfg: CostConfig,
    max_cases: int | None = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Запускает синтез для начальных состояний из тестовой выборки."""
    ids = list(trajectory_ids)
    if max_cases is not None:
        ids = ids[:int(max_cases)]

    results = []
    for tid in ids:
        row = bundle_dict[int(tid)]
        x0 = np.asarray(row["X"], dtype=float)[0]
        states, controls, score = synthesize_with_controller(x0, controller, cfg)
        results.append({
            "trajectory_id": int(tid),
            "pred_score": float(score),
            "true_score": float(row["score"]),
            "states": states,
            "controls": controls,
        })

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
    """Вычисляет RMSE управления по шагам на тестовой выборке."""
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
    """Вычисляет ошибки по положению в конечный момент времени."""
    rows = list(results)
    if max_cases is not None:
        rows = rows[:int(max_cases)]

    terminal = np.asarray(cfg.terminal_state[:3], dtype=float)
    errors = [float(np.linalg.norm(np.asarray(r["states"], dtype=float)[-1][:3] - terminal)) for r in rows]
    return np.asarray(errors, dtype=float)
