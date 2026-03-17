from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit

from config import CostConfig
from physics import bolza_cost_nb, bolza_cost_bundle_nb, clamp_controls_nb


# ---------------------------------------------------------------------------
# Numba JIT (внутренняя реализация алгоритма отжига)
# ---------------------------------------------------------------------------

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
    """Имитация отжига для одного начального состояния."""
    current_u = clamp_controls_nb(start_controls.copy())
    current_j = bolza_cost_nb(
        initial_state, current_u, dt, num_intervals,
        terminal_state, terminal_penalty_weight,
        cylinders, cylinder_penalty_weight,
        windows, window_penalty_weight,
    )

    best_u = current_u.copy()
    best_j = current_j

    trace = np.empty(n_iter // 100 + 1, dtype=np.float64)
    trace[0] = current_j
    trace_idx = 1

    for i in range(n_iter):
        temp = t0 / (1.0 + 0.01 * i)
        cand_u = clamp_controls_nb(current_u + np.random.normal(0.0, step_size, current_u.shape))
        cand_j = bolza_cost_nb(
            initial_state, cand_u, dt, num_intervals,
            terminal_state, terminal_penalty_weight,
            cylinders, cylinder_penalty_weight,
            windows, window_penalty_weight,
        )

        if cand_j < current_j or np.random.random() < math.exp((current_j - cand_j) / max(temp, 1e-9)):
            current_u = cand_u
            current_j = cand_j
            if cand_j < best_j:
                best_j = cand_j
                best_u = cand_u.copy()

        if i % 100 == 0:
            trace[trace_idx] = best_j
            trace_idx += 1

    return best_u, best_j, trace


@njit(cache=True)
def anneal_bundle_nb(
    initial_states,         # shape (S, 6) — пучок начальных состояний
    start_controls,         # shape (N, 4) — начальное приближение U*
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
    """Имитация отжига, минимизирующая среднее J по пучку начальных состояний."""
    current_u = clamp_controls_nb(start_controls.copy())
    current_j = bolza_cost_bundle_nb(
        initial_states, current_u, dt, num_intervals,
        terminal_state, terminal_penalty_weight,
        cylinders, cylinder_penalty_weight,
        windows, window_penalty_weight,
    )

    best_u = current_u.copy()
    best_j = current_j

    trace = np.empty(n_iter // 100 + 1, dtype=np.float64)
    trace[0] = current_j
    trace_idx = 1

    for i in range(n_iter):
        temp = t0 / (1.0 + 0.01 * i)
        cand_u = clamp_controls_nb(current_u + np.random.normal(0.0, step_size, current_u.shape))
        cand_j = bolza_cost_bundle_nb(
            initial_states, cand_u, dt, num_intervals,
            terminal_state, terminal_penalty_weight,
            cylinders, cylinder_penalty_weight,
            windows, window_penalty_weight,
        )

        if cand_j < current_j or np.random.random() < math.exp((current_j - cand_j) / max(temp, 1e-9)):
            current_u = cand_u
            current_j = cand_j

        if cand_j < best_j:
            best_j = cand_j
            best_u = cand_u.copy()

        if i % 100 == 0:
            trace[trace_idx] = best_j
            trace_idx += 1

    return best_u, best_j, trace


# ---------------------------------------------------------------------------
# Python-обёртки (принимают CostConfig, конвертируют в массивы для Numba)
# ---------------------------------------------------------------------------

def _cfg_to_arrays(cfg: CostConfig):
    return (
        np.asarray(cfg.terminal_state, dtype=np.float64),
        np.asarray([(c.x, c.z, c.radius) for c in cfg.cylinders], dtype=np.float64),
        np.asarray([(w.x, w.z, w.radius) for w in cfg.windows], dtype=np.float64),
    )


def anneal(
    initial_state: np.ndarray,
    start_controls: np.ndarray,
    cfg: CostConfig,
    n_iter: int = 200000,
    step_size: float = 0.01,
    t0: float = 200.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Оптимизация управления для одного начального состояния."""
    terminal_state, cylinders, windows = _cfg_to_arrays(cfg)
    return anneal_nb(
        initial_state=np.asarray(initial_state, dtype=np.float64),
        start_controls=np.asarray(start_controls, dtype=np.float64),
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


def anneal_bundle(
    initial_states: np.ndarray,
    start_controls: np.ndarray,
    cfg: CostConfig,
    n_iter: int = 200000,
    step_size: float = 0.01,
    t0: float = 200.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Оптимизация управления по пучку начальных состояний (минимизация среднего J)."""
    terminal_state, cylinders, windows = _cfg_to_arrays(cfg)
    return anneal_bundle_nb(
        initial_states=np.asarray(initial_states, dtype=np.float64),
        start_controls=np.asarray(start_controls, dtype=np.float64),
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
