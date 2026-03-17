from __future__ import annotations

import math

import numpy as np
from numba import njit

from config import CONTROL_LIMITS, EARTH_GRAVITY, CostConfig


# ---------------------------------------------------------------------------
# Pure Python (используется для интерактивной отладки и в болза-функционале)
# ---------------------------------------------------------------------------

def model(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """
    Трансляционная динамика квадрокоптера.
    state:   [x1, x2, x3, v1, v2, v3]
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
    """Один шаг классического интегратора RK4."""
    k1 = model(state, control)
    k2 = model(state + 0.5 * dt * k1, control)
    k3 = model(state + 0.5 * dt * k2, control)
    k4 = model(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(initial_state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    """Интегрирует траекторию по всей последовательности управлений.
    Возвращает массив формы (len(controls) + 1, 6).
    """
    states = np.zeros((len(controls) + 1, len(initial_state)), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for k, u in enumerate(controls):
        states[k + 1] = rk4_step(states[k], np.asarray(u, dtype=float), dt)
    return states


def clamp_controls(controls: np.ndarray) -> np.ndarray:
    """Ограничивает управления физически допустимым диапазоном."""
    return np.clip(np.asarray(controls, dtype=float), CONTROL_LIMITS[:, 0], CONTROL_LIMITS[:, 1])


def bolza_cost(initial_state: np.ndarray, controls: np.ndarray, cfg: CostConfig) -> float:
    """
    Функционал Больцы:
    dt * N + штраф_терминальный + штрафы_цилиндры + штраф_окно.
    """
    states = rollout(initial_state, controls, cfg.dt)

    cylinder_penalty = 0.0
    for st in states:
        x1, x3 = st[0], st[2]
        for cyl in cfg.cylinders:
            if cyl.radius - math.hypot(x1 - cyl.x, x3 - cyl.z) >= 0:
                cylinder_penalty += cfg.cylinder_penalty

    window_penalty = 0.0
    for wnd in cfg.windows:
        min_dist = min(math.hypot(st[0] - wnd.x, st[2] - wnd.z) - wnd.radius for st in states)
        if min_dist > 0:
            window_penalty += cfg.window_penalty * (min_dist ** 2)

    terminal = np.asarray(cfg.terminal_state, dtype=float)
    terminal_penalty = cfg.terminal_penalty * float(np.linalg.norm(states[cfg.num_intervals] - terminal))

    return cfg.dt * cfg.num_intervals + terminal_penalty + cylinder_penalty + window_penalty


# ---------------------------------------------------------------------------
# Numba JIT (используется в имитации отжига и пакетном вычислении функционала)
# ---------------------------------------------------------------------------

@njit(cache=True)
def model_nb(state, control):
    phi, theta, psi, thrust = control
    out = np.empty(6, dtype=np.float64)
    out[0] = state[3]
    out[1] = state[4]
    out[2] = state[5]
    out[3] = (math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)) * thrust
    out[4] = (math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)) * thrust
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
    cylinders,              # shape (n_cyl, 3): x, z, radius
    cylinder_penalty_weight,
    windows,                # shape (n_win, 3): x, z, radius
    window_penalty_weight,
):
    states = rollout_nb(initial_state, controls, dt)

    cylinder_penalty = 0.0
    for i in range(states.shape[0]):
        x1 = states[i, 0]
        x3 = states[i, 2]
        for j in range(cylinders.shape[0]):
            signed_dist = cylinders[j, 2] - math.hypot(x1 - cylinders[j, 0], x3 - cylinders[j, 1])
            if signed_dist >= 0.0:
                cylinder_penalty += cylinder_penalty_weight

    window_penalty = 0.0
    for j in range(windows.shape[0]):
        wx, wz, wr = windows[j, 0], windows[j, 1], windows[j, 2]
        min_dist = 1e300
        for i in range(states.shape[0]):
            d = math.hypot(states[i, 0] - wx, states[i, 2] - wz) - wr
            if d < min_dist:
                min_dist = d
        if min_dist > 0.0:
            window_penalty += window_penalty_weight * (min_dist * min_dist)

    terminal_penalty = terminal_penalty_weight * euclidean_nb(states[num_intervals], terminal_state)

    return dt * num_intervals + terminal_penalty + cylinder_penalty + window_penalty


@njit(cache=True)
def bolza_cost_bundle_nb(
    initial_states,         # shape (S, 6) — пучок начальных состояний
    controls,               # shape (N, 4) — общее управление
    dt,
    num_intervals,
    terminal_state,
    terminal_penalty_weight,
    cylinders,
    cylinder_penalty_weight,
    windows,
    window_penalty_weight,
):
    """Среднее значение функционала по всем начальным состояниям пучка."""
    total = 0.0
    for k in range(initial_states.shape[0]):
        total += bolza_cost_nb(
            initial_states[k], controls, dt, num_intervals,
            terminal_state, terminal_penalty_weight,
            cylinders, cylinder_penalty_weight,
            windows, window_penalty_weight,
        )
    return total / initial_states.shape[0]
