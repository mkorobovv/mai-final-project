# Фасад обратной совместимости.
# Импортируйте напрямую из соответствующих модулей:
#   config.py     — константы и конфигурация задачи
#   data.py       — загрузка и разбивка данных
#   physics.py    — динамика квадрокоптера и функционал Больцы
#   annealing.py  — имитация отжига
#   controller.py — квадратичный регулятор и оценка качества

from config import (
    CONTROL_LIMITS,
    EARTH_GRAVITY,
    CostConfig,
    Cylinder,
    Window,
)
from data import (
    build_trajectory_bundle,
    load_training_samples,
    parse_pg_array,
    select_top_k_trajectories,
    split_trajectory_ids,
)
from physics import (
    bolza_cost,
    bolza_cost_bundle_nb,
    bolza_cost_nb,
    clamp_controls,
    clamp_controls_nb,
    euclidean_nb,
    model,
    model_nb,
    rollout,
    rollout_nb,
    rk4_step,
    rk4_step_nb,
)
from annealing import (
    anneal,
    anneal_bundle,
    anneal_bundle_nb,
    anneal_nb,
)
from controller import (
    QuadraticBundleController,
    evaluate_closed_loop,
    evaluate_pointwise_rmse,
    synthesize_with_controller,
    terminal_errors,
    train_quadratic_controllers,
)

__all__ = [
    # config
    "CONTROL_LIMITS", "EARTH_GRAVITY", "CostConfig", "Cylinder", "Window",
    # data
    "build_trajectory_bundle", "load_training_samples", "parse_pg_array",
    "select_top_k_trajectories", "split_trajectory_ids",
    # physics
    "bolza_cost", "bolza_cost_bundle_nb", "bolza_cost_nb",
    "clamp_controls", "clamp_controls_nb", "euclidean_nb",
    "model", "model_nb", "rollout", "rollout_nb", "rk4_step", "rk4_step_nb",
    # annealing
    "anneal", "anneal_bundle", "anneal_bundle_nb", "anneal_nb",
    # controller
    "QuadraticBundleController", "evaluate_closed_loop", "evaluate_pointwise_rmse",
    "synthesize_with_controller", "terminal_errors", "train_quadratic_controllers",
]
