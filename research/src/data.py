from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_pg_array(value: str) -> np.ndarray:
    """Разбирает строку массива PostgreSQL в вещественный вектор NumPy."""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    cleaned = str(value).replace("{", "[").replace("}", "]")
    return np.asarray(ast.literal_eval(cleaned), dtype=float)


def _load_csvs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectories = pd.read_csv(
        data_dir / "trajectories.csv",
        header=None,
        names=["trajectory_id", "created_at"],
    )
    states = pd.read_csv(
        data_dir / "trajectory_states.csv",
        header=None,
        names=["state_id", "trajectory_id", "step", "state", "control", "created_at"],
    )
    scores = pd.read_csv(
        data_dir / "scores.csv",
        header=None,
        names=["score_id", "trajectory_id", "score"],
    )
    return trajectories, states, scores


def load_training_samples(data_dir: str | Path) -> pd.DataFrame:
    """
    Возвращает плоскую таблицу выборки:
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


def build_trajectory_bundle(samples: pd.DataFrame) -> Dict[int, Dict]:
    """
    Преобразует плоскую таблицу в словарь траекторий:
    bundle[trajectory_id] = {"X": (N, 6), "U": (N, 4), "score": float}
    """
    bundle: Dict[int, Dict] = {}
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
    bundle: Mapping,
    top_k: int,
) -> Tuple[List[int], Dict[int, Dict], pd.DataFrame]:
    """Выбирает top-k траекторий с наименьшим значением функционала."""
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
    """Перемешивает идентификаторы траекторий и разбивает на train/test."""
    ids = np.asarray(trajectory_ids, dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    train_cut = int(float(train_ratio) * len(shuffled))
    return shuffled[:train_cut], shuffled[train_cut:]
