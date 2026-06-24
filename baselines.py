"""Comparator metrics for proof-of-concept neoantigen ranking."""
from __future__ import annotations
import numpy as np


def topk_indices(score: np.ndarray, k: int, eligible: np.ndarray | None = None) -> np.ndarray:
    s = score.copy().astype(float)
    if eligible is not None:
        s[eligible == 0] = -np.inf
    return np.argsort(-s)[:k]


def precision_at_k(top: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(y[top]))


def enrichment(top: np.ndarray, y: np.ndarray) -> float:
    bg = float(np.mean(y)) or 1e-12
    return precision_at_k(top, y) / bg


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    A, B = set(map(int, a)), set(map(int, b))
    return len(A & B) / max(1, len(A | B))


def rank_stability(score_fn, X_raw: np.ndarray, k: int, seed: int = 159, n_rep: int = 40) -> float:
    rng = np.random.default_rng(seed)
    base_top = topk_indices(score_fn(X_raw), k)
    vals = []
    for _ in range(n_rep):
        jitter = X_raw + rng.normal(0, 0.025, size=X_raw.shape)
        vals.append(jaccard(base_top, topk_indices(score_fn(jitter), k)))
    return float(np.mean(vals))


def baseline_scores(X_raw: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "binding_only": X_raw[:, 0] - X_raw[:, 1],
        "netmhcpan_style": X_raw[:, 2],
        "docking_only": X_raw[:, 9],
        "weighted_aggregation": (
            .18 * X_raw[:, 0] - .08 * X_raw[:, 1] + .14 * X_raw[:, 2] + .12 * X_raw[:, 3]
            + .10 * X_raw[:, 8] + .12 * X_raw[:, 9] + .08 * X_raw[:, 10] + .12 * X_raw[:, 13]
            - .08 * X_raw[:, 14]
        ),
    }
