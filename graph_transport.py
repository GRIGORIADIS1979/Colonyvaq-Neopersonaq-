"""Evidence graph, CTQW/RWR comparators, teleport consensus and MBHA scoring."""
from __future__ import annotations
import numpy as np


def rbf_kernel(X: np.ndarray, cols: list[int], gamma: float | None = None) -> np.ndarray:
    Z = X[:, cols]
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    if gamma is None:
        med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
        gamma = 1.0 / (med + 1e-12)
    K = np.exp(-gamma * sq)
    np.fill_diagonal(K, 0.0)
    return K


def build_affinity(X: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    groups = {
        "presentation": [0, 1, 2],
        "processing_expression": [3, 4, 5, 6, 7],
        "immunogenicity": [8],
        "chemistry": [9, 10, 11, 12],
        "manufacturability": [13, 17],
        "redundancy": [14],
        "transport_prior": [15, 16],
    }
    W = np.zeros((X.shape[0], X.shape[0]))
    total = sum(weights.values()) or 1.0
    for name, cols in groups.items():
        W += (weights.get(name, 0.0) / total) * rbf_kernel(X, cols)
    W = 0.5 * (W + W.T)
    return W


def normalized_laplacian(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = W.sum(axis=1)
    inv = 1.0 / np.sqrt(d + 1e-12)
    L = np.eye(W.shape[0]) - (inv[:, None] * W * inv[None, :])
    return L, d


def quotient_labels(W: np.ndarray, n_basins: int = 8) -> np.ndarray:
    # Lightweight spectral partition: first nontrivial eigenvectors -> kmeans-like deterministic assignment.
    L, _ = normalized_laplacian(W)
    vals, vecs = np.linalg.eigh(L)
    Y = vecs[:, 1:min(n_basins, W.shape[0])]
    # deterministic 1D projection assignment for reproducibility
    score = Y[:, 0] if Y.ndim == 2 and Y.shape[1] else vecs[:, 0]
    return np.digitize(score, np.quantile(score, np.linspace(0, 1, n_basins + 1)[1:-1]))


def quotient_matrix(W: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = int(labels.max()) + 1
    Q = np.zeros((k, k))
    for a in range(k):
        ia = labels == a
        for b in range(k):
            ib = labels == b
            if ia.any() and ib.any():
                Q[a, b] = W[np.ix_(ia, ib)].mean()
    return Q


def ctqw_marginal(L: np.ndarray, t: float, psi0: np.ndarray | None = None) -> np.ndarray:
    n = L.shape[0]
    if psi0 is None:
        psi0 = np.ones(n, dtype=complex) / np.sqrt(n)
    vals, vecs = np.linalg.eigh(L)
    Upsi = vecs @ (np.exp(-1j * t * vals) * (vecs.T.conj() @ psi0))
    p = np.abs(Upsi) ** 2
    return p / p.sum()


def random_walk_restart(W: np.ndarray, alpha: float = 0.15, steps: int = 200) -> np.ndarray:
    P = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    prior = np.ones(W.shape[0]) / W.shape[0]
    p = prior.copy()
    for _ in range(steps):
        p = (1 - alpha) * (p @ P) + alpha * prior
    return p / p.sum()


def teleport_consensus(L: np.ndarray, alpha: float, steps: int, t: float) -> np.ndarray:
    n = L.shape[0]
    vals, vecs = np.linalg.eigh(L)
    U = vecs @ np.diag(np.exp(-1j * t * vals)) @ vecs.T.conj()
    rho = np.eye(n, dtype=complex) / n
    reset = np.eye(n, dtype=complex) / n
    for _ in range(steps):
        rho = (1 - alpha) * (U @ rho @ U.conj().T) + alpha * reset
    p = np.real(np.diag(rho))
    return p / p.sum()


def score_mbha(X_raw, p_tel, rwr, min_score: float = 0.0):
    # Positive directions: binding, processing, expression, immunogenicity, chemistry, manufacturability.
    beta = np.array([.10, -.05, .09, .06, .04, .03, .03, .05, .12, .10, .07, .04, .04, .05, -.08, .00, .00, .05])
    base = X_raw @ beta
    margins = np.column_stack([
        X_raw[:, 13] - 0.35,  # manufacturability minimum
        X_raw[:, 9] - 0.30,   # docking support minimum
        0.85 - X_raw[:, 14],  # redundancy ceiling
    ])
    mandatory = (margins >= 0).all(axis=1).astype(int)
    score = base + 4.0 * p_tel + 1.5 * rwr + 0.20 * margins.sum(axis=1) - 0.15 * X_raw[:, 14]
    eligible = ((score >= min_score) & (mandatory == 1)).astype(int)
    return score, eligible, margins
