"""Feature-matrix generation and preprocessing for the reproducible proof of concept.

This demonstrator creates a deterministic 64x18 matrix matching the manuscript's
source-code architecture. Replace data/candidate_feature_matrix_64x18.csv with
real endpoint-separated inputs to rerun the same pipeline on study data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hla_binding_score", "binding_percentile_rank", "netmhcpan_style_score",
    "mhcflurry_style_processing", "rna_expression_log2", "dna_naf", "rna_naf",
    "clonality_prior", "immunogenicity_prior", "docking_support",
    "contact_persistence", "rmsd_stability", "rmsf_stability",
    "manufacturability", "redundancy_penalty", "ctqw_support_prior",
    "teleport_consensus_prior", "reserve_margin_prior",
]
ENDPOINT_COLUMNS = ["audit_positive", "pepmix", "cd34_hpc_total", "cd34_cd45dim_parent"]

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def make_demo_matrix(path: str | Path, seed: int = 159, n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    basins = np.repeat(np.arange(8), n // 8)
    if len(basins) < n:
        basins = np.concatenate([basins, rng.integers(0, 8, n - len(basins))])
    rng.shuffle(basins)

    latent = rng.normal(size=(n, 5))
    basin_effect = rng.normal(0, 0.55, size=(8, 5))[basins]
    Z = latent + basin_effect

    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    X = np.column_stack([
        logistic(1.2 * Z[:, 0] - 0.2 * Z[:, 1]),
        1.0 - logistic(1.1 * Z[:, 0] + rng.normal(0, .2, n)),
        logistic(0.9 * Z[:, 0] + 0.4 * Z[:, 2]),
        logistic(0.7 * Z[:, 1] + 0.3 * Z[:, 3]),
        np.clip(5 + 2 * Z[:, 1] + rng.normal(0, .5, n), 0, 12),
        logistic(Z[:, 2]),
        logistic(Z[:, 2] + 0.2 * Z[:, 3]),
        logistic(0.5 * Z[:, 2] + 0.5 * Z[:, 4]),
        logistic(0.8 * Z[:, 3] - 0.1 * Z[:, 0]),
        logistic(0.7 * Z[:, 4] + 0.3 * Z[:, 0]),
        logistic(0.8 * Z[:, 4]),
        logistic(-0.7 * np.abs(Z[:, 4]) + 0.5),
        logistic(-0.6 * np.abs(Z[:, 3]) + 0.4),
        logistic(0.4 * Z[:, 0] + 0.4 * Z[:, 1]),
        logistic(-0.7 * Z[:, 0] + 0.2 * Z[:, 2]),
        logistic(0.2 * Z[:, 0] + 0.5 * Z[:, 4]),
        logistic(0.3 * Z[:, 1] + 0.4 * Z[:, 3]),
        logistic(0.5 * Z[:, 0] + 0.5 * Z[:, 2] - 0.2),
    ])
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df.insert(0, "candidate_id", [f"CNQ159_{i:03d}" for i in range(n)])
    df.insert(1, "peptide", ["".join(rng.choice(AA, size=rng.integers(8, 12))) for _ in range(n)])
    df.insert(2, "basin_hint", basins)
    df["audit_positive"] = ((df["hla_binding_score"] + df["immunogenicity_prior"] + df["docking_support"] + rng.normal(0, .15, n)) > 1.75).astype(int)
    pepmix = np.array(["PepMix1", "PepMix2", "PepMix3", "PepMix4", "PepMix5", "PepMix6"])
    df["pepmix"] = pepmix[np.arange(n) % len(pepmix)]
    hpc_map = {"PepMix1": 0.26, "PepMix2": 0.45, "PepMix3": 0.27, "PepMix4": 0.24, "PepMix5": 0.29, "PepMix6": 0.15}
    dim_map = {"PepMix1": 0.26, "PepMix2": 0.31, "PepMix3": 0.24, "PepMix4": 0.22, "PepMix5": 0.24, "PepMix6": 0.14}
    df["cd34_hpc_total"] = df["pepmix"].map(hpc_map)
    df["cd34_cd45dim_parent"] = df["pepmix"].map(dim_map)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_or_create_matrix(path: str | Path, seed: int, n: int) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return make_demo_matrix(path, seed=seed, n=n)


def preprocess(df: pd.DataFrame):
    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df = feature_df.clip(lower=feature_df.quantile(0.01), upper=feature_df.quantile(0.99), axis=1)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    mu = feature_df.mean(axis=0)
    sigma = feature_df.std(axis=0).replace(0, 1.0)
    Xz = (feature_df - mu) / (sigma + 1e-12)
    audit = {
        "feature_columns": FEATURE_COLUMNS,
        "endpoint_columns_excluded": ENDPOINT_COLUMNS,
        "candidate_id_excluded": True,
        "peptide_excluded": True,
        "endpoint_separated": True,
    }
    return Xz.to_numpy(float), feature_df, audit
