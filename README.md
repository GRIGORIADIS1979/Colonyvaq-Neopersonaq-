# Colonyvaq-Neopersonaq reproducible source

GitHub-ready proof-of-concept source package for the manuscript concept **Teleport-Stabilized Quantum-Walk Transport for Robust Neoantigen Ranking in Near-Tie Regimes**.

The package implements the reproducible architecture described in the attached manuscript: a 64-candidate by 18-predictor matrix, endpoint-separated preprocessing, weighted evidence graph construction, normalized Laplacian, CTQW and diffusion/RWR comparators, teleport-consensus stabilization, MBHA boundary margins, ranked candidates, benchmark tables, PepMix CD45/CD34 external-anchor summaries, figures, and a SHA-256 audit manifest.

## What this repository contains

```text
README.md
requirements.txt
environment.yml
configs/reproducible_manifest.yml
data/candidate_feature_matrix_64x18.csv        # generated deterministically if absent
src/features.py                                # matrix loading, demo matrix, leakage audit
src/graph_transport.py                         # W, L_sym, quotient basins, CTQW, RWR, teleport
src/baselines.py                               # baseline scores and benchmark metrics
src/flow_anchor.py                             # endpoint-separated CD45/CD34 PepMix anchor
src/audit.py                                   # SHA-256 manifest utilities
run_all.py                                     # single-command reproducible runner
results/                                       # generated CSV outputs and audit manifest
figures/                                       # generated benchmark/transport/anchor figures
```

## Quick start

```bash
git clone <your-repository-url>
cd Colonyvaq-Neopersonaq-reproducible-source
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_all.py --config configs/reproducible_manifest.yml --emit tables figures manifest
```

After the run, inspect:

- `results/ranked_candidates.csv`
- `results/benchmark_metrics.csv`
- `results/pepmix_endpoint_separated_anchor.csv`
- `results/audit_manifest.json`
- `figures/fig1_benchmark_precision.png`
- `figures/fig2_quotient_matrix.png`
- `figures/fig3_transport_marginals.png`
- `figures/fig4_pepmix_anchor.png`

## Endpoint-separation policy

The code excludes `candidate_id`, `peptide`, `audit_positive`, `pepmix`, `cd34_hpc_total`, and `cd34_cd45dim_parent` from graph construction, CTQW initialization, teleport prior, MBHA scoring, and baseline aggregation. These variables are used only for metadata or external validation.

## Replacing the synthetic demonstrator matrix

The manuscript text specifies the repository contract but does not include a raw 64x18 CSV in the uploaded document. Therefore this package generates a deterministic demonstrator matrix when `data/candidate_feature_matrix_64x18.csv` is absent. To use real study inputs, replace that CSV with a file containing the feature columns declared in `src/features.py`, while keeping endpoint columns outside the ranking feature set.

## Scientific scope

This is a computational reproducibility scaffold and proof-of-concept implementation. It does not make clinical-efficacy claims and does not replace validated neoantigen-prediction or clinical decision software.
