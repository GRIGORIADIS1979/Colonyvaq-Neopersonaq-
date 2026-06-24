# Source alignment note

The attached manuscript asks for a GitHub-style package with `README.md`, environment files, `run_all.py`, `src/`, `data/`, `results/`, and `figures/`. This repository implements that architecture and reconstructs the stated workflow:

1. candidate matrix creation or loading,
2. preprocessing and endpoint-separation audit,
3. graph-kernel fusion into `W`,
4. normalized Laplacian `L_sym`,
5. quotient-basin summaries,
6. CTQW and RWR comparators,
7. teleport-consensus stabilization,
8. MBHA margin ledger,
9. ranked candidates and benchmark tables,
10. PepMix CD45/CD34 endpoint-separated validation summaries,
11. SHA-256 audit manifest.

The generated matrix is synthetic and deterministic because the attached manuscript describes the data schema but does not provide the raw numerical CSV.
