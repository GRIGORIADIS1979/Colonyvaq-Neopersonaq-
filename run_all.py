#!/usr/bin/env python3
"""Single-command reproducible run for Colonyvaq-Neopersonaq proof of concept."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features import load_or_create_matrix, preprocess, FEATURE_COLUMNS
from src.graph_transport import build_affinity, normalized_laplacian, quotient_labels, quotient_matrix, ctqw_marginal, random_walk_restart, teleport_consensus, score_mbha
from src.baselines import baseline_scores, topk_indices, precision_at_k, enrichment, jaccard
from src.flow_anchor import pepmix_anchor_table
from src.audit import sha256_file, sha256_array, write_manifest


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/reproducible_manifest.yml')
    ap.add_argument('--emit', nargs='*', default=['tables', 'figures', 'manifest'])
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path('results'); figs = Path('figures'); data = Path('data')
    out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True); data.mkdir(exist_ok=True)

    matrix_path = data / 'candidate_feature_matrix_64x18.csv'
    df = load_or_create_matrix(matrix_path, seed=cfg['seed'], n=cfg['n_candidates'])
    Xz, Xraw_df, feat_audit = preprocess(df)
    Xraw = Xraw_df.to_numpy(float)

    W = build_affinity(Xz, cfg['kernel_weights'])
    L, degree = normalized_laplacian(W)
    labels = quotient_labels(W, n_basins=8)
    Q = quotient_matrix(W, labels)
    p_ctqw = ctqw_marginal(L, cfg['ctqw_time'])
    p_rwr = random_walk_restart(W, alpha=cfg['teleport_alpha'])
    p_tel = teleport_consensus(L, alpha=cfg['teleport_alpha'], steps=cfg['teleport_steps'], t=cfg['ctqw_time'])
    score, eligible, margins = score_mbha(Xraw, p_tel, p_rwr)

    k = cfg['top_k']; y = df['audit_positive'].to_numpy(int)
    base = baseline_scores(Xraw)
    base['diffusion_rwr'] = p_rwr
    base['terminal_ctqw'] = p_ctqw
    base['teleport_ctqw'] = p_tel
    base['teleport_ctqw_mbha'] = score

    rows = []
    top_sets = {}
    for name, s in base.items():
        elig = eligible if name == 'teleport_ctqw_mbha' else None
        top = topk_indices(np.asarray(s), k, elig)
        top_sets[name] = top
        rows.append({
            'method': name,
            'precision_at_k': precision_at_k(top, y),
            'enrichment_factor': enrichment(top, y),
            'jaccard_vs_mbha': jaccard(top, top_sets.get('teleport_ctqw_mbha', top)),
            'top_k_candidates': ';'.join(df.loc[top, 'candidate_id']),
        })
    metrics = pd.DataFrame(rows)

    ranked = df[['candidate_id','peptide','basin_hint','pepmix','audit_positive']].copy()
    ranked['quotient_basin'] = labels
    ranked['degree'] = degree
    ranked['rwr_mass'] = p_rwr
    ranked['ctqw_mass'] = p_ctqw
    ranked['teleport_consensus'] = p_tel
    ranked['mbha_score'] = score
    ranked['eligible'] = eligible
    ranked['rank'] = ranked['mbha_score'].rank(ascending=False, method='first').astype(int)
    ranked = ranked.sort_values('rank')

    flow = pepmix_anchor_table()
    pepmix_support = ranked.groupby('pepmix', as_index=False).agg(
        mean_teleport=('teleport_consensus','mean'), mean_mbha=('mbha_score','mean'), selected_count=('eligible','sum')
    ).merge(flow, on='pepmix', how='left')

    if 'tables' in args.emit:
        ranked.to_csv(out / 'ranked_candidates.csv', index=False)
        metrics.to_csv(out / 'benchmark_metrics.csv', index=False)
        pd.DataFrame(W).to_csv(out / 'affinity_matrix.csv', index=False)
        pd.DataFrame(Q).to_csv(out / 'quotient_matrix.csv', index=False)
        pepmix_support.to_csv(out / 'pepmix_endpoint_separated_anchor.csv', index=False)
        pd.DataFrame(margins, columns=['manufacturability_margin','docking_margin','redundancy_margin']).to_csv(out / 'mbha_margins.csv', index=False)

    if 'figures' in args.emit:
        plt.figure(figsize=(8,4.5))
        plt.bar(metrics['method'], metrics['precision_at_k'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Precision@K')
        plt.tight_layout(); plt.savefig(figs / 'fig1_benchmark_precision.png', dpi=180); plt.close()

        plt.figure(figsize=(5.5,4.5))
        plt.imshow(Q)
        plt.title('Quotient-basin affinity')
        plt.colorbar(label='mean affinity')
        plt.tight_layout(); plt.savefig(figs / 'fig2_quotient_matrix.png', dpi=180); plt.close()

        plt.figure(figsize=(8,4.5))
        plt.plot(np.sort(p_rwr)[::-1], label='RWR')
        plt.plot(np.sort(p_ctqw)[::-1], label='terminal CTQW')
        plt.plot(np.sort(p_tel)[::-1], label='teleport consensus')
        plt.ylabel('sorted mass'); plt.xlabel('candidate rank by mass'); plt.legend()
        plt.tight_layout(); plt.savefig(figs / 'fig3_transport_marginals.png', dpi=180); plt.close()

        plt.figure(figsize=(7,4.5))
        pepmix_support.sort_values('cd34_hpc_percent_total').plot(x='pepmix', y=['mean_teleport','cd34_hpc_percent_total'], kind='bar')
        plt.tight_layout(); plt.savefig(figs / 'fig4_pepmix_anchor.png', dpi=180); plt.close()

    if 'manifest' in args.emit:
        manifest = write_manifest(out / 'audit_manifest.json', {
            'seed': cfg['seed'], 'n_candidates': int(len(df)), 'n_predictors': len(FEATURE_COLUMNS),
            'top_k': k, 'endpoint_separation': feat_audit,
            'matrix_sha256': sha256_file(matrix_path), 'affinity_sha256': sha256_array(W),
            'teleport_consensus_sha256': sha256_array(p_tel),
            'ranked_candidates_sha256': sha256_file(out / 'ranked_candidates.csv') if (out / 'ranked_candidates.csv').exists() else None,
        })
        print(json.dumps(manifest, indent=2)[:1200])
    print('Run complete. See results/ and figures/.')

if __name__ == '__main__':
    main()
