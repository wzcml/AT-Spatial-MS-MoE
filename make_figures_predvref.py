"""
Unified figure source for the SCI report, "prediction vs reference" convention (11-element nutrient vector)
============================================================

Convention note:
  All figures in this script use "each algorithm/model's own prediction vs its own reference (ground truth)".
  That is, each algorithm pairs its own prediction u_hat with its own truth u point-by-point to compute R^2, Pearson r,
  per-scale R^2, and paired t / Wilcoxon significance. **NO cross-algorithm comparison (vs MoE)**.

Output figures (with report numbers):
  Fig 2   fig1_r2_comparison.png        12-algorithm R^2 comparison (prediction vs truth)
  Fig 2b  fig5_pearson_r_comparison.png 12-algorithm Pearson r comparison (prediction vs truth)
  Fig 3   fig2_per_scale_r2_heatmap.png per-scale R^2 heatmap (prediction vs truth)
  Fig 4   fig4_lineage_ablation.png     lineage ablation (prediction vs truth R^2/RMSE at each stage)
  Fig 5   fig3_discovering_physics.png  Discovering Physics inversion (inverted curve vs true curve)

Data sources:
  - algorithm-comparison figures: output/pointwise_predictions_11elements.json
    (produced by full_spectrum_12algos_11elements.py; 11-element pointwise prediction + truth)
  - inversion figure: output/inversion_curves.npz
    (Discovering Physics inversion: true vs inverted porosity field)

Note: the Pseudomonas case figures (Hopf bifurcation / limit cycle / dosage / community) are not in this script;
their data are model simulation curves (not "algorithm prediction vs reference"); see the standalone scripts
inhibition_dde_v2.py and multi_pathway_inhibition.py.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PRED_FILE = 'output/pointwise_predictions_11elements.json'
INV_FILE = 'output/inversion_curves.npz'
SCALES = ['minute', 'hour', 'day', 'month', 'year']
NAME_MAP = {}  # names already consistent in the 11-element training script
REF = 'AT-Spatial-MS-MoE'


# ---------- data loading: pointwise "prediction vs truth" statistics ----------
def load_pred_stats():
    """Return {algo: {rmse, r2, pearson_r, per_scale_r2}}, all computed as own-prediction vs own-truth."""
    d = json.load(open(PRED_FILE))
    stats = {}
    for algo, sdict in d.items():
        pred, true = [], []
        per = {}
        for s in SCALES:
            p = np.array(sdict[s]['pred']); t = np.array(sdict[s]['truth'])
            pred.append(p); true.append(t)
            ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean()) ** 2).sum()
            per[s] = float(1 - ssr / sst) if sst > 1e-12 else 0.0
        p = np.concatenate(pred); t = np.concatenate(true)
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean()) ** 2).sum()
        r2 = float(1 - ssr / sst) if sst > 1e-12 else 0.0
        r, _ = st.pearsonr(p, t)
        stats[NAME_MAP.get(algo, algo)] = {'rmse': rmse, 'r2': r2, 'r': r, 'per': per}
    return stats


# ---------- Fig 2: R^2 comparison (prediction vs truth) ----------
def fig_r2_comparison(stats):
    order = sorted([a for a in stats if a != REF], key=lambda a: stats[a]['r2']) + [REF]
    r2_vals = [stats[a]['r2'] for a in order]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    colors = ['#E53935' if a == REF else '#4A90D9' for a in order]
    bars = ax.bar(range(len(order)), r2_vals, color=colors, alpha=0.88)
    for i, a in enumerate(order):
        ax.text(i, r2_vals[i] + 0.025, f'{r2_vals[i]:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold' if a == REF else 'normal')
    bars[-1].set_edgecolor('black'); bars[-1].set_linewidth(2.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('$R^2$ (predicted vs reference)', fontsize=12)
    ax.set_title('11-element model: $R^2$ of each algorithm vs its own reference',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(-0.05, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig('fig1_r2_comparison.png', dpi=200, bbox_inches='tight'); plt.close()


# ---------- Fig 2b: Pearson r comparison (prediction vs truth) ----------
def fig_pearson_r(stats):
    order = sorted([a for a in stats if a != REF], key=lambda a: stats[a]['r']) + [REF]
    r_vals = [stats[a]['r'] for a in order]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    colors = ['#E53935' if a == REF else '#7E57C2' for a in order]
    bars = ax.bar(range(len(order)), r_vals, color=colors, alpha=0.88)
    for i, a in enumerate(order):
        ax.text(i, r_vals[i] + 0.02, f'{r_vals[i]:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold' if a == REF else 'normal')
    bars[-1].set_edgecolor('black'); bars[-1].set_linewidth(2.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('Pearson $r$ (predicted vs reference)', fontsize=12)
    ax.set_title('11-element model: Pearson correlation of each algorithm vs its own reference',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig('fig5_pearson_r_comparison.png', dpi=200, bbox_inches='tight'); plt.close()


# ---------- Fig 3: per-scale R^2 heatmap (prediction vs truth) ----------
def fig_per_scale_heatmap(stats):
    order = sorted([a for a in stats if a != REF], key=lambda a: stats[a]['r2']) + [REF]
    mat = np.array([[stats[a]['per'][s] for a in order] for s in SCALES])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.4, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='black')
    cb = plt.colorbar(im, ax=ax); cb.set_label('$R^2$', fontsize=10)
    ax.set_title('Per-scale $R^2$ heatmap (predicted vs reference)', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig('fig2_per_scale_r2_heatmap.png', dpi=200, bbox_inches='tight'); plt.close()


# ---------- Fig 4: lineage ablation (prediction vs truth at each stage) ----------
def fig_lineage_ablation(stats):
    stages = ['Standard\nPINN', 'Spatial\n-MS', '+AT\n(single-head)', '+MoE\n(P·S·G)']
    stage_r2 = [stats['Standard PINN']['r2'], stats['Spatial-MS']['r2'],
                stats['R3R4-AT']['r2'], stats['AT-Spatial-MS-MoE']['r2']]
    stage_rmse = [stats['Standard PINN']['rmse'], stats['Spatial-MS']['rmse'],
                  stats['R3R4-AT']['rmse'], stats['AT-Spatial-MS-MoE']['rmse']]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(stages))
    ax.bar(x - 0.2, stage_r2, width=0.4, label='$R^2$', color='#4A90D9')
    ax.set_ylim(0, 1.15); ax.set_ylabel('$R^2$')
    for i, v in enumerate(stage_r2):
        ax.text(i - 0.2, v + 0.05, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, stage_rmse, 'o-', color='#E91E63', lw=2, label='RMSE')
    ax2.set_ylabel('RMSE', color='#E91E63'); ax2.set_ylim(0, 0.25)
    ax2.tick_params(axis='y', labelcolor='#E91E63')
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_title('Lineage ablation (predicted vs reference of each stage)', fontsize=11, fontweight='bold')
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc='upper center', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig('fig4_lineage_ablation.png', dpi=200, bbox_inches='tight'); plt.close()


# ---------- Fig 5: Discovering Physics inversion (inverted vs true curve) ----------
def fig_inversion():
    d = np.load(INV_FILE)
    x = d['x']; phi_true = d['phi_true']; phi_inv = d['phi_inverted']
    depth_cm = x * 50  # x∈[0,1] → 0–50 cm
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(depth_cm, phi_true, 'k-', lw=2.2, label='True φ(x) (reference)')
    ax.plot(depth_cm, phi_inv, 'r--', lw=2.2, label='Inverted φ(x) (prediction)')
    ax.axvline(30, color='blue', ls=':', lw=1.5, alpha=0.7)
    ax.annotate('30 cm breakpoint', xy=(30, 0.5), xytext=(33, 0.62), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
    ax.set_xlabel('Soil depth (cm)', fontsize=11)
    ax.set_ylabel('Porosity φ', fontsize=11)
    ax.set_title('Discovering Physics: inversion (prediction) recovers reference breakpoint',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('fig3_discovering_physics.png', dpi=200, bbox_inches='tight'); plt.close()


def main():
    stats = load_pred_stats()
    fig_r2_comparison(stats)
    fig_pearson_r(stats)
    fig_per_scale_heatmap(stats)
    fig_lineage_ablation(stats)
    fig_inversion()
    print('ALL PREDICTED-vs-REFERENCE FIGURES REGENERATED')
    print(f'  MoE R^2={stats["AT-Spatial-MS-MoE"]["r2"]:.4f}, '
          f'Pearson r={stats["AT-Spatial-MS-MoE"]["r"]:.4f}')


if __name__ == '__main__':
    main()
