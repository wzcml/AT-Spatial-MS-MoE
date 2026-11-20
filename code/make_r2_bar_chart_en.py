"""
R^2 comparison chart for the 12 algorithms (English mode, self-contained)
==========================================================================
Formulation: cosine spatial basis (analytically no-flux) + no-flux soft
constraint + explicit D_eff(x) + 11-element environmental modulation.

What this script produces
    fig_r2_bar_12algos.png
        Vertical bar chart of the coefficient of determination R^2 for the
        twelve algorithms, sorted in ascending order so that the proposed
        model appears at the right-hand end. Bars are colour-coded by design
        lineage. The proposed model is outlined and annotated.

Panel / data provenance
    This figure shows ONE panel only: the overall R^2 comparison under the
    latest formulation. It is NOT the two-formulation comparison figure
    (fig_ns_r2_comparison.png, produced by make_report_figures.py), which
    plots old (sin / Dirichlet) and new (cos / no-flux) R^2 side by side.
    The two figures therefore contain no duplicated panel.

    The numerical values below were measured on N = 500 test points per
    algorithm and match task_139/output/full_spectrum_12algos_11elements.json.

Self-contained design
    The script is runnable in isolation. If the JSON results file is found it
    is used; otherwise the embedded REFERENCE table (identical values) is used,
    so no external project structure is required.

Algorithm naming
    JSON keys are kept verbatim. Two keys are remapped for display only:
        Single head AT-Spatial-MS  ->  R3R4-AT     (single-head AT variant)
        SiGMA_up            ->  SiGMA-G     (gated variant, informational suffix removed)

Usage
    python3 make_r2_bar_chart_en.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

OUT = 'fig_r2_bar_12algos.png'

# ---------------------------------------------------------------------------
# Display order: ascending R^2, so the best model sits on the right-hand side.
# Keys are exactly those stored in the results JSON.
# ---------------------------------------------------------------------------
ORDER = ['Temporal-MS', 'Single head AT-Spatial-MS', 'Fourier-Spatial', 'msMamba',
         'HMformer', 'SiGMA_up', 'TX-Spatial-MS', 'CG-PINN', 'Causal PINN',
         'Standard PINN', 'Spatial-MS', 'AT-Spatial-MS-MoE']

#LABEL = {'SiGMA_up': 'SiGMA-G', 'Single head AT-Spatial-MS': 'R3R4-AT'}
LABEL = {'SiGMA_up': 'SiGMA-G'}
BEST = 'AT-Spatial-MS-MoE'

# Design lineage used for colour coding
GROUP = {
    'Standard PINN': 'Baseline PINN',
    'CG-PINN': 'Baseline PINN',
    'Causal PINN': 'Baseline PINN',
    'Fourier-Spatial': 'Spatial basis',
    'TX-Spatial-MS': 'Multi-scale / SSM',
    'HMformer': 'Multi-scale / SSM',
    'msMamba': 'Multi-scale / SSM',
    'Temporal-MS': 'Multi-scale / SSM',
    'SiGMA_up': 'Gated',
    'Single head AT-Spatial-MS': 'Gated',
    'Spatial-MS': 'Spatial-MS lineage',
    'AT-Spatial-MS-MoE': 'Proposed',
}
COLOR = {
    'Baseline PINN': '#95a5a6',
    'Spatial basis': '#7f8c8d',
    'Multi-scale / SSM': '#aeb6bf',
    'Gated': '#85929e',
    'Spatial-MS lineage': '#2980b9',
    'Proposed': '#c0392b',
}

# ---------------------------------------------------------------------------
# Embedded reference values (N = 500 test points per algorithm).
# Used only when the results JSON cannot be located.
# ---------------------------------------------------------------------------
REFERENCE = {
    'Temporal-MS':          {'R2': -0.0206803048, 'RMSE': 0.5206798562},
    'Single head AT-Spatial-MS':   {'R2': -0.0148883768, 'RMSE': 0.5192004357},
    'Fourier-Spatial':      {'R2': 0.3432766618,  'RMSE': 0.4176544702},
    'msMamba':              {'R2': 0.3864382350,  'RMSE': 0.4036965640},
    'HMformer':             {'R2': 0.3994952871,  'RMSE': 0.3993779827},
    'SiGMA_up':             {'R2': 0.4036464223,  'RMSE': 0.3979951900},
    'TX-Spatial-MS':        {'R2': 0.4468122278,  'RMSE': 0.3833206314},
    'CG-PINN':              {'R2': 0.4627752911,  'RMSE': 0.3777495003},
    'Causal PINN':          {'R2': 0.4797312957,  'RMSE': 0.3717403986},
    'Standard PINN':        {'R2': 0.4866979111,  'RMSE': 0.3692431309},
    'Spatial-MS':           {'R2': 0.4980845785,  'RMSE': 0.3651246711},
    'AT-Spatial-MS-MoE':    {'R2': 0.9667042168,  'RMSE': 0.0940417119},
}


def load_data():
    """Load the measured results if available, otherwise the embedded table."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, 'output', 'full_spectrum_12algos_11elements.json'),
        os.path.join(here, 'full_spectrum_12algos_11elements.json'),
        os.path.join(here, 'dataset', 'results',
                     'full_spectrum_12algos_11elements.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            res = d['results'] if 'results' in d else d
            if all(k in res for k in ORDER):
                return res, os.path.basename(path)
    return REFERENCE, 'embedded reference table'


def main():
    res, source = load_data()

    names = list(ORDER)
    r2 = np.array([res[k]['R2'] for k in names])

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    x = np.arange(len(names))
    colors = [COLOR[GROUP.get(n, 'Baseline PINN')] for n in names]
    bars = ax.bar(x, r2, width=0.68, color=colors, edgecolor='black',
                  linewidth=0.6)

    # Emphasise the proposed model
    i_best = names.index(BEST)
    bars[i_best].set_edgecolor('#7b241c')
    bars[i_best].set_linewidth(1.8)

    # Zero reference line
    ax.axhline(0, color='black', linewidth=0.9)

    # Numerical value above (or below) each bar
    for xi, v in zip(x, r2):
        va = 'bottom' if v >= 0 else 'top'
        off = 0.018 if v >= 0 else -0.018
        ax.text(xi, v + off, f'{v:.3f}', ha='center', va=va, fontsize=8.5,
                fontweight='bold' if abs(v) > 0.9 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(n, n) for n in names], rotation=32,
                       ha='right', fontsize=9)
    ax.set_ylabel(r'Coefficient of determination $R^2$', fontsize=11)
    ax.set_xlabel('Algorithm', fontsize=11)
    ax.set_title(r'$R^2$ of the 12 algorithms under the no-flux formulation'
                 '\n'
                 r'(cosine basis + explicit $D_{eff}(x)$, '
                 r'$N$ = 500 test points per algorithm)',
                 fontsize=11.5, fontweight='bold')
    ax.set_ylim(min(-0.12, r2.min() - 0.08), 1.12)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    # Annotate the proposed model
    ax.annotate(f'Proposed\n$R^2$ = {r2[i_best]:.4f}',
                xy=(i_best, r2[i_best]), xytext=(i_best - 2.2, 0.72),
                fontsize=9, fontweight='bold', color='#7b241c',
                arrowprops=dict(arrowstyle='->', color='#7b241c', lw=1.3))

    # Legend (lineage colours + mean of the 11 competitors)
    handles = [Patch(facecolor=COLOR[g], edgecolor='black', label=g)
               for g in ['Baseline PINN', 'Spatial basis',
                         'Multi-scale / SSM', 'Gated',
                         'Spatial-MS lineage', 'Proposed']]
    comp = np.delete(r2, i_best)
    ax.axhline(comp.mean(), color='#27ae60', linestyle=':', linewidth=1.2)
    handles.append(plt.Line2D([], [], color='#27ae60', linestyle=':',
                              label=f'Mean of the 11 competitors = '
                                    f'{comp.mean():.3f}'))
    ax.legend(handles=handles, fontsize=8, loc='upper left', ncol=2,
              framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ---------------- console summary (English) ----------------
    print('=' * 66)
    print('R^2 comparison of the 12 algorithms (latest formulation)')
    print('=' * 66)
    print(f'Data source       : {source}')
    print(f'Best model        : {BEST}   R^2 = {r2[i_best]:.4f}')
    print(f'Mean of others    : {comp.mean():.4f}   '
          f'(spread = {r2[i_best] - comp.mean():.4f})')
    print(f'Negative R^2 count: {int((r2 < 0).sum())}   '
          '(Temporal-MS, R3R4-AT)')
    print(f'Range (non-best)  : {comp.min():.4f} - {comp.max():.4f}')
    print(f'\nWritten: {OUT} (300 dpi, English labels)')


if __name__ == '__main__':
    main()
