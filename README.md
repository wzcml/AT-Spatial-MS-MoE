# Reproduction Package: AT-Spatial-MS-MoE for Multi-factor Microbial Growth PDE

This package bundles the core code, data, and figures of this study for full reproduction.

## Directory Structure

```
reproducible_pkg/
├── requirements.txt                          # pip dependencies (with system-level notes)
├── README.md                                 # this file
├── code/
│   ├── at_spatial_ms_moe.py                  # AT-Spatial-MS-MoE core (self-contained, runs a self-check)
│   ├── train_12algos_11elements_predvref.py  # 12-algorithm lineage training (11-element + own-prediction-vs-reference)
│   └── make_figures_predvref.py              # figure script for Fig 2/2b/3/4/5 (prediction-vs-reference)
├── data/
│   ├── pointwise_predictions_11elements.json # pointwise prediction + truth (source for figures and P values)
│   └── inversion_curves.npz                  # Discovering Physics inversion curves (true vs inverted field)
└── figures/
    ├── fig1_r2_comparison.png                # Fig 2: 12-algorithm R^2 comparison
    ├── fig5_pearson_r_comparison.png         # Fig 2b: 12-algorithm Pearson r comparison
    ├── fig2_per_scale_r2_heatmap.png         # Fig 3: per-scale R^2 heatmap
    ├── fig4_lineage_ablation.png             # Fig 4: lineage ablation
    └── fig3_discovering_physics.png          # Fig 5: Discovering Physics inversion
```

## Environment Setup

```bash
pip install -r requirements.txt
```

System-level dependencies (not pip; see comments in requirements.txt):
- `xelatex` (to compile the report PDF; optional)
- `poppler-utils` (to check PDF page count; optional)

> Note: the plotting scripts are fully English; no CJK font is required.

## Run Order

### 1. Algorithm self-check (seconds)

```bash
cd code
python at_spatial_ms_moe.py
# Output: parameter count, forward output shape and sample values
```

### 2. 12-algorithm training and evaluation (about 1 minute, CPU)

```bash
cd code
python train_12algos_11elements_predvref.py
# Output: output/full_spectrum_12algos_11elements.json
#         output/pointwise_predictions_11elements.json
# Prints: 12-algorithm RMSE/R^2, per-scale R^2, and each algorithm's own-prediction-vs-reference P value / Pearson r
```

> Note: the script generates its own data (`make_data`) and ground truth (`pde_theory`); no external dataset is needed.
> Training is done algorithm by algorithm. If memory is limited, reduce `N_TRAIN`.

### 3. Figure generation

```bash
cd code
# Put the files from data/ into code/output/ (the script reads from the relative path output/)
mkdir -p output
cp ../data/pointwise_predictions_11elements.json output/
cp ../data/inversion_curves.npz output/
python make_figures_predvref.py
# Generates fig1/fig2/fig3/fig4/fig5, matching the files in figures/
```

## Key Conventions

- **Nutrient vector**: 11 elements `M = [N,P,K | Mg,Ca,S | Fe,Mn,Zn,Cu,B]`
- **Statistical convention**: paired test of each algorithm's **own prediction vs its own reference (ground truth)**; **NOT** a cross-algorithm comparison
- **Metrics**: RMSE, R^2, Pearson r, per-scale R^2, paired t P value, Wilcoxon P value

## Results at a Glance

| Algorithm | RMSE | R^2 | Pearson r |
|---|---|---|---|
| AT-Spatial-MS-MoE | 0.0302 | 0.9753 | 0.9876 |
| Spatial-MS (runner-up) | 0.1602 | 0.3061 | 0.5552 |
| Other algorithms | 0.16-0.18 | 0.13-0.29 | 0.38-0.67 |
