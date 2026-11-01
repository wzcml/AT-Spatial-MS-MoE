# Dataset for the Latest Formulation — Data Dictionary and Download Guide

**Version**: v1.0 | **Date**: 2026-10-29
**Formulation**: cosine spatial basis (analytically no-flux) + explicit $D_{eff}(x)$ + 11-element environmental modulation
**Source script**: `full_spectrum_cos_noflux.py` (fixed seeds, exactly reproducible)

---

## 1. Dataset Overview

| Item | Content |
|---|---|
| Time scales | 5: minute / hour / day / month / year |
| Training set | 400 points per scale, **2000** in total |
| Test set | 100 points per scale, **500** in total |
| Columns | 19 (including the `scale` label) |
| Reproduction seeds | training `61000 + s`, test `60000 + s` ($s$ = 0..4) |

### File Inventory

| File | Rows | Description |
|---|---|---|
| `train_scale0_minute.csv` … `train_scale4_year.csv` | 400 × 5 | Training set, one file per scale |
| `test_scale0_minute.csv` … `test_scale4_year.csv` | 100 × 5 | Test set, one file per scale |
| `all_data.npz` | 30 arrays | All data bundled (x / t_rel / u per scale) |
| `dataset_manifest.json` | — | Shapes, seeds and SHA256 checksums |

---

## 2. Field Dictionary

| Column | Meaning | Unit | Range |
|---|---|---|---|
| `scale` | Time-scale label | — | minute/hour/day/month/year |
| `x` | **Normalised depth** | dimensionless $[0,1]$ | 0 = soil surface, 1 = deep layer |
| `t_rel` | Elapsed time since the anchor | **s** | minute 60–3600; year 3.15×10⁷–1.89×10⁸ |
| `t_abs` | Absolute timestamp | **s** (unix) | — |
| `T` | Temperature | **°C** | **−0.98 – 31.0** (see Section 4.2) |
| `H` | Relative humidity | dimensionless | 0–1 |
| `N`,`P`,`K` | Macro nutrients | dimensionless (normalised) | 0–1 |
| `Mg`,`Ca`,`S` | Secondary nutrients | dimensionless (normalised) | 0–1 |
| `Fe`,`Mn`,`Zn`,`Cu`,`B` | Micro nutrients | dimensionless (normalised) | 0–1 |
| `phi` | **Porosity** | **dimensionless volume fraction** (m³/m³) | 0.45–0.65 |
| `u` | **Reference microbial density** | dimensionless | **−1.74 – 3.74 (may be negative, see Section 4)** |

> **Dimensional note**: `phi` is a dimensionless volume fraction and is **not** expressed in cm. `x` is a normalised depth (dimensionless); converting it to a physical depth requires multiplication by the total depth (the 30 cm interface lies at $x=0.6$).

---

## 3. Environmental Factor Model

$$u = (1+0.6x^2)\cdot \underbrace{f_T(T)f_H(H)f_M(\mathbf{M})f_\phi(\phi)}_{\text{env\_mod}}\cdot G(x,t)$$

| Factor | Model | Parameters |
|---|---|---|
| Temperature | Ratkowsky (square-root type) | $T_{\min}$=2, $T_{\max}$=35, $C$=0.25, normalised to peak |
| Humidity | Gaussian bell | $H_{opt}$=0.5, $\sigma_H$=0.25 |
| 11 elements | Liebig minimum factor + Monod | $K_{EL}$=[0.32,0.26,0.34, 0.30,0.36,0.33, 0.17,0.10,0.12,0.07,0.09] |
| Porosity | Gaussian bell | $\phi_{opt}$=0.5, $\sigma_\phi$=0.2 |

**Element order** (11 elements): `N,P,K | Mg,Ca,S | Fe,Mn,Zn,Cu,B`

### Porosity Depth Profile

$$\phi(x)=\mathrm{clip}\big(0.45+0.2\sin(\pi x),\,0.15,\,0.85\big)$$

**Surface $x$=0 → 0.45, peak $x$=0.5 → 0.65, bottom $x$=1 → 0.45** (bell-shaped)

> ⚠️ Note: this is the porosity used in the main PDE model. The cross-validation figure (Figure 3) employs a different **step-type** profile (0.60 at the surface → 0.45 at depth). The two conventions differ and must not be mixed.


## 4. ⚠️ Important: Nearly Half of the Reference Field $u$ Is Negative

### Measured Statistics

| Scale | Fraction with $u<0$ | min $u$ | max $u$ |
|---|---|---|---|
| minute | 43.5% | −1.206 | 1.249 |
| hour | 42.5% | −1.491 | 1.930 |
| day | 46.5% | −1.537 | 2.488 |
| month | 50.5% | −1.591 | 3.421 |
| year | 50.8% | −1.738 | 3.739 |
| **Total** | **46.8% (935/2000)** | — | — |

### Cause

`env_mod` ∈ [0,1] and $(1+0.6x^2)>0$ are always positive, so the sign of $u$ is **determined entirely by $G$**. The `_BJK` coefficients carry signs (e.g. −0.554, −0.691, −0.965); when the cosine terms superpose in negative phase, $G<0$ and hence $u<0$.

### Recommended Use

| Purpose | Permitted |
|---|---|
| Training / evaluating a PINN (regression task) | ✅ Yes; the reported $R^2$=0.9667 was obtained on exactly this data |
| Physical modelling requiring non-negative output | ❌ Requires a positivity constraint or $u \leftarrow \max(u,0)$ |
| Cross-comparison with other datasets | ⚠️ Must state that the reference field may be negative |



## 4.2 ⚠️ Temperature Out of Range: 7.64% of Points Have $T<2\,°C$

### Measurements

| Item | Value |
|---|---|
| Global $T$ range | **−0.976 – 30.994 °C** |
| Points with $T<2\,°C$ | **191 / 2500 (7.64%)** |
| Points with $T>35\,°C$ | 0 |

### Cause and Consequence

Temperature is the superposition of a seasonal term and a diurnal term, **without lower clipping**:

$$T(t)=T_{mean}+T_{amp}\cos\!\left(\tfrac{2\pi(\mathrm{doy}-\mathrm{peak})}{365.24}\right)+T_{diurnal}\cos(2\pi t_{day})$$

The Ratkowsky growth function is

$$f_T=\Big[\mathrm{clip}\big((T-2)(1-e^{0.25(T-35)}),\,0,\,+\infty\big)\Big]^2\Big/ f_T^{\max}$$

For $T<2\,°C$, $(T-2)<0$, so after clipping $f_T=0$ → `env_mod`=0 → **$u=0$**.

**Verified**: the 31 points with $T<2$ in the day scale all have $u$ strictly equal to 0.

### Implications for Evaluation

These 191 points are **zero-valued points (stagnation at the biological zero)**, physically reasonable but affecting the statistics:

| Effect | Description |
|---|---|
| $R^2$ denominator | Increases SST; the effect on $R^2$ is comparatively mild |
| RMSE | Amplified if the model predicts non-zero values at these points |
| $A_f^{\mathrm{rob}}$ | The denominator $|\hat u|+|u|+\delta$ saturates at $u$=0, **which is precisely why $|u|$ weighting is required** |

**Recommendation**: in stratified evaluation, report points with $T<2\,°C$ as a separate "stagnation zone" rather than pooling them with the growth zone.

---

## 5. Reproduction

```bash
# 1. Export the data (regenerated from the source script; identical to training time)
SRC_DIR=/path/to/task_139 python3 export_dataset.py

# 2. Verify
python3 -c "
import pandas as pd
df = pd.read_csv('dataset/train_scale2_day.csv')
print(df.shape, df.u.min(), df.u.max())
"
```

**Seed protocol** (must not be altered):
- Training: `make_data(s, 400, seed=61000+s)`
- Test: `make_data(s, 100, seed=60000+s)`

## 6. Companion Result Data (provided separately)

This dataset constitutes the **model input**. The corresponding **model output / results** are:

| File | Content |
|---|---|
| `output/full_spectrum_12algos_11elements.json` | Overall RMSE / R² / per-scale R² / runtime for the 12 algorithms |
| `output/pointwise_predictions_11elements.json` | Pointwise predictions and ground truth for the 12 algorithms (100 points per scale) |
| `metrics_src/table_self_vs_moe.csv` | Summary of RMSE / R² / r / Af / Bf / P for the 12 algorithms |
| `g_basis_validation.csv` | Spatial-basis validation (extrema count, boundary derivatives) |
| `table_new_vs_old.csv` | Comparison of the new and old formulations (sin vs cos) |
| `best_B.npy` | 5×5 coefficient matrix |
| `multi_pathway_results.npz` | Time series of the 18-parameter *Pseudomonas* case model |
