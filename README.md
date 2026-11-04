# Ginseng-Land Amelioration Agent Platform — README 

**Version**: v3.0 | **Date**: 2026-10-29
**Title**: Physics-informed neural network solution of a multi-factor microbial growth PDE based on a no-flux self-consistent cosine basis and depth-explicit diffusivity


---

## 1. Overview

This project addresses **continuous-cropping obstacles in ginseng cultivation (ginseng-land amelioration)** by building a technical framework that combines physical constraints, multi-factor coupling and intelligent solvers:

```
Soil environment data (T, H, 11 elements, φ)  ─┐
                                                ├─→  Multi-factor microbial growth PDE  ─→  PINN  ─→  Pathogen control decisions
Absolute time / multi-scale delays τ_k        ─┘          (AT-Spatial-MS-MoE)
```

**Core contribution**: under an identical PDE, identical training budget (1000 epochs) and identical loss, we propose a multiplicative mixture-of-experts architecture partitioned by **physical modality**, raising overall $R^2$ from the baseline 0.498 to **0.9667**.

---

## 2. Model System

### 2.1 Main Model: Multi-factor Microbial Growth PDE

$$u(x,t_{rel})=(1+\beta x^2)\cdot\underbrace{f_T f_H f_M f_\phi}_{\text{env\_mod}}\cdot\Big(\prod_{k=1}^{5}\mu_k\Big)^{1/5}\cdot\sum_{j}\sum_{k}b_{jk}\cos\big((j{+}1)\pi x\big)\sigma\Big(\ln(1+\tfrac{t_{rel}}{\tau_k})-1\Big)$$

| Component | Content | Basis |
|---|---|---|
| Spatial basis | $\cos(j\pi x)$, $j\ge2$ | **Analytically satisfies no-flux** (derivative vanishes at $x=0,1$) |
| Diffusivity | Explicit $D_{eff}(x)$, smooth 30 cm interface | $D_{up}=1.0\times10^{-10} > D_{dn}=3.0\times10^{-11}$ m²/s |
| Temperature | Ratkowsky square-root form | $T_{\min}$=2, $T_{\max}$=35 °C |
| Humidity | Gaussian bell | $H_{opt}$=0.5, $\sigma_H$=0.25 |
| 11 elements | Liebig minimum factor + Monod | N,P,K \| Mg,Ca,S \| Fe,Mn,Zn,Cu,B |
| Porosity | Gaussian bell | $\phi_{opt}$=0.5, $\sigma_\phi$=0.2 |
| Time scales | 5 delays $\tau_k$ | minute/hour/day/month/year |

### 2.2 Best Algorithm: AT-Spatial-MS-MoE

**Multiplicative decomposition by physical modality** (not by time scale):

$$u = P(x)\cdot S(\text{env})\cdot G(x,t)$$

| Expert | Component | Structure | Parameters |
|---|---|---|---|
| **P** | Spatial | `1→64→64→1` | 4,353 |
| **S** | Environmental modulation | `14→64→64→1` + `exp` | 5,185 |
| **G** | Growth accumulation | x-embedding + 9-dim time embedding → 2×Block | 38,529 |
| Total | — | — | **48,067** |

Source: `at_spatial_ms_moe.py` (standalone, runnable)

### 2.3 Case Validation: *Pseudomonas* Suppressing *Fusarium*

A three-variable delay differential equation used to validate the design principles of biocontrol agents:

$$\frac{dF}{dt}=r_FF\big(1-\tfrac{F}{K_F}\big)-g(P,A)F,\quad \frac{dP}{dt}=r_PP\big(1-\tfrac{P}{K_P}\big)h(F)-\gamma AP,\quad \frac{dA}{dt}=\alpha P(t-\tau)-\delta A$$

**Hopf bifurcation** at $\tau_c\approx9.0$ d: beyond this delay the system shifts from a stable equilibrium to sustained oscillation.

---

## 3. Key Revisions

| # | Item | Previous | Current | Effect |
|---|---|---|---|---|
| 1 | Spatial basis | $\sin((j{+}1)\pi x)$ (implicit Dirichlet) | $\cos(j\pi x)$, $j\ge2$ | Boundary derivative 14.4/45.3 → **0** |
| 2 | Boundary treatment | Implicit Dirichlet (zero at both ends) | No-flux soft constraint $\lambda_{bc}$=0.1 | Surface value $\|G(0)\|$=0.732 |
| 3 | Diffusivity | Constant | Explicit $D_{eff}(x)$ | Flux continuity at 30 cm |
| 4 | Environmental factors | N/P/K only | **11 elements** + normalisation | RMSE 14.33 → 0.094 |
| 5 | Expert partition | By time scale (deprecated) | **By physical modality P·S·G** | $R^2$ 0.9753 → 0.9667 (old basis) |

> Independent verification: the cosine basis yields **5** spatial extrema, attaining the theoretical upper limit of five basis functions.

---

## 4. Benchmark Results

### 4.1 Overall Comparison of 12 Algorithms (unified 1000 epochs)

| Rank | Algorithm | RMSE | $R^2$ |
|---|---|---|---|
| 🥇 | **AT-Spatial-MS-MoE** | **0.0940** | **0.9667** |
| 2 | Spatial-MS (baseline) | 0.3651 | 0.4981 |
| 3 | Standard PINN | 0.3692 | 0.4867 |
| 4 | Causal PINN | 0.3717 | 0.4797 |
| 5 | CG-PINN | 0.3777 | 0.4628 |
| 6 | TX-Spatial-MS | 0.3833 | 0.4468 |
| 7 | SiGMA-G | 0.3980 | 0.4036 |
| 8 | HMformer | 0.3994 | 0.3995 |
| 9 | msMamba | 0.4037 | 0.3864 |
| 10 | Fourier-Spatial | 0.4177 | 0.3433 |
| 11 | Singe head AT-Spatial-MS | 0.5192 | −0.0149 |
| 12 | Temporal-MS | 0.5207 | −0.0207 |

### 4.2 Per-scale $R^2$ (AT-Spatial-MS-MoE)

| minute | hour | day | month | year |
|---|---|---|---|---|
| 0.9424 | 0.9656 | **0.9801** | 0.9650 | 0.9606 |

**All scales exceed 0.94**; performance does not rely on any single scale.

### 4.3 The Two Negative-$R^2$ Algorithms

| Algorithm | Failure mechanism |
|---|---|
| **Temporal-MS** | Only 966 parameters (2.8% of SiGMA-G) + three independent experts with no sharing + hard gating leaving $e_0$ at a 2.6% weight (dead parameters) |
| **Singe head AT-Spatial-MS** | An $x$-independent additive bias branch offers a "short path"; gradients flow preferentially into the shortcut and the backbone's spatial response is flattened |


---

## 5. Dataset

| Item | Content |
|---|---|
| Size | 5 scales × (400 train + 100 test) = **2500 points** |
| Fields | 19 columns (x, t_rel, t_abs, T, H, 11 elements, phi, u) |
| Reproduction seeds | training `61000+s`, test `60000+s` |
| Data dictionary | `dataset/README_dataset_dictionary_EN.md` |

---

## 6. Figure Inventory

| Figure | File | Source |
|---|---|---|
| Figure 1 Spatial basis comparison | `fig_g_basis_comparison.png` | `make_report_figures.py` |
| Figure 2 $D_{eff}(x)$ profile | `fig_deff_profile.png` | `make_report_figures.py` |
| Figure 3 Cross-validation | `fig_cross_validation.png` | `make_fig_cross_validation_en.py` |
| Figure 4 $R^2$ comparison | `fig_ns_r2_comparison.png` | `make_report_figures.py` |
| Figure 5 Per-scale heatmap | `fig_ns_scale_r2_heatmap.png` | `make_scale_r2_heatmap.py` |
| Figure 10 Hopf bifurcation | `fig_hopf_bifurcation.png` | `make_hopf_bifurcation.py` |
| Figure 11 Limit-cycle portrait | `fig_limit_cycle_FP.png` | `make_limit_cycle_figures.py` |
| Figure 12 Oscillation time series | `fig_oscillation_timeseries.png` | `make_limit_cycle_figures.py` |
| Figure 13 τ–μ bifurcation map | `fig_bifurcation_tau_mu.png` | `make_bifurcation_tau_mu.py` |
| Figure 14 Four-parameter role map | `fig_four_parameter_roles.png` | `make_four_parameter_roles.py` |

> All figures: English labels, 300 dpi, with standalone runnable source code.

---

## 7. Reproduction

```bash
# 1. Train and evaluate the 12 algorithms (~5 min)
python3 full_spectrum_cos_noflux.py

# 2. Export the dataset
SRC_DIR=. python3 export_dataset.py

# 3. Generate figures
python3 make_report_figures.py
python3 make_scale_r2_heatmap.py
python3 make_hopf_bifurcation.py
python3 make_four_parameter_roles.py
```

**Unified configuration**: `epochs=1000, lr=1e-3, seed=0, D=64, λ_bc=0.1`

---

## 8. Known Limitations (read before use)

| # | Item | Status | Impact |
|---|---|---|---|
| 1 | **Reference field $u$ may be negative** | 46.8% (935/2000) negative, range [−1.74, 3.74] | Regression training is unaffected; a positivity constraint is required for non-negative output |
| 2 | **Temperature out of range** | 7.64% of points have $T<2$ °C, where $u$=0 | Report these separately as a "stagnation zone" in stratified evaluation |
| 3 | **Missing geometric-mean factor in $G$** | Code does not explicitly apply $\left(\prod\mu_k\right)^{1/5}$; report Eq. (5) includes it | Equivalent if $\mu_k$ is constant (already absorbed into $b_{jk}$)|
| 4 | **Dual porosity conventions** | Main model bell-shaped (0.45→0.65→0.45); Figure 3 step-type (0.60→0.45) | Serve different purposes; do not mix |
| 5 | **Parameter-set dependence** | $\tau_c$, dose capacity and window width all derive from the current parameter set | Must be recomputed after calibration against measured data |

---

## 9. File Index

### Core code
| File | Description |
|---|---|
| `full_spectrum_cos_noflux.py` | Main experiment script (12 algorithms + data generation + training) |
| `at_spatial_ms_moe.py` | Standalone definition of the best algorithm |
| `export_dataset.py` | Dataset export |


### Data
`dataset/` (2500 points + result data), `output/` (12-algorithm results), `metrics_src/` (metric summaries)

---

## 10. Citation

If you use this work, please cite:

> Ginseng-Land Amelioration Agent Platform Group. Physics-informed neural network solution of a multi-factor microbial growth PDE based on a no-flux self-consistent cosine basis and depth-explicit diffusivity. Technical Report, 2026.



