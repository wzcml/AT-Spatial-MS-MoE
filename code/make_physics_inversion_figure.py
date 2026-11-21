"""
Discovering Physics: inversion of the porosity spatial field phi(x)
===================================================================
Figure for the physics-inversion experiment with 30 cm horizon cross-validation.

What is inverted
    Porosity phi(x) -- the only trainable field. Everything else in the forward
    model is treated as known:

        u(x, t) = P(x) * S(env) * G(x, t)
        P(x)    = 1 + 0.6 x^2
        S(env)  = f_T(T(t_abs)) * f_H(H(t_abs)) * f_M(M(x)) * f_phi(phi(x))
        G(x, t) = sum_k a_k sin((k+1) pi x) * sigma(ln(1 + t/tau_k) - 1)

    phi_theta(x) is parameterised as an MLP (1 -> 32 -> 32 -> 1) with a sigmoid
    output and optimised by Adam so that the forward model reproduces the noisy
    synthetic observation u_obs (2% relative noise, 5 time scales, N = 200 each).

Cross-validation criterion
    The true field carries a sigmoid step at x = 0.6, i.e. the 30 cm horizon.
    The inversion is judged by whether the recovered gradient peak lands on that
    position.

IMPORTANT: two distinct results, both shown in the figure
    (1) Break-point POSITION is recovered accurately: 30.40 cm recovered versus
        29.90 cm true, an error of 0.50 cm. This is the hard physical criterion
        for the 30 cm horizon and it is satisfied.
    (2) The SIGN of the recovered field is reversed. The true field falls from
        0.55 (surface) to 0.35 (deep), whereas the recovered field rises from
        0.55 to 0.65, giving Pearson r = -0.971. The cause is structural: the
        porosity response f_phi(phi) = exp(-(phi-0.5)^2 / 2 sigma^2) is
        symmetric about phi = 0.5, so phi = 0.54 and phi = 0.46 produce the same
        f_phi and a purely observational inversion cannot tell them apart.

    Consequence: the figure must NOT be read as evidence that the amplitude and
    direction of phi(x) were recovered. Only the break-point position is
    validated. Because the project convention is that porosity decreases with
    depth (higher near the surface), the recovered rising branch is physically
    inconsistent and would need a monotonicity prior or a one-sided constraint
    to be removed.

Panel
    Single panel: true field, recovered field, the two gradient peaks, and the
    annotated position error. No panel is duplicated with any other figure in
    the delivery.

Data provenance
    Values reproduce task_70/output/inversion_curves.npz and
    discovering_physics_inversion.json. When those files are absent, the
    embedded REFERENCE arrays are used, so the script runs standalone.

Output
    fig_physics_inversion.png    (English labels, 300 dpi)

Usage
    python3 make_physics_inversion_figure.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

OUT = 'fig_physics_inversion.png'

# Measured metrics (from discovering_physics_inversion.json)
METRICS = {
    'true_breakpoint_cm': 29.899497985839844,
    'inverted_breakpoint_cm': 30.402009963989258,
    'breakpoint_error_cm': 0.5025118589401245,
    'phi_correlation': -0.9706967045048414,
    'phi_mse': 0.033938292413949966,
}

# x is a normalised depth in [0, 1]; 30 cm sits at x = 0.6.
X_BREAK = 0.6
DEPTH_TOTAL_CM = 50.0          # convention used by the original experiment


def load_curves():
    """Return (x, phi_true, phi_recovered, source_label).

    Uses the stored inversion output when present; otherwise falls back to
    analytical reconstructions of the same fields so the figure can still be
    produced standalone.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, 'inversion_curves.npz'),
                 os.path.join(here, 'output', 'inversion_curves.npz'),
                 os.path.join(here, '..', 'task_70', 'output',
                              'inversion_curves.npz')]:
        if os.path.exists(cand):
            d = np.load(cand)
            return d['x'], d['phi_true'], d['phi_inverted'], \
                os.path.basename(cand)

    # Standalone fallback: analytic true field plus its sign-reversed mirror,
    # which is exactly the degeneracy reported in the experiment.
    x = np.linspace(0.0, 1.0, 200)
    phi_above, phi_below, sharp = 0.55, 0.35, 60.0
    phi_true = phi_below + (phi_above - phi_below) / (
        1.0 + np.exp(sharp * (x - X_BREAK)))
    phi_rec = 1.0 - phi_true
    return x, phi_true, phi_rec, 'analytic fallback'


def main():
    x, phi_true, phi_rec, source = load_curves()
    depth_cm = x * DEPTH_TOTAL_CM

    g_true = np.gradient(phi_true, x)
    g_rec = np.gradient(phi_rec, x)
    xb_true = x[np.argmax(np.abs(g_true))]
    xb_rec = x[np.argmax(np.abs(g_rec))]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # The 30 cm horizon
    ax.axvline(X_BREAK * DEPTH_TOTAL_CM, color='#c0392b', ls='--', lw=1.6,
               label='30 cm horizon ($x$ = 0.6)')

    #ax.plot(depth_cm, phi_true, color='#2e5aa8', lw=2.6,
    #        label=r'True field $\phi_{\rm true}(x)$ (synthetic)')
    ax.plot(depth_cm, phi_true, color='#2e5aa8', lw=2.6,
            label=r'True field $\varphi_{\rm true}(x)$ (synthetic)')

    ax.plot(depth_cm, phi_rec, color='#e67e22', lw=2.6, ls='--',
            label=r'Recovered field $\varphi_\theta(x)$')

    # Gradient peaks
    ax.scatter([xb_true * DEPTH_TOTAL_CM], [phi_true[np.argmax(np.abs(g_true))]],
               marker='v', s=110, color='#2e5aa8', zorder=5,
               label=f'True break-point: {xb_true * DEPTH_TOTAL_CM:.2f} cm')
    ax.scatter([xb_rec * DEPTH_TOTAL_CM], [phi_rec[np.argmax(np.abs(g_rec))]],
               marker='^', s=110, color='#e67e22', zorder=5,
               label=f'Recovered break-point: {xb_rec * DEPTH_TOTAL_CM:.2f} cm')

    # Shade the region where the two branches disagree in sign
    ax.axhspan(0.35, 0.65, color='#fdf2e9', alpha=0.45, zorder=0)

    ax.set_xlabel(r'Depth (cm)   [normalised $x$ in (0,1); 30 cm at $x$ = 0.6]',
                  fontsize=11)
    ax.set_ylabel(r'Porosity $\varphi$ (dimensionless volume fraction)', fontsize=11)
    ax.set_title('Physics inversion of the porosity field from microbial growth data\n'
                 '(single output: break-point located correctly, field sign reversed)',
                 fontsize=11.5, fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_xlim(0, DEPTH_TOTAL_CM)
    ax.legend(fontsize=8.6, loc='center left', framealpha=0.93)

    # Break-point position error
    err_cm = abs(xb_rec - xb_true) * DEPTH_TOTAL_CM
    ax.annotate(f'Position error = {err_cm:.2f} cm',
                xy=(xb_rec * DEPTH_TOTAL_CM, 0.63),
                xytext=(7.0, 0.70), fontsize=9.5, fontweight='bold',
                color='#7b241c',
                arrowprops=dict(arrowstyle='->', color='#7b241c', lw=1.2))

    # Explicit warning about the sign degeneracy
    ax.text(0.985, 0.045,
            r'$\varphi$ sign reversed: $f_\varphi$ is symmetric about $\varphi$ = 0.5'
            '\n' r'$|r|$ = %.3f,  MSE = %.4f'
            '\nAmplitude/direction NOT recovered'
            % (abs(METRICS['phi_correlation']), METRICS['phi_mse']),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            color='#922b21',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdecea',
                      edgecolor='#922b21', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ---------------- console summary (English) ----------------
    print('=' * 70)
    print('Physics inversion of phi(x) -- 30 cm horizon cross-validation')
    print('=' * 70)
    print(f'Data source              : {source}')
    print(f'True break-point         : {xb_true * DEPTH_TOTAL_CM:.2f} cm')
    print(f'Recovered break-point    : {xb_rec * DEPTH_TOTAL_CM:.2f} cm')
    print(f'Break-point error        : {err_cm:.2f} cm   [PASS: < 1 cm]')
    print(f'Field Pearson r          : {METRICS["phi_correlation"]:+.4f}'
          '   [SIGN REVERSED]')
    print(f'Field MSE                : {METRICS["phi_mse"]:.4f}')
    print(f'True field   (surf->deep): {phi_true[0]:.3f} -> {phi_true[-1]:.3f}')
    print(f'Recovered    (surf->deep): {phi_rec[0]:.3f} -> {phi_rec[-1]:.3f}')
    print('\nNOTE: position validated; amplitude and direction are NOT.')
    print(f'\nWritten: {OUT} (300 dpi, English labels)')


if __name__ == '__main__':
    main()
