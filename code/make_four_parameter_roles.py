"""
Four-parameter role map: what each knob actually controls
=========================================================
Companion figure to the design guide, summarising in one place:

    tau   -> whether the system is STABLE or oscillates
    P0    -> how FAST suppression is achieved (not how deep the final state is)
    gamma -> how HIGH a dose can be tolerated before Pseudomonas self-collapses
    mu    -> how WIDE the process window is (tau_c moves)

Model (three-variable DDE)
    dF/dt = r_F*F*(1 - F/K_F) - g(P, A)*F
    dP/dt = r_P*P*(1 - P/K_P)*h(F) - gamma*A*P
    dA/dt = alpha*P(t - tau) - delta*A
    g(P, A) = mu * P/(Km+P) * A/(Ka+A)

All four panels are computed from the same integrator, so the numbers are
mutually consistent.

Panels
    (a) Amplitude of F vs tau          -> stability       (tau = binary switch)
    (b) Time to reach F < 0.2 vs P0    -> speed           (P0 = rate knob)
    (c) Collapse threshold P0* vs gamma-> dose tolerance  (gamma = capacity)
    (d) Hopf boundary tau_c vs mu      -> window width    (mu = tolerance width)

Output
    fig_four_parameter_roles.png   (English labels, 300 dpi)

Usage
    python3 make_four_parameter_roles.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- parameters
r_F, K_F = 0.30, 1.0
r_P, K_P = 0.45, 1.0
Km, Ka = 0.20, 0.15
Kf_p = 0.30
alpha, delta = 0.25, 0.15

DT = 0.05
T_LONG = 1500.0
FRAC_TRANS = 0.78


def h(F):
    return F / (Kf_p + F)


def run(mu, tau, P0, gamma, T=T_LONG, dt=DT):
    """Integrate the DDE; returns (t, F, P, A)."""
    n = int(T / dt)
    t = np.zeros(n + 1)
    Y = np.zeros((3, n + 1))
    Y[:, 0] = np.array([0.8, P0, 0.0])
    lag = max(1, int(round(tau / dt)))
    for i in range(n):
        F, P, A = Y[:, i]
        g = mu * (P / (Km + P)) * (A / (Ka + A))
        dF = r_F * F * (1 - F / K_F) - g * F
        dP = r_P * P * (1 - P / K_P) * h(F) - gamma * A * P
        Pd = Y[1, 0] if i < lag else Y[1, i - lag]
        dA = alpha * Pd - delta * A
        Y[:, i + 1] = np.maximum(
            np.array([F, P, A]) + dt * np.array([dF, dP, dA]), 0.0)
        t[i + 1] = t[i] + dt
    return t, Y[0], Y[1], Y[2]


def amplitude(mu, tau, P0=0.30, gamma=0.25):
    t, F, _, _ = run(mu, tau, P0, gamma)
    m = t > FRAC_TRANS * t.max()
    return float(np.ptp(F[m]))


def time_to_threshold(mu, tau, P0, gamma=0.25, thr=0.2):
    t, F, _, _ = run(mu, tau, P0, gamma, T=400.0)
    idx = np.where(F < thr)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def collapse_threshold(mu, tau, gamma, p_floor=5e-3, lo=0.1, hi=6.0, step=0.1):
    """Highest P0 for which Pseudomonas never collapses below p_floor."""
    thresh = 0.0
    for p0 in np.arange(lo, hi + 1e-9, step):
        _, _, P, _ = run(mu, tau, p0, gamma, T=1000.0)
        if P.min() < p_floor:
            break
        thresh = float(p0)
    return thresh


def hopf_tau(mu, tau_lo=2.0, tau_hi=60.0, tol=0.3, thr=5e-3):
    if amplitude(mu, tau_lo) > thr or amplitude(mu, tau_hi) < thr:
        return np.nan
    while tau_hi - tau_lo > tol:
        mid = 0.5 * (tau_lo + tau_hi)
        if amplitude(mu, mid) > thr:
            tau_hi = mid
        else:
            tau_lo = mid
    return 0.5 * (tau_lo + tau_hi)


def main(out='fig_four_parameter_roles.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))

    # ---------------- (a) tau -> stability ----------------
    ax = axes[0, 0]
    taus = np.arange(4, 30.5, 2.0)
    amps = [amplitude(0.60, t) for t in taus]
    ax.plot(taus, amps, 'o-', color='#2e5aa8', lw=2.0, ms=6)
    ax.axvline(15, color='#7d3c98', ls=':', lw=1.8, label=r'$\tau$ = 15 d')
    ax.set_xlabel(r'Delay $\tau$ (d)', fontsize=11.5)
    ax.set_ylabel(r'Amplitude of $F$', fontsize=11.5)
    ax.set_title(r'(a) $\tau$ decides STABILITY', fontsize=12,
                 fontweight='bold')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.5)
    ax.annotate('no oscillation', xy=(6, 0.01), fontsize=9.5, color='#1e8449',
                fontweight='bold')
    ax.annotate('sustained oscillation', xy=(19, 0.55), fontsize=9.5,
                color='#b9770e', fontweight='bold')

    # ---------------- (b) P0 -> speed ----------------
    ax = axes[0, 1]
    p0s = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5])
    tt = [time_to_threshold(0.60, 15.0, p) for p in p0s]
    ax.plot(p0s, tt, 's-', color='#c0392b', lw=2.0, ms=6)
    ax.set_xlabel(r'Inoculum $P_0$ (-)', fontsize=11.5)
    ax.set_ylabel(r'Time to reach $F<0.2$ (d)', fontsize=11.5)
    ax.set_title(r'(b) $P_0$ decides SPEED', fontsize=12,
                 fontweight='bold')
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.text(0.6, 0.90, r'asymptotic amplitude is independent of $P_0$',
            transform=ax.transAxes, fontsize=9.5, color='#7f8c8d',
            style='italic')

    # ---------------- (c) gamma -> dose tolerance ----------------
    ax = axes[1, 0]
    gs = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.35])
    thr = [collapse_threshold(0.60, 15.0, g) for g in gs]
    ax.plot(gs, thr, '^-', color='#8e44ad', lw=2.0, ms=7)
    ax.fill_between(gs, 0, thr, color='#e8daef', alpha=0.6)
    ax.set_xlabel(r'Self-inhibition $\gamma$ (-)', fontsize=11.5)
    ax.set_ylabel(r'Tolerable dose $P_0^{*}$ (-)', fontsize=11.5)
    ax.set_title(r'(c) $\gamma$ decides DOSE CAPACITY', fontsize=12,
                 fontweight='bold')
    ax.grid(alpha=0.25)
    ax.text(0.05, 0.55, 'safe dosing region', fontsize=9.5, color='#6c3483',
            fontweight='bold')
    ax.text(0.05, 0.20, r'above the curve: $P$ collapses -> antifungal '
            'supply lost -> $F$ rebounds', fontsize=9, color='#922b21')

    # ---------------- (d) mu -> window width ----------------
    ax = axes[1, 1]
    mus = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    tcs = [hopf_tau(m) for m in mus]
    ok = ~np.isnan(tcs)
    ax.plot(mus[ok], np.array(tcs)[ok], 'D-', color='#1a5276', lw=2.2, ms=6)
    ax.fill_between(mus[ok], 0, np.array(tcs)[ok], color='#d6eaf8', alpha=0.7)
    ax.set_xlabel(r'Antifungal strength $\mu$ (-)', fontsize=11.5)
    ax.set_ylabel(r'Critical delay $\tau_c$ (d)', fontsize=11.5)
    ax.set_title(r'(d) $\mu$ decides WINDOW WIDTH', fontsize=12,
                 fontweight='bold')
    ax.grid(alpha=0.25)
    ax.text(0.30, 12.0, 'stable for any $\\tau$ below the curve',
            fontsize=9.5, color='#1a5276', fontweight='bold')
    ax.annotate('', xy=(0.80, 6.0), xytext=(0.80, 14.0),
                arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.8))
    ax.text(0.815, 9.5, 'window\nshrinks', fontsize=9, color='#c0392b')

    plt.suptitle('What each design parameter actually controls\n'
                 '($\\tau$ = stability, $P_0$ = speed, $\\gamma$ = dose capacity, '
                 '$\\mu$ = window width)',
                 fontsize=13.5, fontweight='bold', y=1.005)
    plt.tight_layout(h_pad=3.2, w_pad=2.6)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print('=' * 68)
    print('(a) tau -> amplitude')
    for t_, a_ in zip(taus[::3], amps[::3]):
        print('    tau=%.0f d  amp=%.4f' % (t_, a_))
    print('(b) P0 -> time to F<0.2')
    for p_, t_ in zip(p0s, tt):
        print('    P0=%.2f  %.1f d' % (p_, t_))
    print('(c) gamma -> tolerable dose P0*')
    for g_, th_ in zip(gs, thr):
        print('    gamma=%.2f  P0*=%.1f' % (g_, th_))
    print('(d) mu -> tau_c')
    for m_, t_ in zip(mus[ok], np.array(tcs)[ok]):
        print('    mu=%.2f  tau_c=%.1f d' % (m_, t_))
    print('\nSaved: %s (%.0f KB)' % (out, os.path.getsize(out) / 1024))


if __name__ == '__main__':
    main()
