"""
Limit-cycle phase portrait and oscillatory time series
======================================================
Pseudomonas (P) suppressing Fusarium (F) via a delayed negative-feedback loop.

Model: three-variable delay differential equation (DDE)
    dF/dt = r_F*F*(1 - F/K_F) - g(P, A)*F          Fusarium: logistic growth - killing
    dP/dt = r_P*P*(1 - P/K_P)*h(F) - gamma*A*P      Pseudomonas: growth promoted by F - self-inhibition
    dA/dt = alpha*P(t - tau) - delta*A              antifungal: delayed secretion - first-order decay

Feedback loop
    F --(+)--> P   (colonisation / chemotaxis toward Fusarium)
    P --(delayed)--> A  (antifungal secretion, delay tau)
    A --(-)--> F   (killing)   and   A --(-)--> P  (self-inhibition)
Hence F -> P -> A -> F is a negative feedback loop. The delay tau destabilises the
interior equilibrium through a Hopf bifurcation, producing a stable limit cycle for
tau >= tau_c (numerically tau_c ~ 8-10 d; amplitude saturates for tau >= 12 d).

NOTE ON OSCILLATIONS
    The non-delayed pathway ODE model (siderophore + antifungal, used for the
    inhibition-rate figures) converges to a steady state and does NOT oscillate.
    Oscillations and a genuine limit cycle require the delay term implemented here.

Outputs
    fig_limit_cycle_FP.png        phase portrait F vs P with the limit cycle
    fig_oscillation_timeseries.png oscillatory F, P and A versus time
Both figures use English labels only.

Usage
    python3 make_limit_cycle_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- parameters
r_F, K_F = 0.30, 1.0
r_P, K_P = 0.45, 1.0
mu, Km, Ka = 0.60, 0.20, 0.15       # inhibition response
gamma = 0.25                        # antifungal self-inhibition on Pseudomonas
Kf_p = 0.30                         # half-saturation of colonisation promotion
alpha, delta = 0.25, 0.15           # antifungal secretion / decay

TAU = 15.0                          # delay (d) -- inside the limit-cycle regime
Y0 = (0.9, 0.3, 0.2)                # initial state, displaced from equilibrium
T_END = 400.0
DT = 0.05


def h(F):
    """Colonisation promotion of Pseudomonas by Fusarium."""
    return F / (Kf_p + F)


def inhibition(P, A):
    """Dual-saturating killing term exerted on Fusarium."""
    return mu * (P / (Km + P)) * (A / (Ka + A))


def equilibrium():
    """Interior equilibrium (numerically solved)."""
    def eq(v):
        F, P, A = v
        g = inhibition(P, A)
        return [r_F * F * (1 - F / K_F) - g * F,
                r_P * P * (1 - P / K_P) * h(F) - gamma * A * P,
                alpha * P - delta * A]
    return fsolve(eq, [0.3, 0.5, 0.5])


def solve_dde(y0=Y0, tau=TAU, T=T_END, dt=DT):
    """Forward Euler integration of the DDE with constant delay tau."""
    n = int(T / dt)
    t = np.zeros(n + 1)
    Y = np.zeros((3, n + 1))
    Y[:, 0] = np.array(y0, dtype=float)
    lag = max(1, int(round(tau / dt)))
    for i in range(n):
        F, P, A = Y[:, i]
        g = inhibition(P, A)
        dF = r_F * F * (1 - F / K_F) - g * F
        dP = r_P * P * (1 - P / K_P) * h(F) - gamma * A * P
        Pd = Y[1, 0] if i < lag else Y[1, i - lag]
        dA = alpha * Pd - delta * A
        Y[:, i + 1] = np.maximum(
            np.array([F, P, A]) + dt * np.array([dF, dP, dA]), 0.0)
        t[i + 1] = t[i] + dt
    return t, Y


def cycle_amplitude(t, Y, t_from=0.75):
    """Peak-to-peak amplitude over the settled portion of the trajectory."""
    m = t > t_from * t.max()
    return Y[0, m].max() - Y[0, m].min()


# ---------------------------------------------------------------- figure 1
def make_phase_portrait(t, Y, eq, out='fig_limit_cycle_FP.png'):
    F, P = Y[0], Y[1]

    # steady portion only, to draw the closed orbit cleanly
    m = t > 0.5 * t.max()
    Fs, Ps = F[m], P[m]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    # transient (faded) and settled orbit (solid)
    mt = t <= 0.5 * t.max()
    ax.plot(F[mt], P[mt], color='#bdc3c7', lw=1.2, ls=':',
            label='Transient approach')
    ax.plot(Fs, Ps, color='#2e5aa8', lw=2.2, label='Limit cycle (settled)')

    ax.plot(eq[0], eq[1], marker='*', ms=16, color='#c0392b', zorder=5,
            label=f'Unstable equilibrium\n($F^*$={eq[0]:.3f}, $P^*$={eq[1]:.3f})')
    ax.plot(Y0[0], Y0[1], marker='o', ms=8, color='#27ae60', zorder=5,
            label='Initial state')

    ax.set_xlabel('Fusarium density $F$', fontsize=12)
    ax.set_ylabel('Pseudomonas density $P$', fontsize=12)
    ax.set_title('Limit cycle in the $F$–$P$ plane ($\\tau$ = %.0f d)' % TAU,
                 fontsize=12.5, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='best', framealpha=0.92)

    amp = cycle_amplitude(t, Y)
    ax.text(0.03, 0.03, f'Peak-to-peak $F$ amplitude = {amp:.3f}',
            transform=ax.transAxes, fontsize=10, color='#2e5aa8',
            fontweight='bold')

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return out


# ---------------------------------------------------------------- figure 2
def make_timeseries(t, Y, eq, out='fig_oscillation_timeseries.png'):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)

    specs = [(0, 'Fusarium $F$', '#c0392b', eq[0]),
             (1, 'Pseudomonas $P$', '#2e5aa8', eq[1]),
             (2, 'Antifungal $A$', '#8e44ad', eq[2])]

    for ax, (idx, ylab, color, eqv) in zip(axes, specs):
        ax.plot(t, Y[idx], color=color, lw=1.6)
        ax.axhline(eqv, color='k', ls='--', lw=1.0, alpha=0.6,
                   label=f'Equilibrium = {eqv:.3f}')
        ax.set_ylabel(ylab, fontsize=11.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel('Time (d)', fontsize=12)
    axes[0].set_title(
        'Sustained oscillations of the Pseudomonas–Fusarium system '
        '(delay $\\tau$ = %.0f d)' % TAU,
        fontsize=12.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return out


if __name__ == '__main__':
    eq = equilibrium()
    t, Y = solve_dde()
    p1 = make_phase_portrait(t, Y, eq)
    p2 = make_timeseries(t, Y, eq)

    print('=' * 70)
    print('Delay = %.1f d,  T = %.0f d,  dt = %.2f' % (TAU, T_END, DT))
    print('=' * 70)
    print('Interior equilibrium : F* = %.4f, P* = %.4f, A* = %.4f' % tuple(eq))
    amp = cycle_amplitude(t, Y)
    print('Settled F amplitude  : %.4f  (peak-to-peak)' % amp)
    print('F range in cycle     : [%.4f, %.4f]' % (Y[0][t > 0.75 * t.max()].min(),
                                                   Y[0][t > 0.75 * t.max()].max()))
    print('\nSaved: %s (%d KB)' % (p1, os.path.getsize(p1) / 1024))
    print('Saved: %s (%d KB)' % (p2, os.path.getsize(p2) / 1024))
