"""
AT-Spatial-MS-MoE -- standalone model definition
================================================
Latest implementation, extracted verbatim from

    task_139/full_spectrum_cos_noflux.py      (lines 146-155, 282-300)

so that the architecture can be inspected and re-used without the full
experiment script.

Architecture: multiplicative physical-modality decomposition  u = P(x) * S(env) * G(x, t)

    ExpertP  : spatial component            P(x)       1 -> d -> d -> 1
    ExpertS  : environmental modulation     exp(.)     14 -> d -> d -> 1   (positivity by exp)
    ExpertG  : growth-accumulation mode     G(x, t)     1-dim x embedding + 9-dim time embedding,
                                                      2 x Block (LayerNorm + gated residual), d -> 1

    u = P(x) * S(env) * G(x, offset)

This differs from the earlier "one expert per time scale" design, which is
deprecated: the experts are now split by PHYSICAL MODALITY (P / S / G), while
the five time scales (minute/hour/day/month/year) remain only as multi-scale
task heads and per-scale evaluation dimensions.

The environmental vector `env` has 14 columns, assembled in build_dict() as

    env = [ T , H , M(11 trace elements) , phi ]        ->  1 + 1 + 11 + 1 = 14

Sign note: only ExpertS is constrained positive (exp). P and G are free and
may be negative; the product P*S*G is therefore NOT guaranteed non-negative.
In practice the network fits the non-negative reference field by letting the
P and G factors take matching signs. This is the behaviour of the shipped
model (it attains R^2 = 0.9667); no positivity projection is applied.

Verification (unified benchmark, 12-algorithm comparison)
    R^2 = 0.9667     per-scale R^2 all > 0.94     (best of the 12 algorithms)
    reference: task_139/output/full_spectrum_12algos_11elements.json

Usage
    python3 at_spatial_ms_moe.py            # instantiate + forward/backward smoke test
"""
import torch
import torch.nn as nn

D = 64          # hidden width used throughout the experiments
N_ENV = 14      # T, H, 11 trace elements, phi


# ---------------------------------------------------------------- building block
class Block(nn.Module):
    """Pre-norm block with gated residual, identical to the shared Block class."""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.p = nn.Linear(d, 2 * d)
        self.sc = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.a = nn.SiLU()

    def forward(self, u):
        z = self.ln(u)
        b, c = self.p(z).chunk(2, -1)
        h = self.a(self.sc(z))
        h = self.o(h * self.a(c))
        return u + h * torch.sigmoid(b)


# ---------------------------------------------------------------- experts
class ExpertP(nn.Module):
    """Spatial component P(x)."""
    def __init__(self, d=D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x)


class ExpertS(nn.Module):
    """Environmental modulation S(env); exponentiated for positivity."""
    def __init__(self, d=D, n_env=N_ENV):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_env, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 1))

    def forward(self, env):
        return torch.exp(self.net(env))


class ExpertG(nn.Module):
    """Growth-accumulation mode G(x, t_rel)."""
    def __init__(self, d=D):
        super().__init__()
        self.x_enc = nn.Linear(1, d)
        self.t_enc = nn.Sequential(
            nn.Linear(9, d), nn.SiLU(), nn.Linear(d, d))
        self.b = nn.ModuleList([Block(d) for _ in range(2)])
        self.o = nn.Linear(d, 1)

    def forward(self, x, offset):
        h = self.x_enc(x) + self.t_enc(offset)
        for bl in self.b:
            h = bl(h)
        return self.o(h)


# ---------------------------------------------------------------- the model
class ATSpatialMSMoE(nn.Module):
    """AT-Spatial-MS-MoE: multiplicative P * S * G decomposition."""
    def __init__(self, d=D):
        super().__init__()
        self.P = ExpertP(d)
        self.S = ExpertS(d)
        self.G = ExpertG(d)

    def forward(self, x, offset, anchor=None, span=None, env=None):
        u = self.P(x) * self.S(env) * self.G(x, offset)
        # The variance head is intentionally inactive (returns zeros), so the
        # NLL term reduces to a plain squared error on the mean.
        return u, torch.zeros_like(u)


# ---------------------------------------------------------------- smoke test
if __name__ == '__main__':
    torch.manual_seed(0)
    n = 8
    m = ATSpatialMSMoE()

    x = torch.rand(n, 1)
    offset = torch.rand(n, 9)          # 1 log-time + 4freq x 2 sin/cos
    env = torch.rand(n, N_ENV)
    target = torch.rand(n, 1)

    mu, lv = m(x, offset, env=env)
    loss = 0.5 * (lv + (target - mu) ** 2 / (torch.exp(lv) + 1e-6)).mean()
    loss.backward()

    n_par = sum(p.numel() for p in m.parameters())
    print('=' * 62)
    print('AT-Spatial-MS-MoE  smoke test')
    print('=' * 62)
    print('  parameters        : %d' % n_par)
    print('  P branch          : %d' % sum(p.numel() for p in m.P.parameters()))
    print('  S branch          : %d' % sum(p.numel() for p in m.S.parameters()))
    print('  G branch          : %d' % sum(p.numel() for p in m.G.parameters()))
    print('  forward out shape : %s' % (tuple(mu.shape),))
    print('  output sign       : %s (unconstrained by design)' % ('all >0' if bool((mu>0).all().item()) else 'contains <=0'))
    print('  loss              : %.6f' % loss.item())
    print('  grad reached P    : %s' % (m.P.net[0].weight.grad is not None))
    print('  grad reached S    : %s' % (m.S.net[0].weight.grad is not None))
    print('  grad reached G    : %s' % (m.G.x_enc.weight.grad is not None))
