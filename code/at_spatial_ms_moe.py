"""
AT-Spatial-MS-MoE core implementation (self-contained, runnable standalone)
========================================================

Full name: Absolute-Time + Spatial Multi-Scale + Mixture-of-Experts
(Chinese: absolute-time constraint + spatial multi-scale + physical-modality MoE)

Technical route (three-part structural prior):
  AT         -- absolute-time triple (anchor / offset / span) injects seasonal & annual cycles
  Spatial-MS -- spatial three-region (root / middle / boundary) decomposition
  MoE        -- physical-modality multiplicative mixture-of-experts P . S . G

Core formula (multiplicative decomposition):
  u(x,t) = P(x) · S(T,H,M,φ) · G(x,t_rel)
    P(x)          : spatial modality (1 + beta*x^2, beta = 0.6)
    S(T,H,M,phi)  : environmental modulation = f_T(Ratkowsky) * f_H(bell) * f_M(Liebig-Monod) * f_phi(bell)
    G(x,t_rel)    : growth accumulation = sum_k a_k * sin((k+1)pi*x) * sigmoid(ln(1+t_rel/tau_k) - 1)

Nutrients (11 elements): M = [N,P,K | Mg,Ca,S | Fe,Mn,Zn,Cu,B]
  macronutrients N/P/K | secondary Mg/Ca/S | micronutrients Fe/Mn/Zn/Cu/B

Dependencies: numpy, torch, scipy
"""

import time
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

# =====================================================================
# 1. Physical model (PDE ground-truth generator)
# =====================================================================
TAU   = [60.0, 3600.0, 86400.0, 86400*30.0, 86400*365.0]   # characteristic time lags: minute/hour/day/month/year
AMPS  = [1.0, 0.8, 0.6, 0.4, 0.25]                         # growth amplitude per scale
ANCHOR_UNIX = 1577836800.0
DAY = 86400.0; YEAR = 86400.0*365.24; HOUR = 3600.0

def day_of_year(ts):
    return ((ts - ANCHOR_UNIX) / DAY) % 365.24

# --- seasonal fields of environmental factors ---
T_MEAN=15.0; T_AMP=12.0; T_DIURNAL=4.0; DAY_PEAK_T=200.0
H_MEAN=0.5; H_AMP=0.2; DAY_PEAK_H=140.0

def temperature(t_abs):
    doy = day_of_year(t_abs); tday = (t_abs % DAY) / DAY
    return T_MEAN + T_AMP*np.cos(2*np.pi*(doy-DAY_PEAK_T)/365.24) + T_DIURNAL*np.cos(2*np.pi*tday)

def humidity(t_abs):
    doy = day_of_year(t_abs)
    return np.clip(H_MEAN + H_AMP*np.cos(2*np.pi*(doy-DAY_PEAK_H)/365.24), 0.05, 1.0)

def trace_elements(x, t_abs):
    doy = day_of_year(t_abs)
    N  = np.clip(0.6+0.3*x+0.1*np.cos(2*np.pi*(doy-120)/365.24), 0.01, 2.0)
    P  = np.clip(0.5+0.2*(1-x)+0.1*np.cos(2*np.pi*(doy-160)/365.24), 0.01, 2.0)
    K  = np.clip(0.7+0.15*np.sin(np.pi*x)+0.08*np.cos(2*np.pi*(doy-200)/365.24), 0.01, 2.0)
    Mg = np.clip(0.45+0.10*(1-x)+0.06*np.cos(2*np.pi*(doy-170)/365.24), 0.01, 1.5)
    Ca = np.clip(0.55+0.20*(1-x)+0.08*np.cos(2*np.pi*(doy-190)/365.24), 0.01, 2.0)
    S  = np.clip(0.35+0.12*x+0.05*np.cos(2*np.pi*(doy-210)/365.24), 0.01, 1.5)
    Fe = np.clip(0.35+0.10*x+0.06*np.cos(2*np.pi*(doy-180)/365.24), 0.005, 1.0)
    Mn = np.clip(0.20+0.05*(1-x)+0.04*np.cos(2*np.pi*(doy-140)/365.24), 0.005, 1.0)
    Zn = np.clip(0.25+0.04*np.sin(np.pi*x)+0.03*np.cos(2*np.pi*(doy-200)/365.24), 0.005, 1.0)
    Cu = np.clip(0.15+0.03*np.sin(np.pi*x)+0.02*np.cos(2*np.pi*(doy-220)/365.24), 0.005, 1.0)
    B  = np.clip(0.12+0.03*np.sin(np.pi*x)+0.02*np.cos(2*np.pi*(doy-150)/365.24), 0.003, 0.8)
    return np.stack([N,P,K,Mg,Ca,S,Fe,Mn,Zn,Cu,B], -1)   # (n, 11)

def porosity(x):
    return np.clip(0.45 + 0.2*np.sin(np.pi*x), 0.15, 0.85)

# --- factors of environmental modulation ---
T_MIN=2.0; T_MAX=35.0; C=0.25
H_OPT=0.5; SIGMA_H=0.25
K_EL = np.array([0.32,0.26,0.34, 0.30,0.36,0.33, 0.17,0.10,0.12,0.07,0.09])  # half-saturation constants of 11 elements
PHI_OPT=0.5; SIGMA_PHI=0.2

def _fT_raw(T):
    return (np.clip((T-T_MIN)*(1-np.exp(C*(T-T_MAX))), 0, None))**2
_FT_MAX = float(_fT_raw(np.linspace(T_MIN, T_MAX, 10001)).max())
def f_T(T):   return _fT_raw(T) / _FT_MAX                        # Ratkowsky square-root type (normalized)
def f_H(H):   return np.exp(-(H-H_OPT)**2/(2*SIGMA_H**2))        # optimal-humidity bell curve
def f_M(M):   return np.min(M/(M+K_EL), axis=-1)                 # Liebig minimum factor + Monod
def f_phi(phi): return np.exp(-(phi-PHI_OPT)**2/(2*SIGMA_PHI**2))  # optimal-porosity bell curve
def env_mod(T,H,M,phi):
    return f_T(T)*f_H(H)*f_M(M)*f_phi(phi)

def growth_accum(x, t_rel):
    u = np.zeros_like(x + t_rel*0.0)
    for k in range(5):
        z = np.log1p(t_rel/TAU[k]); s = 1.0/(1.0+np.exp(-(z-1.0)))
        u += AMPS[k]*np.sin((k+1)*np.pi*x)*s
    return u

def pde_theory(x, t_rel, t_abs):
    """Ground truth: u = P(x) * S(E) * G(x,t_rel) with spatial modality beta = 0.6"""
    T = temperature(t_abs); H = humidity(t_abs)
    M = trace_elements(x, t_abs); phi = porosity(x)
    return (1.0 + 0.6*x**2) * env_mod(T,H,M,phi) * growth_accum(x, t_rel)

# =====================================================================
# 2. Absolute-time triple (AT) encoding
# =====================================================================
def encode_anchor(anchor_unix, t_abs):
    """Anchor: season (annual sin/cos + 2nd harmonic) + intraday (daily sin/cos) -> 6 dims"""
    doy = day_of_year(t_abs)
    f = [np.sin(2*np.pi*doy/365.24).astype(np.float32),
         np.cos(2*np.pi*doy/365.24).astype(np.float32),
         np.sin(4*np.pi*doy/365.24).astype(np.float32),
         np.cos(4*np.pi*doy/365.24).astype(np.float32)]
    tday = (t_abs % DAY) / DAY
    f += [np.sin(2*np.pi*tday).astype(np.float32),
          np.cos(2*np.pi*tday).astype(np.float32)]
    return np.stack(f, -1).astype(np.float32)               # (n, 6)

def encode_offset(t_rel):
    """Offset: log-normalized + multi-frequency Fourier features -> 9 dims"""
    lo = np.log1p(t_rel)/np.log1p(6*YEAR)
    f = [lo.astype(np.float32)]
    for freq in [1.0, 4.0, 16.0, 64.0]:
        w = 2*np.pi*freq
        f += [np.sin(lo*w).astype(np.float32), np.cos(lo*w).astype(np.float32)]
    return np.stack(f, -1)                                   # (n, 9)

def encode_span(span_sec, n):
    """Span: log-normalized -> 1 dim"""
    span_sec = np.asarray(span_sec, dtype=np.float32)
    if span_sec.ndim == 0:
        span_sec = np.full(n, span_sec, dtype=np.float32)
    return (np.log1p(span_sec)/np.log1p(6*YEAR)).astype(np.float32).reshape(-1, 1)

# =====================================================================
# 3. Mixture-of-Experts (MoE) -- physical-modality multiplicative decomposition P.S.G
# =====================================================================
D = 64

class Block(nn.Module):
    """Gated residual block used inside the G expert (Mamba-like)"""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d); self.p = nn.Linear(d, 2*d)
        self.sc = nn.Linear(d, d); self.o = nn.Linear(d, d); self.a = nn.SiLU()
    def forward(self, u):
        z = self.ln(u); b, c = self.p(z).chunk(2, -1)
        h = self.a(self.sc(z)); h = self.o(h*self.a(c))
        return u + h*torch.sigmoid(b)

class ExpertP(nn.Module):
    """P expert: spatial modality P(x) -- input is spatial coordinate x only"""
    def __init__(self, d=D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, x):
        return self.net(x)

class ExpertS(nn.Module):
    """S expert: environmental modulation S(T,H,M,phi) -- input is 14-dim env [T,H,M(11),phi], output exp(.) to stay positive"""
    def __init__(self, d=D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, env):
        return torch.exp(self.net(env))

class ExpertG(nn.Module):
    """G expert: growth accumulation G(x, t_rel) -- input x + 9-dim time offset through gated residual blocks"""
    def __init__(self, d=D):
        super().__init__()
        self.x_enc = nn.Linear(1, d)
        self.t_enc = nn.Sequential(nn.Linear(9, d), nn.SiLU(), nn.Linear(d, d))
        self.b = nn.ModuleList([Block(d) for _ in range(2)])
        self.o = nn.Linear(d, 1)
    def forward(self, x, offset):
        h = self.x_enc(x) + self.t_enc(offset)
        for bl in self.b:
            h = bl(h)
        return self.o(h)

class ATSpatialMSMoE(nn.Module):
    """AT-Spatial-MS-MoE main model: u = P(x) * S(env) * G(x, offset)"""
    def __init__(self, d=D):
        super().__init__()
        self.P = ExpertP(d); self.S = ExpertS(d); self.G = ExpertG(d)
    def forward(self, x, offset, anchor=None, span=None, env=None):
        u = self.P(x) * self.S(env) * self.G(x, offset)
        return u, torch.zeros_like(u)   # second output is a placeholder (no uncertainty)


# =====================================================================
# 4. Training and evaluation (minimal example)
# =====================================================================
SCALE_NAMES = ["minute", "hour", "day", "month", "year"]
SCALE_RESOLUTION = [60.0, 3600.0, 86400.0, 86400*30.0, 86400*365.0]
SCALE_SPAN = [HOUR, DAY, 86400*30.0, YEAR, 6*YEAR]

def sample_scale(s, n, seed):
    rng = np.random.default_rng(seed)
    span, res = SCALE_SPAN[s], SCALE_RESOLUTION[s]
    xs = rng.uniform(0, 1, n)
    t_rel = np.clip(np.exp(rng.uniform(np.log(res), np.log(span), n)), res, span)
    phase = rng.uniform(0, 365.24*DAY, n)
    t_abs = ANCHOR_UNIX + t_rel + phase
    return xs, t_rel, t_abs

def make_data(s, n, seed):
    xs, t_rel, t_abs = sample_scale(s, n, seed)
    T = temperature(t_abs); H = humidity(t_abs)
    M = trace_elements(xs, t_abs); phi = porosity(xs)
    return xs, t_rel, t_abs, T, H, M, phi, pde_theory(xs, t_rel, t_abs)

def nll(mu, lv, t):
    """Negative log-likelihood (heteroscedastic Gaussian): mu = mean, lv = log-variance"""
    v = torch.exp(lv) + 1e-6
    return 0.5*(lv + (t-mu)**2/v)

def build_input(xs, t_rel, t_abs, T, H, M, phi, span_sec):
    x = torch.tensor(xs.reshape(-1, 1).astype(np.float32))
    offset = torch.tensor(encode_offset(t_rel))
    env = torch.tensor(np.concatenate([T.reshape(-1,1), H.reshape(-1,1), M, phi.reshape(-1,1)], -1).astype(np.float32))
    return x, offset, env

def r2(p, t):
    ssr = ((t-p)**2).sum(); sst = ((t-t.mean())**2).sum()
    return 0.0 if sst < 1e-12 else 1 - ssr/sst

def train_moe(model, steps=1000, lr=1e-3):
    """Train the MoE model on the five-scale training set"""
    xs, tr, ta, T, H, M, phi, u = [], [], [], [], [], [], [], []
    for s in range(5):
        a,b,c,d,e,f,g,h = make_data(s, 200, seed=61000+s)
        xs.append(a); tr.append(b); ta.append(c); T.append(d); H.append(e); M.append(f); phi.append(g); u.append(h)
    xs = np.concatenate(xs); tr = np.concatenate(tr); ta = np.concatenate(ta)
    T = np.concatenate(T); H = np.concatenate(H); M = np.concatenate(M); phi = np.concatenate(phi)
    u = np.concatenate(u)
    x, offset, env = build_input(xs, tr, ta, T, H, M, phi, None)
    U = torch.tensor(u.astype(np.float32)).reshape(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        mu, lv = model(x, offset, env=env)
        loss = nll(mu, lv, U).mean()
        loss.backward(); opt.step()
    return model


if __name__ == '__main__':
    print("=== AT-Spatial-MS-MoE self-check ===")
    model = ATSpatialMSMoE()
    print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
    # forward self-check
    xs, t_rel, t_abs = sample_scale(3, 32, seed=0)  # day scale
    T = temperature(t_abs); H = humidity(t_abs); M = trace_elements(xs, t_abs); phi = porosity(xs)
    x, offset, env = build_input(xs, t_rel, t_abs, T, H, M, phi, None)
    mu, _ = model(x, offset, env=env)
    print(f"forward output shape: {mu.shape}, sample values: {mu.detach().numpy().reshape(-1)[:5]}")
    print("Ran OK -- see run_model / make_data for the full training pipeline")
