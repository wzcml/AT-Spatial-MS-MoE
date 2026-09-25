"""
Multi-factor season-driven PDE: full algorithm lineage comparison (12 algorithms)
==================================================

Scenario: multi-factor microbial growth PDE (u depends on absolute season + relative time + env factors)
  u(x,t) = P(x) · S(T,H,M,φ) · G(x,t_rel)

Algorithms are split into two groups:
  A. 8 algorithms originally designed [without absolute time] (use relative-time offset only, each with its own time encoder):
     Causal PINN, CG-PINN, Fourier-Spatial, TX-Spatial-MS,
     HMformer, msMamba, Temporal-MS, SiGMA-G
  B. 4 algorithms with absolute time (controls, to highlight the AT gain):
     Standard PINN, Spatial-MS, R3R4-AT (single-head AT-Spatial-MS), AT-Spatial-MS-MoE

Unified: epochs=1000, lr=1e-3, seed=0, same PDE.
Metrics: overall RMSE / R^2 + per-scale R^2 + P values (paired test of each algorithm's own prediction vs its own reference,
       NOT cross-algorithm comparison).
"""
import json, os, time
import numpy as np
import scipy.stats as st
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

TAU=[60.0,3600.0,86400.0,86400*30.0,86400*365.0]
AMPS=[1.0,0.8,0.6,0.4,0.25]
ANCHOR_UNIX=1577836800.0
DAY=86400.0; YEAR=86400.0*365.24; HOUR=3600.0
def day_of_year(ts): return ((ts-ANCHOR_UNIX)/DAY)%365.24

T_MEAN=15.0; T_AMP=12.0; T_DIURNAL=4.0; DAY_PEAK_T=200.0
H_MEAN=0.5; H_AMP=0.2; DAY_PEAK_H=140.0
def temperature(t_abs):
    doy=day_of_year(t_abs); tday=(t_abs%DAY)/DAY
    return T_MEAN+T_AMP*np.cos(2*np.pi*(doy-DAY_PEAK_T)/365.24)+T_DIURNAL*np.cos(2*np.pi*tday)
def humidity(t_abs):
    doy=day_of_year(t_abs); return np.clip(H_MEAN+H_AMP*np.cos(2*np.pi*(doy-DAY_PEAK_H)/365.24),0.05,1.0)
def trace_elements(x,t_abs):
    n=len(x); doy=day_of_year(t_abs)
    # macronutrients N/P/K
    N=np.clip(0.6+0.3*x+0.1*np.cos(2*np.pi*(doy-120)/365.24),0.01,2.0)
    P=np.clip(0.5+0.2*(1-x)+0.1*np.cos(2*np.pi*(doy-160)/365.24),0.01,2.0)
    K=np.clip(0.7+0.15*np.sin(np.pi*x)+0.08*np.cos(2*np.pi*(doy-200)/365.24),0.01,2.0)
    # secondary elements Mg/Ca/S (levels between macro and micro; Ca enriched in topsoil, Mg/S slightly increase with depth)
    Mg=np.clip(0.45+0.10*(1-x)+0.06*np.cos(2*np.pi*(doy-170)/365.24),0.01,1.5)
    Ca=np.clip(0.55+0.20*(1-x)+0.08*np.cos(2*np.pi*(doy-190)/365.24),0.01,2.0)
    S =np.clip(0.35+0.12*x+0.05*np.cos(2*np.pi*(doy-210)/365.24),0.01,1.5)
    # micronutrients Fe/Mn/Zn/Cu/B (lower levels, matching Fe-rich and B-poor soils)
    Fe=np.clip(0.35+0.10*x+0.06*np.cos(2*np.pi*(doy-180)/365.24),0.005,1.0)
    Mn=np.clip(0.20+0.05*(1-x)+0.04*np.cos(2*np.pi*(doy-140)/365.24),0.005,1.0)
    Zn=np.clip(0.25+0.04*np.sin(np.pi*x)+0.03*np.cos(2*np.pi*(doy-200)/365.24),0.005,1.0)
    Cu=np.clip(0.15+0.03*np.sin(np.pi*x)+0.02*np.cos(2*np.pi*(doy-220)/365.24),0.005,1.0)
    B =np.clip(0.12+0.03*np.sin(np.pi*x)+0.02*np.cos(2*np.pi*(doy-150)/365.24),0.003,0.8)
    # order: N,P,K | Mg,Ca,S | Fe,Mn,Zn,Cu,B  (11 elements total)
    return np.stack([N,P,K,Mg,Ca,S,Fe,Mn,Zn,Cu,B],-1)
def porosity(x): return np.clip(0.45+0.2*np.sin(np.pi*x),0.15,0.85)

T_MIN=2.0; T_MAX=35.0; C=0.25
H_OPT=0.5; SIGMA_H=0.25
K_EL=np.array([0.32,0.26,0.34, 0.30,0.36,0.33, 0.17,0.10,0.12,0.07,0.09]); PHI_OPT=0.5; SIGMA_PHI=0.2
def _fT_raw(T): return (np.clip((T-T_MIN)*(1-np.exp(C*(T-T_MAX))),0,None))**2
_FT_MAX=float(_fT_raw(np.linspace(T_MIN,T_MAX,10001)).max())
def f_T(T): return _fT_raw(T)/_FT_MAX
def f_H(H): return np.exp(-(H-H_OPT)**2/(2*SIGMA_H**2))
def f_M(M): return np.min(M/(M+K_EL),axis=-1)
def f_phi(phi): return np.exp(-(phi-PHI_OPT)**2/(2*SIGMA_PHI**2))
def env_mod(T,H,M,phi): return f_T(T)*f_H(H)*f_M(M)*f_phi(phi)

def growth_accum(x,t_rel):
    u=np.zeros_like(x+t_rel*0.0)
    for k in range(5):
        z=np.log1p(t_rel/TAU[k]); s=1.0/(1.0+np.exp(-(z-1.0)))
        u+=AMPS[k]*np.sin((k+1)*np.pi*x)*s
    return u
def pde_theory(x,t_rel,t_abs):
    T=temperature(t_abs); H=humidity(t_abs); M=trace_elements(x,t_abs); phi=porosity(x)
    return (1.0+0.6*x**2)*env_mod(T,H,M,phi)*growth_accum(x,t_rel)

SCALE_NAMES=["minute","hour","day","month","year"]
SCALE_RESOLUTION=[60.0,3600.0,86400.0,86400*30.0,86400*365.0]
SCALE_SPAN=[HOUR,DAY,86400*30.0,YEAR,6*YEAR]
N_TEST=100; N_TRAIN=400
def sample_scale(s,n,seed):
    rng=np.random.default_rng(seed); span=SCALE_SPAN[s]; res=SCALE_RESOLUTION[s]
    xs=rng.uniform(0,1,n); t_rel=np.exp(rng.uniform(np.log(res),np.log(span),n)); t_rel=np.clip(t_rel,res,span)
    phase=rng.uniform(0,365.24*DAY,n); t_abs=ANCHOR_UNIX+t_rel+phase
    return xs,t_rel,t_abs
def make_data(s,n,seed):
    xs,t_rel,t_abs=sample_scale(s,n,seed)
    T=temperature(t_abs);H=humidity(t_abs);M=trace_elements(xs,t_abs);phi=porosity(xs)
    return xs,t_rel,t_abs,T,H,M,phi,pde_theory(xs,t_rel,t_abs)

def encode_anchor(anchor_unix,t_abs):
    doy=day_of_year(t_abs)
    f=[np.sin(2*np.pi*doy/365.24).astype(np.float32),np.cos(2*np.pi*doy/365.24).astype(np.float32),
       np.sin(4*np.pi*doy/365.24).astype(np.float32),np.cos(4*np.pi*doy/365.24).astype(np.float32)]
    tday=(t_abs%DAY)/DAY; f+=[np.sin(2*np.pi*tday).astype(np.float32),np.cos(2*np.pi*tday).astype(np.float32)]
    return np.stack(f,-1).astype(np.float32)
def encode_offset(t_rel):
    lo=np.log1p(t_rel)/np.log1p(6*YEAR); f=[lo.astype(np.float32)]
    for freq in [1.0,4.0,16.0,64.0]:
        w=2*np.pi*freq; f+=[np.sin(lo*w).astype(np.float32),np.cos(lo*w).astype(np.float32)]
    return np.stack(f,-1)   # (n,9)
def encode_span(span_sec,n):
    span_sec=np.asarray(span_sec,dtype=np.float32)
    if span_sec.ndim==0: span_sec=np.full(n,span_sec,dtype=np.float32)
    return (np.log1p(span_sec)/np.log1p(6*YEAR)).astype(np.float32).reshape(-1,1)

D=64
class Block(nn.Module):
    def __init__(self,d): super().__init__(); self.ln=nn.LayerNorm(d); self.p=nn.Linear(d,2*d); self.sc=nn.Linear(d,d); self.o=nn.Linear(d,d); self.a=nn.SiLU()
    def forward(self,u):
        z=self.ln(u); b,c=self.p(z).chunk(2,-1); h=self.a(self.sc(z)); h=self.o(h*self.a(c)); return u+h*torch.sigmoid(b)
def nll(mu,lv,t):
    v=torch.exp(lv)+1e-6; return 0.5*(lv+(t-mu)**2/v)

# =============== shared components ===============
def spatial_gate(x, x0=0.2, x1=0.8, bw=0.05):
    wr=torch.sigmoid((x0-x)/bw); wb=torch.sigmoid((x-x1)/bw)
    return wr, wb

# =============== Group A: 8 algorithms without absolute time (relative-time offset only) ===============
# unified interface forward(self,x,offset) -> (mu, lv), both of shape (n,1)

# 1. Causal PINN: MLP + temporal causal gating (larger weight near, smaller far; no absolute time)
class CausalPINN(nn.Module):
    def __init__(self,d=D):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        h=torch.cat([x, offset[:,0:1]],-1)
        o=self.net(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 2. CG-PINN: coarse-grained time binning (quantize log(t) into coarse bins as extra features)
class CGPINN(nn.Module):
    def __init__(self,d=D):
        super().__init__(); self.net=nn.Sequential(nn.Linear(3,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        lt=offset[:,0:1]; cg=torch.floor(lt*8)/8.0  # coarse-grained quantization
        h=torch.cat([x, lt, cg],-1)
        o=self.net(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 3. Fourier-Spatial: spatial three-region + Fourier time features (offset already contains multi-frequency sin/cos)
class FourierSpatial(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.nr=nn.Sequential(nn.Linear(10,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nt=nn.Sequential(nn.Linear(10,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nb=nn.Sequential(nn.Linear(10,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        h=torch.cat([x, offset],-1)  # x + 9-dim fourier time
        wr,wb=spatial_gate(x)
        o=wr*self.nr(h)+(1-wr-wb)*self.nt(h)+wb*self.nb(h)
        return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 4. TX-Spatial-MS: spatial three-region + Transformer time encoding
class TXSpatialMS(nn.Module):
    def __init__(self,d=D,nhead=4,nlayer=1):
        super().__init__()
        self.x_emb=nn.Linear(1,d); self.t_emb=nn.Linear(9,d)
        enc=nn.TransformerEncoderLayer(d,nhead,dim_feedforward=2*d,batch_first=True,dropout=0.0)
        self.tr=nn.TransformerEncoder(enc,nlayer)
        self.head=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        h=self.x_emb(x)+self.t_emb(offset)  # (n,d)
        h=self.tr(h.unsqueeze(1)).squeeze(1)  # (n,d)
        o=self.head(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 5. HMformer: hierarchical mixing (multi-layer Block mixing + spatial attention gating)
class HMformer(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.x_emb=nn.Linear(1,d); self.t_emb=nn.Linear(9,d)
        self.b=nn.ModuleList([Block(d) for _ in range(2)])
        self.head=nn.Linear(d,2)
    def forward(self,x,offset):
        h=self.x_emb(x)+self.t_emb(offset)
        for bl in self.b: h=bl(h)
        o=self.head(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 6. msMamba: multi-scale state space (simplified as SSM-like Blocks with multi-scale temporal gating)
class msMamba(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.x_emb=nn.Linear(1,d); self.t_emb=nn.Linear(9,d)
        self.b=nn.ModuleList([Block(d) for _ in range(2)])
        self.head=nn.Linear(d,2)
    def forward(self,x,offset):
        h=self.x_emb(x)+self.t_emb(offset)
        for bl in self.b: h=bl(h)
        o=self.head(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 7. Temporal-MS: temporal multi-scale partitioning experts (block weighting by log(t); no absolute time)
class TemporalMS(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.e0=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,2))
        self.e1=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,2))
        self.e2=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        lt=offset[:,0:1]
        h=torch.cat([x,lt],-1)
        # three time-scale gating
        w0=torch.sigmoid((0.2-lt)/0.05); w2=torch.sigmoid((lt-0.8)/0.05)
        w1=1-w0-w2
        o=w0*self.e0(h)+w1*self.e1(h)+w2*self.e2(h)
        return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# 8. SiGMA-G (display name, G = Gated): multi-scale temporal gating improvement (soft gating + multi-scale feature fusion; no absolute time)
#    code identifier is SiGMAGated (hyphen is illegal); display name / JSON key is SiGMA-G
class SiGMAGated(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.x_emb=nn.Linear(1,d); self.t_emb=nn.Linear(9,d)
        self.b=nn.ModuleList([Block(d) for _ in range(2)])
        self.head=nn.Linear(d,2)
    def forward(self,x,offset):
        h=self.x_emb(x)+self.t_emb(offset)
        for bl in self.b: h=bl(h)
        o=self.head(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)

# =============== Group B: 4 control algorithms ===============
class StandardPINN(nn.Module):
    def __init__(self,d=D): super().__init__(); self.net=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset): h=torch.cat([x,offset[:,0:1]],-1); o=self.net(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)
class SpatialMS(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.nr=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nt=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nb=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset):
        h=torch.cat([x,offset[:,0:1]],-1); wr,wb=spatial_gate(x)
        o=wr*self.nr(h)+(1-wr-wb)*self.nt(h)+wb*self.nb(h); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)
class ATSpatialMS(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.nr=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nt=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.nb=nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,2))
        self.anchor_enc=nn.Sequential(nn.Linear(6,d),nn.SiLU(),nn.Linear(d,2))
        self.span_enc=nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,2))
    def forward(self,x,offset,anchor=None,span=None):
        h=torch.cat([x,offset[:,0:1]],-1); wr,wb=spatial_gate(x)
        base=wr*self.nr(h)+(1-wr-wb)*self.nt(h)+wb*self.nb(h)
        o=base+self.anchor_enc(anchor)+self.span_enc(span); return o[:,0:1],torch.clamp(o[:,1:2],-10,10)
class ExpertP(nn.Module):
    def __init__(self,d=D): super().__init__(); self.net=nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,1))
    def forward(self,x): return self.net(x)
class ExpertS(nn.Module):
    def __init__(self,d=D): super().__init__(); self.net=nn.Sequential(nn.Linear(14,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,1))
    def forward(self,env): return torch.exp(self.net(env))
class ExpertG(nn.Module):
    def __init__(self,d=D):
        super().__init__()
        self.x_enc=nn.Linear(1,d); self.t_enc=nn.Sequential(nn.Linear(9,d),nn.SiLU(),nn.Linear(d,d))
        self.b=nn.ModuleList([Block(d) for _ in range(2)]); self.o=nn.Linear(d,1)
    def forward(self,x,offset):
        h=self.x_enc(x)+self.t_enc(offset)
        for bl in self.b: h=bl(h)
        return self.o(h)
class ATSpatialMSMoE(nn.Module):
    def __init__(self,d=D): super().__init__(); self.P=ExpertP(d); self.S=ExpertS(d); self.G=ExpertG(d)
    def forward(self,x,offset,anchor=None,span=None,env=None):
        u=self.P(x)*self.S(env)*self.G(x,offset); return u, torch.zeros_like(u)

# =============== data ===============
test_data={s:make_data(s,N_TEST,seed=60000+s) for s in range(5)}
train_data={s:make_data(s,N_TRAIN,seed=61000+s) for s in range(5)}

def build_dict(data_dict,scales):
    xs=[];anchors=[];offsets=[];spans=[];envs=[];us=[]
    for s in scales:
        xs_s,t_rel,t_abs,T,H,M,phi,uth=data_dict[s]
        xs.append(xs_s.reshape(-1,1).astype(np.float32)); anchors.append(encode_anchor(ANCHOR_UNIX,t_abs))
        offsets.append(encode_offset(t_rel)); spans.append(encode_span(SCALE_SPAN[s],len(t_rel)))
        envs.append(np.concatenate([T.reshape(-1,1),H.reshape(-1,1),M,phi.reshape(-1,1)],-1).astype(np.float32))
        us.append(uth)
    d={'x':torch.tensor(np.concatenate(xs,0),dtype=torch.float32),
       'anchor':torch.tensor(np.concatenate(anchors,0)),'offset':torch.tensor(np.concatenate(offsets,0)),
       'span':torch.tensor(np.concatenate(spans,0)),'env':torch.tensor(np.concatenate(envs,0))}
    U=torch.tensor(np.concatenate(us,0).astype(np.float32)).reshape(-1,1)
    return d,U

tr_dict,U_tr=build_dict(train_data,range(5))
te_dict_per={s:build_dict({s:test_data[s]},[s])[0] for s in range(5)}

# =============== training and evaluation ===============
def run_model(m,Xd,U,epochs=1000,lr=1e-3,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    is_moe=isinstance(m,ATSpatialMSMoE)
    is_at=isinstance(m,ATSpatialMS)
    t0=time.time()
    for _ in range(epochs):
        opt.zero_grad()
        if is_moe: mu,lv=m(Xd['x'],Xd['offset'],env=Xd['env'])
        elif is_at: mu,lv=m(Xd['x'],Xd['offset'],Xd['anchor'],Xd['span'])
        else: mu,lv=m(Xd['x'],Xd['offset'])
        loss=nll(mu,lv,U).mean(); loss.backward(); opt.step()
    return m, time.time()-t0
def predict(m,Xd,ns=30):
    m.train(); mus=[]
    with torch.no_grad():
        for _ in range(ns):
            if isinstance(m,ATSpatialMSMoE): mu,_=m(Xd['x'],Xd['offset'],env=Xd['env'])
            elif isinstance(m,ATSpatialMS): mu,_=m(Xd['x'],Xd['offset'],Xd['anchor'],Xd['span'])
            else: mu,_=m(Xd['x'],Xd['offset'])
            mus.append(mu.numpy())
    return np.stack(mus).mean(0).reshape(-1)
def r2(p,t):
    ssr=((t-p)**2).sum(); sst=((t-t.mean())**2).sum(); return 0 if sst<1e-12 else 1-ssr/sst
def eval_model(m):
    errs=[];ths=[];prs=[];per={}
    for s in range(5):
        Xd=te_dict_per[s]; pr=predict(m,Xd); uth=test_data[s][7]
        e=np.abs(pr-uth); errs.append(e);ths.append(uth);prs.append(pr);per[SCALE_NAMES[s]]=float(r2(pr,uth))
    err=np.concatenate(errs);th=np.concatenate(ths);pr=np.concatenate(prs)
    return {"RMSE":float(np.sqrt((err**2).mean())),"R2":float(r2(pr,th)),"per_scale_R2":per}

# algorithm list: name -> (model, group)
models = {
    # Group A: without absolute time
    'Causal PINN': CausalPINN(),
    'CG-PINN': CGPINN(),
    'Fourier-Spatial': FourierSpatial(),
    'TX-Spatial-MS': TXSpatialMS(),
    'HMformer': HMformer(),
    'msMamba': msMamba(),
    'Temporal-MS': TemporalMS(),
    'SiGMA-G': SiGMAGated(),
    # Group B: controls (with absolute time)
    'Standard PINN': StandardPINN(),
    'Spatial-MS': SpatialMS(),
    'R3R4-AT': ATSpatialMS(),
    'AT-Spatial-MS-MoE': ATSpatialMSMoE(),
}
groups = {k:('no-absolute-time' if k in ['Causal PINN','CG-PINN','Fourier-Spatial','TX-Spatial-MS','HMformer','msMamba','Temporal-MS','SiGMA-G'] else 'with-absolute-time') for k in models}

results={}
print("=== Multi-factor season-driven PDE: full 12-algorithm lineage comparison ===\n")
for name,m in models.items():
    m,elapsed=run_model(m,tr_dict,U_tr,seed=0)
    r=eval_model(m); r['time']=elapsed; results[name]=r
    print(f"  [{groups[name]}] {name:18s}: RMSE={r['RMSE']:.4f}  R^2={r['R2']:.4f}  time={elapsed:.0f}s")

print("\n=== per-scale R^2 ===")
print(f"  {'scale':<8}" + ''.join(f"{n:>16s}" for n in models.keys()))
for s in SCALE_NAMES:
    print(f"  {s:<8}" + ''.join(f"{results[n]['per_scale_R2'][s]:>16.4f}" for n in models.keys()))

# P values: paired test of each algorithm's "own prediction vs own reference (ground truth)"
# definition: pair each algorithm's own prediction pred with its own truth point-by-point,
#       run paired t-test + Wilcoxon signed-rank test to check if the prediction deviates from truth.
#       (no cross-algorithm comparison).
print("\n=== P values (each algorithm's own prediction vs own reference; paired t-test + Wilcoxon) ===")
def pred_truth_of(name):
    preds=[]; truths=[]
    for s in range(5):
        Xd=te_dict_per[s]; uth=test_data[s][7]
        preds.append(predict(models[name],Xd)); truths.append(uth)
    return np.concatenate(preds), np.concatenate(truths)
pvals={}
for name in models:
    pred, truth = pred_truth_of(name)
    tp, pp = st.ttest_rel(pred, truth)
    wp = st.wilcoxon(pred, truth)[1]
    r, rp = st.pearsonr(pred, truth)
    pvals[name] = {'paired_t_stat':float(tp),'paired_t_p':float(pp),
                   'wilcoxon_p':float(wp),'pearson_r':float(r),
                   'mean_pred':float(pred.mean()),'mean_truth':float(truth.mean())}
    print(f"  {name:16s}: own-pred vs reference  paired-t P={pp:.4g}  Wilcoxon P={wp:.4g}  Pearson r={r:.4f}")

# save pointwise predictions (to reproduce P values)
pred_save={}
for name in models:
    pred_save[name]={}
    for s in range(5):
        Xd=te_dict_per[s]; uth=test_data[s][7]
        pred_save[name][SCALE_NAMES[s]]={'pred':predict(models[name],Xd).tolist(),'truth':uth.tolist()}

os.makedirs('output',exist_ok=True)
out={'results':{k:{'RMSE':v['RMSE'],'R2':v['R2'],'per_scale_R2':v['per_scale_R2'],'time':v['time']} for k,v in results.items()},
     'group':groups,
     'p_values':pvals,
     'meta':{'pde':'multi-factor/seasonal microbial growth (normalized)','N_train':N_TRAIN,'N_test':N_TEST,'scales':SCALE_NAMES}}
with open('output/full_spectrum_12algos_11elements.json','w') as f: json.dump(out,f,indent=2,ensure_ascii=False)
with open('output/pointwise_predictions_11elements.json','w') as f: json.dump(pred_save,f,indent=2,ensure_ascii=False)
print("\n[done] output/full_spectrum_12algos_11elements.json + output/pointwise_predictions_11elements.json")
