"""
假单胞菌通过多种途径综合抑制镰刀菌的宏观群落动力学仿真（v4 双直接抑制项版）
==============================================================================

设计原则：让两条抑制途径都是"直接、机理清晰、可分解贡献"的抑制项，
避免 v3 中"共享底物竞争"被 Logistic 反馈抵消、贡献无法分离的问题。

两条抑制途径：
  途径① 铁载体竞争（siderophore，资源剥夺型营养竞争）：
       假单胞菌高效分泌铁载体 Q，螯合铁离子，剥夺镰刀菌生长所需的 Fe，
       体现为对镰刀菌生长率的直接抑制 term1 = ε_P * Q/(Kq+Q) * F

  途径② 抑菌代谢产物（DAPG/吩嗪类，直接致死）：
       假单胞菌分泌抑菌物质 A，直接致死镰刀菌 term2 = mu_F * A/(Ka+A) * F

宏观群落（生态位分化）：
  F 镰刀菌、P 假单胞菌、B 有益菌；F 抑制 B（连作障碍），P 压制 F 后 B 释放。

状态变量（6）：F、P、B、Q（铁载体）、A（抑菌产物）、S（共享底物）

输出：群落时间序列 + 相图 + 抑制率/途径贡献分解
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

# ═══ 参数 ═══
r_F, r_P, r_B = 0.30, 0.35, 0.32
K_F, K_P, K_B = 1.0, 1.0, 1.0
Ks_S = 0.25   # 共享底物半饱和（三菌共用，但不再是"竞争差异化"手段）

# 途径① 铁载体竞争
eps_P  = 0.14    # 铁载体对镰刀菌的抑制强度
Kq     = 0.30    # 铁载体半饱和
q_P    = 0.30    # 铁载体分泌率
delta_q= 0.10    # 铁载体降解率

# 途径② 抑菌产物
mu_F   = 0.12    # 抑菌产物对镰刀菌致死率
Ka_A   = 0.20    # 抑菌产物半饱和
alpha  = 0.10    # 抑菌产物分泌率
delta_A= 0.12    # 抑菌产物降解率
mu_B   = 0.001   # 抑菌产物对有益菌几乎无害

# 连作障碍：镰刀菌抑制有益菌
sup_B  = 0.12    # 镰刀菌→有益菌抑制系数
Ksup   = 0.2

S_supply = 0.30  # 底物供给（充足，使营养不是唯一瓶颈）
YIELD = 0.5

def deriv(y, t):
    F, P, B, Q, A, S = y
    g_F = r_F*F*(S/(S+Ks_S))*(1-F/K_F)
    g_P = r_P*P*(S/(S+Ks_S))*(1-P/K_P)
    g_B = r_B*B*(S/(S+Ks_S))*(1-B/K_B)

    # 两条直接抑制途径
    inh_sidero = eps_P * (Q/(Kq+Q)) * F        # 途径① 铁载体资源剥夺
    inh_anti   = mu_F  * (A/(Ka_A+A)) * F      # 途径② 抑菌产物直接致死

    # 有益菌：受抑菌产物轻微抑制 + 镰刀菌连作抑制
    kill_B = mu_B * A * B
    sup_B_term = sup_B*(F/(F+Ksup))*B

    dF = g_F - inh_sidero - inh_anti
    dP = g_P
    dB = g_B - kill_B - sup_B_term
    dQ = q_P*P - delta_q*Q
    dA = alpha*P - delta_A*A
    dS = S_supply - (g_F + g_P + g_B)/YIELD
    return np.array([dF, dP, dB, dQ, dA, dS])

def solve(y0, T=150.0, dt=0.05, deriv_fn=deriv):
    n = int(T/dt)
    t = np.zeros(n+1); Y = np.zeros((6, n+1))
    Y[:,0] = np.array(y0, dtype=float); t[0] = 0.0
    for i in range(n):
        k1 = deriv_fn(Y[:,i], t[i])
        k2 = deriv_fn(Y[:,i]+0.5*dt*k1, t[i]+0.5*dt)
        k3 = deriv_fn(Y[:,i]+0.5*dt*k2, t[i]+0.5*dt)
        k4 = deriv_fn(Y[:,i]+dt*k3, t[i]+dt)
        y = Y[:,i] + dt*(k1+2*k2+2*k3+k4)/6
        Y[:,i+1] = np.maximum(y, 0.0); t[i+1] = t[i]+dt
    return t, Y

T, dt = 150.0, 0.05
y0_ctrl  = [0.30, 0.00, 0.10, 0.0, 0.0, 0.8]
y0_treat = [0.30, 0.25, 0.10, 0.0, 0.0, 0.8]

t_c, Y_c = solve(y0_ctrl, T, dt)
t_t, Y_t = solve(y0_treat, T, dt)
Fc,Pc,Bc,Qc,Ac,Sc = Y_c
Ft,Pt,Bt,Qt,At,St = Y_t
Cc = Fc+Pc+Bc; Ct = Ft+Pt+Bt

# ═══ 图1 群落时间序列 ═══
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0,0]
ax.plot(t_c, Fc, label='Fusarium (no $P$)', color='#e74c3c', lw=2)
ax.plot(t_t, Ft, label='Fusarium (with $P$)', color='#c0392b', lw=2, ls='--')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Density $F$')
ax.set_title('Fusarium: dual-pathway suppression', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.plot(t_c, Bc, label='Beneficial (no $P$)', color='#27ae60', lw=2)
ax.plot(t_t, Bt, label='Beneficial (with $P$)', color='#1e8449', lw=2, ls='--')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Density $B$')
ax.set_title('Beneficials: positive increase (release)', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,0]
ax.plot(t_c, Cc, label='Total community (no $P$)', color='#7f8c8d', lw=2)
ax.plot(t_t, Ct, label='Total community (with $P$)', color='#34495e', lw=2, ls='--')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Total community $C$')
ax.set_title('Macroscopic community increase', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.plot(t_t, Pt, label='Pseudomonas $P$', color='#2980b9', lw=2)
ax.plot(t_t, Qt, label='Siderophore $Q$ (pathway 1)', color='#d35400', lw=2, ls='-.')
ax.plot(t_t, At, label='Antifungal $A$ (pathway 2)', color='#8e44ad', lw=2, ls=':')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Density / concentration')
ax.set_title('Biocontrol agent and two inhibitory factors', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig1_community_timeseries.png', dpi=150, bbox_inches='tight'); plt.close()

# ═══ 图2 相图 ═══
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(Fc, Pc, color='#95a5a6', lw=1.5, label='Trajectory without $P$')
ax.plot(Ft, Pt, color='#2980b9', lw=2, label='Trajectory with $P$')
ax.scatter([Fc[-1]],[0.0], marker='*', s=250, color='#e74c3c', zorder=5, label='Steady state without $P$')
ax.scatter([Ft[-1]],[Pt[-1]], marker='*', s=250, color='#1a5276', zorder=5, label='Steady state with $P$')
ax.set_xlabel('Fusarium $F$'); ax.set_ylabel('Pseudomonas $P$')
ax.set_title('Phase portrait: Pseudomonas shifts the community steady state', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig2_phase_FP.png', dpi=150, bbox_inches='tight'); plt.close()

# ═══ 图3 抑制率 + 途径消融（贡献分解）═══
eta = np.clip(1 - Ft/np.maximum(Fc,1e-6), 0, 1)

def ablate_traj(off_sidero=False, off_anti=False):
    def d2(y, t):
        F,P,B,Q,A,S = y
        g_F = r_F*F*(S/(S+Ks_S))*(1-F/K_F)
        g_P = r_P*P*(S/(S+Ks_S))*(1-P/K_P)
        g_B = r_B*B*(S/(S+Ks_S))*(1-B/K_B)
        inh_sidero = (0 if off_sidero else eps_P*(Q/(Kq+Q))*F)
        inh_anti   = (0 if off_anti else mu_F*(A/(Ka_A+A))*F)
        kill_B = mu_B*A*B
        sup_B_term = sup_B*(F/(F+Ksup))*B
        dF = g_F - inh_sidero - inh_anti
        dP = g_P
        dB = g_B - kill_B - sup_B_term
        dQ = q_P*P - delta_q*Q
        dA = alpha*P - delta_A*A
        dS = S_supply - (g_F+g_P+g_B)/YIELD
        return np.array([dF,dP,dB,dQ,dA,dS])
    t, Y = solve(y0_treat, T, dt, deriv_fn=d2)
    return t, Y[0]

def steady_F(off_sidero=False, off_anti=False):
    return ablate_traj(off_sidero, off_anti)[1][-1]

F_base    = Ft[-1]
F_off_sid = steady_F(off_sidero=True)
F_off_ant = steady_F(off_anti=True)
d_sid = F_off_sid - F_base   # 关铁载体后镰刀菌回升量 = 铁载体途径贡献
d_ant = F_off_ant - F_base   # 关抑菌产物后镰刀菌回升量 = 抑菌产物贡献
tot = d_sid + d_ant + 1e-9
contrib = [100*d_sid/tot, 100*d_ant/tot]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.plot(t_t, eta*100, color='#c0392b', lw=2)
ax.axhline(100*(1-Ft[-1]/Fc[-1]), color='blue', ls='--', lw=1.5,
           label=f'Steady-state inhibition {100*(1-Ft[-1]/Fc[-1]):.0f}%')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Fusarium inhibition $\\eta$ (%)')
ax.set_title('Fusarium inhibition rate', fontweight='bold'); ax.set_ylim(0,100); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
t_sid, F_sid = ablate_traj(off_sidero=True)   # pathway 1 off
t_ant, F_ant = ablate_traj(off_anti=True)     # pathway 2 off
ax.plot(t_t, Ft, label='Both pathways active', color='#c0392b', lw=2.2)
ax.plot(t_sid, F_sid, label='Pathway 1 off (siderophore)', color='#d35400', lw=1.8, ls='--')
ax.plot(t_ant, F_ant, label='Pathway 2 off (antifungal)', color='#8e44ad', lw=1.8, ls='-.')
# steady-state markers
ax.scatter([t_t[-1]], [Ft[-1]], s=48, color='#c0392b', zorder=5)
ax.scatter([t_sid[-1]], [F_sid[-1]], s=48, color='#d35400', zorder=5)
ax.scatter([t_ant[-1]], [F_ant[-1]], s=48, color='#8e44ad', zorder=5)
# rebound arrows
ax.annotate('', xy=(t_sid[-1], F_sid[-1]), xytext=(t_sid[-1], Ft[-1]),
            arrowprops=dict(arrowstyle='<->', color='#d35400', lw=1.2))
ax.annotate('', xy=(t_ant[-1], F_ant[-1]), xytext=(t_ant[-1], Ft[-1]),
            arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.2))
ax.text(t_sid[-1]*0.96, (Ft[-1]+F_sid[-1])/2, f'+{d_sid:.2f}', color='#d35400',
        fontsize=9, ha='right', va='center', fontweight='bold')
ax.text(t_ant[-1]*0.96, (Ft[-1]+F_ant[-1])/2, f'+{d_ant:.2f}', color='#8e44ad',
        fontsize=9, ha='right', va='center', fontweight='bold')
ax.set_xlabel('Time (day)'); ax.set_ylabel('Fusarium density $F$')
ax.set_title('Ablation: rebound of Fusarium when one pathway is switched off',
             fontweight='bold', fontsize=10)
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.92, borderpad=0.5)
ax.set_ylim(0, 1.35)
ax.grid(alpha=0.3)

ax = axes[2]
labels = ['Siderophore\n(pathway 1)', 'Antifungal\n(pathway 2)']
ax.bar(labels, contrib, color=['#d35400','#8e44ad'])
ax.set_ylabel('Relative contribution (%)'); ax.set_ylim(0,100)
ax.set_title('Relative contribution of two pathways (ablation)', fontweight='bold')
for i,v in enumerate(contrib):
    ax.text(i, v+2, f'{v:.0f}%', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('fig3_inhibition_rate.png', dpi=150, bbox_inches='tight'); plt.close()

# ═══ 结论 ═══
print("=== 宏观抑制效应（双直接抑制途径）===")
print(f"对照(无P) 镰刀菌稳态 F_ctrl = {Fc[-1]:.4f}")
print(f"处理(有P) 镰刀菌稳态 F_treat = {Ft[-1]:.4f}")
print(f"镰刀菌抑制率 η = {100*(1-Ft[-1]/Fc[-1]):.1f}%")
print(f"有益菌: B_ctrl={Bc[-1]:.4f} -> B_treat={Bt[-1]:.4f} "
      f"(涨幅 {100*(Bt[-1]-Bc[-1])/max(Bc[-1],1e-6):+.1f}%)")
print(f"总群落: C_ctrl={Cc[-1]:.4f} -> C_treat={Ct[-1]:.4f}")
print(f"途径贡献: 铁载体竞争 {contrib[0]:.0f}% / 抑菌产物 {contrib[1]:.0f}%")

np.savez('multi_pathway_results.npz',
         t=t_t, F_ctrl=Fc, F_treat=Ft, P=Pt, B_ctrl=Bc, B_treat=Bt,
         C_ctrl=Cc, C_treat=Ct, Q=Qt, A=At, S=St, eta=eta,
         pathway_contrib=np.array(contrib))
print("\n[完成] multi_pathway_results.npz + 3张图")
