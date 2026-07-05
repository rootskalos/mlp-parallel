"""
compare.py
Genera figuras comparativas Mac vs Khipu (mismo algoritmo, dos hardwares).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "lines.linewidth": 2, "lines.markersize": 7})


def speedup_frame(csv, cores):
    a = pd.read_csv(os.path.join(DATA, csv))
    Ts = {n: a[(a.n == n) & (a.p == 1)].T_total.iloc[0] for n in a.n.unique()}
    a["S"] = a.apply(lambda r: Ts[r.n] / r.T_total, axis=1)
    a["E"] = a.S / a.p
    return a, cores


mac, mac_cores = speedup_frame("summary.csv", 10)
khi, khi_cores = speedup_frame("summary_khipu.csv", 32)

# ---------- Fig 1: Speedup comparado (n=40000) ----------
n = 40000
fig, ax = plt.subplots(figsize=(7.5, 4.8))
P = [1, 2, 4, 8, 16, 32]
ax.plot(P, P, "k--", alpha=0.6, label="Ideal S=p")
mm = mac[mac.n == n].sort_values("p")
kk = khi[khi.n == n].sort_values("p")
ax.plot(mm.p, mm.S, "o-", color="#d62728", label=f"Mac M5 (10 cores) — techo ~{mm.S.max():.1f}")
ax.plot(kk.p, kk.S, "s-", color="#1f77b4", label=f"Khipu n003 (32 cores) — techo ~{kk.S.max():.1f}")
ax.axvline(mac_cores, color="#d62728", ls=":", alpha=0.5)
ax.axvline(khi_cores, color="#1f77b4", ls=":", alpha=0.5)
ax.set_xscale("log", base=2); ax.set_xticks(P); ax.set_xticklabels(P)
ax.set_xlabel("Número de procesos p"); ax.set_ylabel("Speedup S(p)")
ax.set_title(f"Speedup Mac vs Khipu (n={n})"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_compare_speedup.png")); plt.close()

# ---------- Fig 2: Eficiencia comparada (n=40000) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="E=1 ideal")
ax.plot(mm.p, mm.E, "o-", color="#d62728", label="Mac M5 (10 cores)")
ax.plot(kk.p, kk.E, "s-", color="#1f77b4", label="Khipu n003 (32 cores)")
ax.axvline(mac_cores, color="#d62728", ls=":", alpha=0.5)
ax.axvline(khi_cores, color="#1f77b4", ls=":", alpha=0.5)
ax.set_xscale("log", base=2); ax.set_xticks(P); ax.set_xticklabels(P)
ax.set_xlabel("p"); ax.set_ylabel("Eficiencia E(p)=S(p)/p")
ax.set_title(f"Eficiencia Mac vs Khipu (n={n})"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_compare_eficiencia.png")); plt.close()

# ---------- Fig 3: Speedup vs p para todos los n (Khipu) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(P, P, "k--", alpha=0.6, label="Ideal S=p")
cols = plt.cm.viridis(np.linspace(0.05, 0.85, 4))
for i, n in enumerate(sorted(khi.n.unique())):
    s = khi[khi.n == n].sort_values("p")
    ax.plot(s.p, s.S, "o-", color=cols[i], label=f"n={n}")
ax.axvline(khi_cores, color="red", ls=":", alpha=0.6, label="cores=32")
ax.set_xscale("log", base=2); ax.set_xticks(P); ax.set_xticklabels(P)
ax.set_xlabel("p"); ax.set_ylabel("Speedup S(p)")
ax.set_title("Speedup en Khipu (32 cores)"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_khipu_speedup_alln.png")); plt.close()

print("Figuras comparativas generadas en", FIG)
