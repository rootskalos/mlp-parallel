"""
analyze.py
Lee data/results.csv y genera todas las figuras y tablas del informe:
  fig_tiempos.png      T_compute, T_reduce, T_total vs p (por n)
  fig_speedup.png      Speedup vs p con linea ideal S=p
  fig_eficiencia.png   Eficiencia vs p (con linea 1.0)
  fig_teorica.png      Tiempo experimental vs teorico (ajuste MC + R^2)
  fig_gflops.png       GFLOP/s vs p
  fig_strong.png       Strong scaling (n=40000 fijo, T vs p)
  fig_weak.png         Weak scaling (n proporcional a p, eficiencia)
  data/summary.csv     tabla agregada (media +/- std)
  data/summary.tex     tabla para el informe
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
os.makedirs(FIG, exist_ok=True)

# Parametrizable via env vars (sirve para datos de Mac y de Khipu):
CSV_IN = os.environ.get("RESULTS_FILE", "results.csv")
FIG_PREFIX = os.environ.get("FIG_PREFIX", "fig_")
SUMMARY_FILE = os.environ.get("SUMMARY_FILE", "summary.csv")
SUMMARY_TEX = os.environ.get("SUMMARY_TEX", "summary.tex")
FIT_FILE = os.environ.get("FIT_FILE", "fit.txt")

PS, E, D, H, C = 32, 100, 32, 128, 10
CORES = int(os.environ.get("CORES", os.cpu_count()))

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333", "axes.labelcolor": "#111",
    "axes.titleweight": "bold", "axes.titlecolor": "#1a3a6b",
    "xtick.color": "#333", "ytick.color": "#333",
    "lines.linewidth": 2.4, "lines.markersize": 7.5,
    "legend.frameon": False, "figure.facecolor": "white",
})

df = pd.read_csv(os.path.join(DATA, CSV_IN))
# agregado: media y desv. por (n,p)
agg = df.groupby(["n", "p"]).agg(
    T_compute=("T_compute", "mean"), T_compute_std=("T_compute", "std"),
    T_reduce=("T_reduce", "mean"), T_reduce_std=("T_reduce", "std"),
    T_total=("T_total", "mean"), T_total_std=("T_total", "std"),
    gflops=("gflops", "mean"), gflops_std=("gflops", "std"),
    best_seed=("best_seed", "first"), best_acc=("best_acc", "first"),
    flops_total=("flops_total", "first"),
).reset_index()
agg = agg.fillna(0)
agg.to_csv(os.path.join(DATA, SUMMARY_FILE), index=False)

# Ts secuencial por n = T_total con p=1
Ts = {n: float(agg[(agg.n == n) & (agg.p == 1)].T_total.iloc[0])
      for n in agg.n.unique()}
agg["speedup"] = agg.apply(lambda r: Ts[r.n] / r.T_total, axis=1)
agg["efficiency"] = agg.speedup / agg.p

N_LIST = sorted(agg.n.unique())
P_LIST = sorted(agg.p.unique())
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(N_LIST)))

# ----------------------------------------------------------------------
# 1) Tiempos: T_compute, T_reduce, T_total vs p
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, n in enumerate(N_LIST):
    sub = agg[agg.n == n].sort_values("p")
    ax.plot(sub.p, sub.T_total, "o-", color=colors[i], label=f"n={n}")
    ax.plot(sub.p, sub.T_compute, "x--", color=colors[i], alpha=0.6)
ax.plot([], [], "k-", label="$T_{total}$")
ax.plot([], [], "kx--", label="$T_{compute}$ (worker crítico)")
ax.set_xscale("log", base=2); ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.axvline(CORES, color="red", ls=":", alpha=0.7,
           label=f"núcleos físicos={CORES}")
ax.set_xlabel("Número de procesos p"); ax.set_ylabel("Tiempo (s)")
ax.set_title("Tiempo de ejecución vs p")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "tiempos.png")); plt.close()

# ----------------------------------------------------------------------
# 2) Speedup vs p (con ideal S=p)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(P_LIST, P_LIST, "k--", alpha=0.7, label="Ideal S=p")
for i, n in enumerate(N_LIST):
    sub = agg[agg.n == n].sort_values("p")
    ax.plot(sub.p, sub.speedup, "o-", color=colors[i], label=f"n={n}")
ax.axvline(CORES, color="red", ls=":", alpha=0.7, label=f"núcleos={CORES}")
ax.set_xscale("log", base=2); ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.set_xlabel("p"); ax.set_ylabel("Speedup S(p)=Ts/Tp")
ax.set_title("Speedup vs p"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "speedup.png")); plt.close()

# ----------------------------------------------------------------------
# 3) Eficiencia vs p
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="E=1 ideal")
for i, n in enumerate(N_LIST):
    sub = agg[agg.n == n].sort_values("p")
    ax.plot(sub.p, sub.efficiency, "o-", color=colors[i], label=f"n={n}")
ax.axvline(CORES, color="red", ls=":", alpha=0.7, label=f"núcleos={CORES}")
ax.set_xscale("log", base=2); ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.set_xlabel("p"); ax.set_ylabel("Eficiencia E(p)=S(p)/p")
ax.set_title("Eficiencia vs p"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "eficiencia.png")); plt.close()

# ----------------------------------------------------------------------
# 4) Curva teorica superpuesta.
#    Teoria:  T_p = a*(PS/p)*n  +  b*log2(p)            (Brent + reduccion arbol)
#    Ajuste por minimos cuadrados NO-NEGATIVOS (nnls): a,b >= 0.
#    Se omite intercepto libre (daria un c negativo sin sentido fisico).
#    El termino a*(PS/p)*n es el dominante (region lineal p<= ~ncpu);
#    b*log2(p) captura el overhead de sincronizacion que crece con p
#    (fork/join + barreras + reduccion arg-max). La saturacion para p>ncpu
#    se ANALIZA como Amdahl/oversubscription, no se absorbe en el modelo.
# ----------------------------------------------------------------------
from scipy.optimize import nnls
Xd = np.column_stack([(PS / agg.p) * agg.n, np.log2(agg.p)])
coef_nn, _ = nnls(Xd, agg.T_total.values)
a, b = coef_nn
yhat = Xd @ coef_nn
ss_res = np.sum((agg.T_total.values - yhat) ** 2)
ss_tot = np.sum((agg.T_total.values - agg.T_total.values.mean()) ** 2)
R2 = 1 - ss_res / ss_tot
agg["T_teorica"] = yhat

with open(os.path.join(DATA, FIT_FILE), "w") as f:
    f.write(f"Modelo: Tp = a*(PS/p)*n + b*log2(p)   (nnls, a,b>=0, sin intercepto)\n")
    f.write(f"a = {a:.4e} s/muestra   (factor de proporcionalidad de O(E*n*d*h))\n")
    f.write(f"b = {b:.4f} s            (overhead de sincronizacion por ronda, crece con log2 p)\n")
    f.write(f"R^2 = {R2:.4f}\n")
    f.write(f"(PS={PS})\n")

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, n in enumerate(N_LIST):
    sub = agg[agg.n == n].sort_values("p")
    ax.plot(sub.p, sub.T_total, "o", color=colors[i], label=f"n={n} (exp.)")
    ax.plot(sub.p, sub.T_teorica, "-", color=colors[i], alpha=0.8)
ax.plot([], [], "k-", label="Teórico (ajuste MC)")
ax.set_xscale("log", base=2); ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.set_xlabel("p"); ax.set_ylabel("Tiempo (s)")
ax.set_title(f"Experimental vs teórico  (R²={R2:.3f})")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "teorica.png")); plt.close()

# ----------------------------------------------------------------------
# 5) GFLOP/s vs p
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, n in enumerate(N_LIST):
    sub = agg[agg.n == n].sort_values("p")
    ax.errorbar(sub.p, sub.gflops, yerr=sub.gflops_std, fmt="o-",
                color=colors[i], label=f"n={n}", capsize=3)
ax.axvline(CORES, color="red", ls=":", alpha=0.7, label=f"núcleos={CORES}")
ax.set_xscale("log", base=2); ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.set_xlabel("p"); ax.set_ylabel("Rendimiento (GFLOP/s)")
ax.set_title("FLOPs/s vs p"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "gflops.png")); plt.close()

# ----------------------------------------------------------------------
# 6) Strong scaling: n=40000 fijo, T vs p (linea ideal Ts/p)
# ----------------------------------------------------------------------
n_big = max(N_LIST)
sub = agg[agg.n == n_big].sort_values("p")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sub.p, sub.T_total, "o-", label="Experimental")
ax.plot(sub.p, Ts[n_big] / sub.p, "k--", label=f"Ideal Ts/p")
ax.axvline(CORES, color="red", ls=":", alpha=0.7, label=f"núcleos={CORES}")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(P_LIST); ax.set_xticklabels(P_LIST)
ax.set_xlabel("p"); ax.set_ylabel("T_total (s)")
ax.set_title(f"Strong scaling (n={n_big})"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "strong.png")); plt.close()

# ----------------------------------------------------------------------
# 7) Weak scaling: n proporcional a p  -> eficiencia se mantiene ~cte?
#    diagonal n = 5000*p  para p en {1,2,4,8}
# ----------------------------------------------------------------------
weak_p = [1, 2, 4, 8]
weak_n = [5000 * p for p in weak_p]
wk = []
for p, n in zip(weak_p, weak_n):
    row = agg[(agg.n == n) & (agg.p == p)]
    if len(row):
        wk.append((p, n, float(row.T_total.iloc[0])))
wk = pd.DataFrame(wk, columns=["p", "n", "T_total"])
if len(wk) > 1:
    wk["eff_paralelo"] = wk.T_total.iloc[0] / wk.T_total  # ideal=cte
fig, ax = plt.subplots(figsize=(7, 4.5))
if len(wk) > 1:
    ax.plot(wk.p, wk.eff_paralelo, "o-", label="Eficiencia weak-scaling")
ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="Ideal (Gustafson)")
ax.set_xscale("log", base=2); ax.set_xticks(weak_p); ax.set_xticklabels(weak_p)
ax.set_xlabel("p (con n=5000·p)"); ax.set_ylabel("T(p)/T(1)")
ax.set_title("Weak scaling (carga por procesador constante)")
ax.set_ylim(0.85, 1.5); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, FIG_PREFIX + "weak.png")); plt.close()

# ----------------------------------------------------------------------
# Tabla resumen en LaTeX
# ----------------------------------------------------------------------
with open(os.path.join(DATA, SUMMARY_TEX), "w") as f:
    f.write("\\begin{tabular}{rrrrrrr}\n\\hline\n")
    f.write("n & p & $T_{total}$ (s) & Speedup & Eficiencia & GFLOP/s & best-acc \\\\\n\\hline\n")
    for _, r in agg.sort_values(["n", "p"]).iterrows():
        f.write(f"{int(r.n)} & {int(r.p)} & {r.T_total:.2f} & "
                f"{r.speedup:.2f} & {r.efficiency:.3f} & {r.gflops:.1f} & "
                f"{r.best_acc:.3f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

print("Figuras generadas en", FIG)
print("Ajuste teorico R^2 =", round(R2, 4), "| a,b =", [round(v, 4) for v in coef_nn])
print("Resumen en", os.path.join(DATA, "summary.csv"))
