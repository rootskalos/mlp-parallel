"""
fill_report.py
Inyecta los valores reales del experimento en informe/informe_final.tex,
reemplazando los marcadores %%...%%. Se ejecuta DESPUES de analyze.py.
"""
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TEX = os.path.join(ROOT, "informe", "informe_final.tex")
DATA = os.path.join(ROOT, "data")

# Usamos los datos de Khipu como resultados principales del informe
FIT = os.path.join(DATA, "fit_khipu.txt")
SUMMARY = os.path.join(DATA, "summary_khipu.csv")

with open(TEX, encoding="utf-8") as f:
    tex = f.read()

# --- constantes del ajuste teorico ---
fit = {}
with open(FIT) as f:
    for line in f:
        m = re.search(r"^(a|b|c)\s*=\s*([0-9.eE+-]+)", line)
        if m:
            fit[m.group(1)] = float(m.group(2))
        m2 = re.search(r"R\^2\s*=\s*([0-9.eE+-]+)", line)
        if m2:
            fit["R2"] = float(m2.group(1))


def sci(x):
    """Formato LaTeX en notacion cientifica con 2 decimales."""
    s = f"{x:.2e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"${mant}\\times 10^{{{exp}}}$"


# --- metricas agregadas (Khipu) ---
agg = pd.read_csv(SUMMARY)
Ts = {n: float(agg[(agg.n == n) & (agg.p == 1)].T_total.iloc[0])
      for n in agg.n.unique()}
agg["speedup"] = agg.apply(lambda r: Ts[r.n] / r.T_total, axis=1)
agg["efficiency"] = agg.speedup / agg.p

n_big = int(agg.n.max())
speedup_max = float(agg.speedup.max())          # maximo global (todos los n)
gflops_peak = float(agg.gflops.max())

repl = {
    "%%SPEEDUP_MAX%%": f"$S\\approx {speedup_max:.1f}$",
    "%%R2%%": f"{fit['R2']:.3f}",
    "%%A%%": sci(fit["a"]),
    "%%B%%": f"{fit['b']:.3f}",
    "%%GFLOPS_PEAK%%": f"{gflops_peak:.1f}",
}
for k, v in repl.items():
    tex = tex.replace(k, v)

# elimina cualquier %%marcador%% restante por seguridad
tex = re.sub(r"%%[^%]*%%", r"--", tex)

with open(TEX, "w", encoding="utf-8") as f:
    f.write(tex)

print("Informe actualizado con valores reales:")
for k, v in repl.items():
    print(f"  {k} -> {v}")
