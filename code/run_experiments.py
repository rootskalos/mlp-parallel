"""
run_experiments.py
Genera data/results.csv con la malla experimental:
  n  in {5000, 10000, 20000, 40000}     (tamano del problema)
  p  in {1, 2, 4, 8, 16, 32}            (procesos; p=1 es la referencia secuencial)
  repeticiones = REPS por punto (media +/- desv. tipica)

Escribe el CSV de forma incremental (flush) para no perder data si se
interrumpe. p=1 de la version paralela ES el tiempo secuencial Ts (mismo
codigo, sin duplicar). NO mide la lectura ni el split de datos.

Variables de entorno (utiles para correr en cluster):
  RESULTS_FILE  nombre del CSV de salida (default: results.csv)
  N_VALUES      lista de n separada por comas (default: 5000,10000,20000,40000)
  P_VALUES      lista de p separada por comas (default: 1,2,4,8,16,32)
  REPS          numero de repeticiones (default: 2)

Uso:  python run_experiments.py
"""
import os
# Limitar BLAS a 1 hilo por proceso ANTES de importar numpy/sklearn, para que
# la unica paralelizacion sea la nuestra (multiprocessing). En Linux (OpenBLAS/
# MKL) esto es determinante para un speedup limpio. En Mac (Accelerate) no afecta.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import beta3_paralelo_opt as b3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "data"))
OUT = os.environ.get("RESULTS_FILE", os.path.join(OUT_DIR, "results.csv"))

N_VALUES = [int(x) for x in os.environ.get(
    "N_VALUES", "5000,10000,20000,40000").split(",")]
P_VALUES = [int(x) for x in os.environ.get(
    "P_VALUES", "1,2,4,8,16,32").split(",")]
REPS = int(os.environ.get("REPS", "2"))

FIELDS = ["n", "p", "rep", "T_compute", "T_reduce", "T_total",
          "best_seed", "best_acc", "flops_total", "gflops"]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        f.flush()
        t_start = time.time()
        for n in N_VALUES:
            for p in P_VALUES:
                for rep in range(1, REPS + 1):
                    _, m = b3.run(n, p)
                    gflops = m["flops_total"] / m["T_total"] / 1e9
                    row = {
                        "n": n, "p": p, "rep": rep,
                        "T_compute": round(m["T_compute"], 4),
                        "T_reduce": round(m["T_reduce"], 4),
                        "T_total": round(m["T_total"], 4),
                        "best_seed": m["best_seed"],
                        "best_acc": round(m["best_acc"], 4),
                        "flops_total": m["flops_total"],
                        "gflops": round(gflops, 2),
                    }
                    w.writerow(row)
                    f.flush()
                    elapsed = time.time() - t_start
                    print(f"[{elapsed:6.0f}s] n={n:6d} p={p:2d} rep={rep}  "
                          f"T_compute={row['T_compute']:7.2f}s "
                          f"T_total={row['T_total']:7.2f}s "
                          f"gflops={row['gflops']:6.1f}  "
                          f"(best seed={row['best_seed']} acc={row['best_acc']})",
                          flush=True)
    print("\nListo. CSV en", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
