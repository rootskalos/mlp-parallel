"""
beta1_secuencial.py  --  VERSION BETA 1 (referencia secuencial, Algoritmo 1)

Entrena las PS semillas en un bucle secuencial puro y conserva el modelo de
mayor exactitud. Sirve como:
  * baseline de tiempo secuencial Ts (para speedup),
  * verificacion de correctitud (oráculo de accuracy).

NO paraleliza nada: es el punto de partida del desarrollo.
"""
import time
import mlp_common as cm


def run(n_samples):
    """Devuelve (resultados, metricas). metricas incluye Ts (tiempo secuencial
    de la region de computo) y T_split."""
    Xtr, Xte, Ytr, Yte = cm.make_dataset(n_samples)

    t_split0 = time.perf_counter()
    cm.set_global_data(Xtr, Ytr, Xte, Yte)
    t_split = time.perf_counter() - t_split0

    results = []
    t_compute0 = time.perf_counter()           # region paralelizable
    for seed in range(cm.PS):
        results.append(cm.train_one(seed))
    t_compute = time.perf_counter() - t_compute0

    best = max(results, key=lambda r: r[1])
    return results, {
        "n": n_samples, "p": 1, "ps": cm.PS,
        "T_split": t_split, "T_compute": t_compute, "T_reduce": 0.0,
        "T_total": t_compute,
        "best_seed": best[0], "best_acc": best[1],
        "flops_total": cm.PS * cm.flops_per_mlp(len(Xtr)),
    }


if __name__ == "__main__":
    import sys, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    res, m = run(n)
    print(json.dumps(m, indent=2))
    accs = [r[1] for r in res]
    print(f"acc media={np:=.4f} min={min(accs):.4f} max={max(accs):.4f}"
          .replace("np:", f"{sum(accs)/len(accs):.4f}"))
