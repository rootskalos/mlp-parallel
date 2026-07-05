"""
beta2_paralelo.py  --  VERSION BETA 2 (paralelizacion ingenua con
multiprocessing.Pool)

Primer intento de paralelizacion: se lanza un Pool de p procesos y se mapea
train_one sobre las PS semillas. Aqui los datos (Xtr, Ytr, ...) se pasan
EXPLICITAMENTE a cada worker (serializacion con pickle por cada llamada), lo
cual introduce un overhead de "comunicacion" alto en Python (spawn).

Esta version es correcta pero INEFICIENTE: muestra por que hay que mover los
datos a memoria compartida (fork + globales) en la beta 3.
"""
import time
import multiprocessing as mp
import mlp_common as cm


def _worker_init(Xtr, Ytr, Xte, Yte):
    cm.set_global_data(Xtr, Ytr, Xte, Yte)


def run(n_samples, p):
    Xtr, Xte, Ytr, Yte = cm.make_dataset(n_samples)
    cm.set_global_data(Xtr, Ytr, Xte, Yte)

    seeds = list(range(cm.PS))
    ctx = mp.get_context("spawn")              # beta2: spawn (serializa datos)
    t0 = time.perf_counter()                   # region paralela end-to-end
    with ctx.Pool(processes=p,
                  initializer=_worker_init,
                  initargs=(Xtr, Ytr, Xte, Yte)) as pool:
        results = pool.map(cm.train_one, seeds)
    t_total = time.perf_counter() - t0

    t_compute = max(r[2] for r in results)     # worker mas lento (pardo)
    t_reduce = t_total - t_compute             # overhead + comunicacion
    best = max(results, key=lambda r: r[1])
    return results, {
        "n": n_samples, "p": p, "ps": cm.PS,
        "T_split": 0.0, "T_compute": t_compute, "T_reduce": t_reduce,
        "T_total": t_total,
        "best_seed": best[0], "best_acc": best[1],
        "flops_total": cm.PS * cm.flops_per_mlp(len(Xtr)),
    }


if __name__ == "__main__":
    import sys, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    res, m = run(n, p)
    print(json.dumps(m, indent=2))
