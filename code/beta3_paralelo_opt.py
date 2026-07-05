"""
beta3_paralelo_opt.py  --  VERSION BETA 3 (optimizada, version de experimentos)

Mejoras respecto a beta2:
  * fork como start method: los workers heredan los arrays (Xtr, Ytr, ...)
    por copy-on-write, evitando serializar/copiar datos en cada llamada.
    Esto es lo mas fiel al modelo PRAM de MEMORIA COMPARTIDA que justifica el
    proyecto (los procesadores leen concurrentemente de memoria global).
  * Reparto de semillas en chunks balanceados (ceil/floor) para que PS no
    tenga que ser divisible entre p.
  * Medicion separada y fina de T_compute (worker mas lento), T_reduce
    (argmax + recoger resultados) y overhead del Pool. NO se mide la lectura
    ni el split de datos (correccion del profesor: "no incluir lectura en la
    medicion de tiempos").
  * Reporta FLOPs totales del entrenamiento para derivar GFLOP/s.

Esta es la version usada para la experimentacion final.
"""
import time
import multiprocessing as mp
import mlp_common as cm


def _train_chunk(chunk):
    """Entrena todas las semillas de un chunk secuencialmente en un worker.
    Devuelve (resultados, tiempo_compute_local)."""
    t0 = time.perf_counter()
    out = [cm.train_one(s) for s in chunk]
    return out, time.perf_counter() - t0


def run(n_samples, p):
    Xtr, Xte, Ytr, Yte = cm.make_dataset(n_samples)
    cm.set_global_data(Xtr, Ytr, Xte, Yte)     # visibles por fork

    seeds = list(range(cm.PS))
    chunks = cm.chunk_seeds(seeds, p)          # reparto balanceado

    ctx = mp.get_context("fork")               # beta3: fork = memoria compartida

    # --- Region paralela: fork-join sobre los p procesadores ---
    t0 = time.perf_counter()
    partials = []
    with ctx.Pool(processes=p) as pool:
        for part in pool.map(_train_chunk, chunks):
            partials.append(part)
    t_total = time.perf_counter() - t0

    results = [r for part in partials for r in part[0]]
    t_compute_workers = [part[1] for part in partials]
    t_compute = max(t_compute_workers)         # tiempo de la rama critica (pardo)
    t_reduce = t_total - t_compute             # fork/join + argmax + overhead

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
