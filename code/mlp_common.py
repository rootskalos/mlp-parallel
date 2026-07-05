"""
mlp_common.py
Componentes compartidos del proyecto de Paralelizacion del entrenamiento de
multiples MLP con distintas semillas (Proyecto #7 - Redes Neuronales).

Modelo conceptual: PRAM CREW de memoria compartida, emulado en Python con
multiprocessing (fork) como analogia de OpenMP (el GIL impide paralelismo real
con hilos puros).

El problema es "embarazosamente paralelo": se entrenan PS perceptrones
multicapa, cada uno con una semilla distinta, y se conserva el de mayor
exactitud sobre el conjunto de prueba. Cada MLP se entrena con el dataset
COMPLETO (ensemble parallelism sobre semillas, NO particion de datos).
"""
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Hiperparametros del problema (fijos para toda la experimentacion)
# ----------------------------------------------------------------------
PS = 32                 # numero de semillas |S| = {0,1,...,PS-1}
D = 32                  # features de entrada (dimension d)
C = 10                  # numero de clases (salida)
H = 128                 # neuronas de la capa oculta
MAX_ITER = 100          # epocas (Tmax)
ALPHA = 1e-4            # termino de regularizacion L2 (alpha)
ETA = 1e-3              # learning rate inicial (eta)
TEST_SIZE = 0.2         # fraccion de test
DATASET_SEED = 7        # semilla de generacion del dataset (unica)

# Variables globales heredadas por fork (memoria compartida): evita
# serializar/copiar (Xtr, Ytr) a cada worker, fiel al modelo PRAM CREW.
_XTR = None
_YTR = None
_XTE = None
_YTE = None


def set_global_data(Xtr, Ytr, Xte, Yte):
    """Carga los arrays en el espacio global para que los workers fork los
    hereden por copy-on-write (modelo de memoria compartida)."""
    global _XTR, _YTR, _XTE, _YTE
    _XTR = Xtr
    _YTR = Ytr
    _XTE = Xte
    _YTE = Yte


def make_dataset(n_samples, seed=DATASET_SEED):
    """Genera un dataset de clasificacion sintetico reproducible de tamano n.
    Se divide UNA vez en train/test (Paso 1 del PRAM)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=D,
        n_informative=min(20, D),
        n_redundant=4,
        n_classes=C,
        n_clusters_per_class=2,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed,
                            stratify=y)


def flops_per_mlp(n_train):
    """FLOPs del entrenamiento de UN MLP (forward + backward por epoca y por
    muestra) derivados de la arquitectura, NO estimados por hw-counters.

    Capa oculta h: W1 (h x d).  Capa salida c: W2 (c x h).
      forward  : 2*d*h + 2*c*h   (MAC contado como 2 FLOP)
      backward : ~2*d*h + 2*c*h  (gradientes)
      total/epoca/muestra ~ 4*h*(d + c)
    Multiplicado por E epocas y n_train muestras.
    """
    return 4 * MAX_ITER * H * (D + C) * n_train


def train_one(seed):
    """Worker: entrena un MLP con la semilla dada sobre los datos globales
    (Xtr, Ytr) y evalua exactitud sobre (Xte, Yte).

    Devuelve (seed, accuracy, train_time, n_iters). El train_time mide SOLO
    el bloque de entrenamiento (region paralelizable), sin incluir lectura de
    datos ni I/O.
    """
    import time
    t0 = time.perf_counter()
    clf = MLPClassifier(
        hidden_layer_sizes=(H,),
        max_iter=MAX_ITER,
        alpha=ALPHA,
        learning_rate_init=ETA,
        solver="adam",
        random_state=seed,
        early_stopping=False,
    )
    clf.fit(_XTR, _YTR)
    t1 = time.perf_counter()
    acc = clf.score(_XTE, _YTE)
    return seed, float(acc), t1 - t0, int(clf.n_iter_)


def build_mlp(seed):
    """Constructor puro (sin entrenar) para validacion de determinismo."""
    return MLPClassifier(
        hidden_layer_sizes=(H,),
        max_iter=MAX_ITER,
        alpha=ALPHA,
        learning_rate_init=ETA,
        solver="adam",
        random_state=seed,
        early_stopping=False,
    )


def chunk_seeds(seeds, p):
    """Reparte las semillas en p grupos de tamano ceil/floor (maneja el caso
    en que PS no sea divisible entre p)."""
    seeds = list(seeds)
    n = len(seeds)
    base, extra = divmod(n, p)
    chunks, i = [], 0
    for k in range(p):
        size = base + (1 if k < extra else 0)
        chunks.append(seeds[i:i + size])
        i += size
    return [c for c in chunks if c]


if __name__ == "__main__":
    print(f"PS={PS} D={D} C={C} H={H} E={MAX_ITER} alpha={ALPHA} eta={ETA}")
    print(f"Nucleos logicos detectados: {os.cpu_count()}")
