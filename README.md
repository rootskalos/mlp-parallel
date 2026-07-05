# Proyecto #7 — Redes Neuronales (Entrenamiento paralelo de múltiples MLP)

Computación Paralela y Distribuida (CS4052) — 2026-I
Integrantes: Kalos Lazo Mera · Aaron Cesar Aldair Navarro Mendoza · Fernando Alonso Usurin Arias

Paralelización del **entrenamiento de 32 MLP con semillas distintas** (selección
del modelo de mayor exactitud). Modelo **PRAM CREW de memoria compartida**
(análogo a OpenMP), implementado en Python con `multiprocessing` (fork).

## Estructura

```
final/
├── code/
│   ├── mlp_common.py          # hiperparámetros, dataset, worker, contador FLOPs
│   ├── beta1_secuencial.py    # BETA 1: referencia secuencial (Algoritmo 1)
│   ├── beta2_paralelo.py      # BETA 2: paralelo ingenuo (spawn, serializa datos)
│   ├── beta3_paralelo_opt.py  # BETA 3: optimizado (fork + memoria compartida + medición)
│   ├── run_experiments.py     # genera data/results.csv (malla n×p×reps)
│   ├── analyze.py             # genera figures/*.png y data/summary.{csv,tex}
│   └── fill_report.py         # inyecta valores reales en el informe .tex
├── data/
│   ├── results.csv            # mediciones crudas (n,p,rep,T_*,gflops,...)
│   ├── summary.csv            # agregado media±std por (n,p)
│   ├── summary.tex            # tabla para el informe
│   ├── fit.txt                # ajuste teórico (a,b,c,R²)
│   └── experiment.log         # log del runner
├── figures/                   # 7 gráficas (PNG)
├── informe/
│   └── informe_final.tex      # informe (compilar en Overleaf → PDF)
├── estudio/
│   └── guia_completa.tex      # guía de estudio TODO desde cero (24 págs)
├── oral/
│   └── guia_oral.md           # Q&A para la exposición oral
└── README.md
```

## Cómo reproducir

```bash
cd code
python -m venv ../.venv && source ../.venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib
python run_experiments.py     # ~40 min (deja data/results.csv)
python analyze.py             # genera figuras + summary + fit
python fill_report.py         # rellena el informe con los valores reales
```

## Hiperparámetros

| Símbolo | Valor | Descripción |
|---|---|---|
| p_s | 32 | semillas (|S|) |
| d | 32 | features de entrada |
| h | 128 | neuronas capa oculta |
| c | 10 | clases |
| E | 100 | épocas (max_iter) |
| α | 1e-4 | regularización L2 |
| η | 1e-3 | learning rate |
| p | {1,2,4,8,16,32} | procesos |
| n | {5k,10k,20k,40k} | muestras |

## Resultados clave (de la experimentación real, dos plataformas)

Experimentamos en dos plataformas de memoria compartida: **MacBook M5 (10
cores)** y **cluster Khipu de UTEC, nodo n003 (Xeon Gold 6130, 32 cores)**.

- **Speedup**: en la Mac satura en ~5 (limitado por los 10 núcleos); en **Khipu
  alcanza ~10.6** (el doble). La comparación valida que el algoritmo escala con
  el hardware.
- La reducción arg-máx es **<1.5%** del tiempo → patrón embarazosamente paralelo.
- Ajuste teórico (Khipu) `T_p = a·(p_s/p)·n + b·log₂p` con **R²=0.989**
  (a=2.42e-4 s/muestra, b=2.18 s).
- Eficiencia: cae en strong scaling (Amdahl), estable en weak scaling (Gustafson).
- GFLOP/s pico: ~53 (Mac), ~76 (Khipu).

> Datos reales medidos en Apple Silicon M5 (10 núcleos). Sin datos sintéticos.
