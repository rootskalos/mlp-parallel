# Guía para la Exposición Oral — Proyecto #7 (MLP paralelo)

> Lee esto hasta entenderlo. El profe bombardea a preguntas; esta guía cubre lo
> que va a preguntar. La oral vale **6 pts** en la final.

---

## 1. El pitch en 60 segundos (empieza siempre con esto)

> "Paralelizamos el entrenamiento de **32 redes neuronales (MLP) con semillas
> distintas**, conservando el modelo de mayor exactitud. Es un problema
> **embarazosamente paralelo**: cada MLP se entrena de forma independiente y solo
> se sincroniza al final con una reducción *arg-máx* (que medimos y resulta ser
> <1.5% del tiempo). Lo modelamos como **PRAM CREW de memoria compartida**
> (análogo a OpenMP) y lo implementamos en Python con **multiprocessing** (fork).
> Medimos en **dos plataformas**: una MacBook M5 (10 núcleos) y el **cluster
> Khipu de UTEC** (nodo de 32 núcleos). Resultados: en la Mac el speedup satura
> en ~5 (limitado por los 10 núcleos y el ancho de banda de memoria), y en
> **Khipu alcanza ~10.6** — el doble. Eso valida que el algoritmo escala mejor
> con más hardware, como predice la teoría."

---

## 2. Conceptos que DEBES dominar (si dudas aquí, pierde puntos)

### PRAM y el modelo CREW
- **PRAM** = *Parallel Random Access Machine*: varios procesadores comparten una
  memoria global y operan en pasos sincrónicos.
- **CREW** = *Concurrent Read, Exclusive Write*: varios procesadores pueden
  **leer** la misma celda a la vez, pero **escribir** es exclusivo (un sólo por
  celda por paso).
- En nuestro algoritmo: los p procesadores **leen concurrentemente** el dataset
  (CREW). Las escrituras del ganador son **exclusivas** (cada P_k escribe en su
  celda propia; la reducción en árbol escribe sin contención).
- **¿Por qué no CRCW o EREW?** No necesitamos escritura concurrente (CRCW) ni
  imponemos lectura exclusiva (EREW). CREW es justo lo necesario y suficiente.

### `pardo` (parallel-do)
- Marca qué bucles se ejecutan **en paralelo**. En el Algoritmo 2: el
  entrenamiento (paso 2) y cada ronda de la reducción (pasos 3+).
- **Punto clave de la corrección:** en el parcial llamamos "Broadcast" al paso
  de leer los datos. **Eso estaba mal**: en memoria compartida NO hay broadcast.
  Los datos ya están en memoria global; los procesadores los **leen
  concurrentemente**. Por eso lo renombramos a "lectura concurrente CREW" y no
  lo paralelizamos aparte.

### Work-Time (WT) y Teorema de Brent
- **Work** $W$ = número total de operaciones = tiempo secuencial $T_s$.
- **Span** $T_\infty$ = camino crítico (tiempo con procesadores infinitos).
- **Brent:** $T_p \le W/p + T_\infty$. El tiempo con p procesadores está acotado.
- En nuestro caso: $W = O(p_s \cdot E \cdot n \cdot d \cdot h)$,
  $T_\infty = O(E \cdot n \cdot d \cdot h)$ (1 MLP cuando p=p_s).

### WT-optimalidad
- Un algoritmo es WT-óptimo si $C_p = T_p \cdot p = O(W)$.
- Se cumple si $p \le W/T_\infty = O(p_s)$. Como siempre usamos $p \le p_s=32$,
  **nuestro algoritmo es WT-óptimo**.

### Memoria compartida vs distribuida (¡PREGUNTA FAVORITA!)
- **Compartida (OpenMP / multiprocessing):** los procesadores ven la misma
  memoria. Comunicación = leer/escribir variables. Aquí los datos son
  compartidos y de **solo lectura**, así que basta con leer referencias
  (copy-on-write del fork).
- **Distribuida (MPI):** cada proceso tiene su memoria; comunican con mensajes
  (send/recv, scatter/broadcast). Requeriría **copiar el dataset a cada
  proceso** (memoria ×p + overhead).
- **Justificación de por qué compartida:** el problema es embarassingly
  parallel, la comunicación es mínima (solo la reducción final) y los datos son
  de solo lectura. La memoria compartida es **más eficiente** aquí: cero copia
  de datos. MPI no aportaría nada y sumaría overhead.

### El GIL de Python
- El *Global Interpreter Lock* impide que 2 hilos ejecuten bytecode Python a la
  vez. Por eso **no usamos hilos** (`threading`): no habría paralelismo real.
- Usamos **procesos** (`multiprocessing`): cada proceso tiene su intérprete y su
  GIL. Es el análogo real a OpenMP en Python.

### fork-join
- Patrón: el proceso raíz **crea (fork)** p trabajadores → cada uno entrena sus
  MLP → se **sincronizan (join)** para la reducción. Es exactamente el modelo
  fork-join de OpenMP (`#pragma omp parallel`).

---

## 3. Métricas: que te las pregunte, que las sepas

| Métrica | Fórmula | En el proyecto |
|---|---|---|
| Tiempo secuencial | $T_s$ | $O(p_s \cdot E \cdot n \cdot d \cdot h)$, p=1 |
| Tiempo paralelo | $T_p$ | $\approx a\frac{p_s}{p}n + b\log_2 p$ |
| Speedup | $S(p)=T_s/T_p$ | ideal → p; medido → Mac ~5, **Khipu ~10.6** |
| Eficiencia | $E(p)=S(p)/p$ | ≤1; cae con p (Amdahl) |
| Costo | $C_p=T_p\cdot p$ | = O(W) → WT-óptimo |

**FLOPs:** $\text{FLOPs por MLP} = 4 \cdot E \cdot h \cdot (d+c) \cdot n$.
Contados a partir de la arquitectura (no por hw-counters). Rendimiento en
GFLOP/s = FLOPs_total / T_p.

---

## 4. Las leyes de escalabilidad (pregunta segura)

- **Amdahl (strong scaling):** $n$ fijo, aumentas $p$. $S \le 1/(f + (1-f)/p)$
  donde $f$ es la fracción secuencial. **El speedup tiene techo** = 1/f.
  → Techo medido: **~5 en la Mac (10 cores), ~10.6 en Khipu (32 cores)**. Fijado
  por el número de núcleos + el ancho de banda de memoria. $E(p)$ **cae**.
- **Gustafson (weak scaling):** aumentas $n$ junto con $p$ (carga por
  procesador constante). $E(p)$ **se mantiene ≈ 1**.
- **Pregunta del enunciado: "¿E se mantiene constante?"**
  → Respuesta matizada: **NO en strong scaling** (cae por Amdahl + contención de
  memoria). **SÍ en weak scaling** (Gustafson). Lo mostramos en dos gráficas.

---

## 5. Preguntas que el profe va a hacer (con respuesta)

**P: ¿Por qué el speedup se satura (no llega a p)?**
R: Por dos causas: (1) el número de **núcleos físicos** pone un techo (la Mac
tiene 10 y satura en ~5; Khipu tiene 32 lógicos ≈ 16 físicos y llega a ~10.6);
(2) **contención de ancho de banda de memoria**, porque el MLP es *memory-bound*
(cada MLP se desacelera cuando hay muchos corriendo). La reducción arg-máx es
<1.5%, así que NO es la comunicación. **Lo clave**: mismo algoritmo, y el speedup
pasa de ~5 (Mac) a ~10.6 (Khipu) al subir los núcleos — valida que escala.

**P: ¿Por qué corrieron en dos máquinas?**
R: Para validar la escalabilidad en hardware distinto. Mismo código, mismo
dataset, dos plataformas de memoria compartida (Mac M5 10 cores y cluster Khipu
Xeon 32 cores). Mostrar que el speedup se duplica al duplicar los núcleos es una
validación experimental fuerte.

**P: ¿Por qué no usaron MPI?**
R: El problema es embarassingly parallel y los datos son compartidos de solo
lectura. En MPI habría que difundir el dataset a cada proceso (scatter), gastando
memoria y tiempo. En memoria compartida cada proceso hereda los datos por
copy-on-write del fork: cero copia. MPI no aportaría ventaja.

**P: ¿Es realmente OpenMP si usan Python?**
R: El MODELO es el de OpenMP (memoria compartida, fork-join, variables
compartidas). La IMPLEMENTACIÓN es multiprocessing porque el GIL impide hilos
reales. Conceptualmente es lo mismo: p workers que comparten memoria y se
sincronizan al final.

**P: ¿Por qué en Khipu (32 cores) el speedup llega solo a ~10.6 y no a 32?**
R: Porque el Xeon Gold 6130 tiene **hyper-threading**: 32 CPUs lógicas pero ~16
físicas, así que el speedup real máximo está entre 10 y 16. Además, la contención
de memoria limita. Aun así, 10.6 es el **doble** que en la Mac (5), lo que
confirma que el algoritmo escala con el hardware.

**P: ¿Cómo verificaron que el paralelo da lo mismo que el secuencial?**
R: El código es determinista (random_state fijo). Comparamos las 32 exactitudes
y el modelo ganador entre sec y par: **idénticos** (seed ganadora y acc).
Validación de correctitud pasada.

**P: ¿Qué es el "worker crítico" que miden?**
R: El tiempo de la región paralela = el worker más lento (el que más MLPs/tarda).
Medir la suma de tiempos sería incorrecto (eso sería trabajo, no tiempo paralelo).
El tiempo paralelo real es el **máximo** entre procesos.

**P: ¿Por qué no midieron la lectura de datos?**
R: Porque la observación del profesor fue no incluir la I/O en la medición de
tiempos. El split de datos y la lectura son setup; medimos solo la región
paralelizable (entrenamiento) + la reducción.

**P: ¿De dónde sale el término $\log_2 p$ en la fórmula de T_p?**
R: De la reducción *arg-máx* en árbol binario: con p resultados locales, hace
$\log_2 p$ rondas para encontrar el máximo global. En el parcial no estaba y por
eso la fórmula no cuadraba del todo con la experimentación; ahora sí (R² alto).

**P: ¿El algoritmo es escalable?**
R: Sí. Strong scaling: lineal hasta los 10 núcleos. Weak scaling: eficiente
(E≈1). El límite es el hardware (10 cores), no el algoritmo.

**P: ¿Qué mejorarían?**
R: (1) Cola de trabajo dinámica para MLPs heterogéneos. (2) Paralelizar dentro
del MLP con BLAS multihilo/GPU (data parallelism dentro de cada ensemble).
(3) Reimplementar en C++/OpenMP para quitar el overhead de proceso de Python.

**P: ¿Por qué PS=32 semillas?**
R: Para que p∈{1,2,4,8,16,32} divida exactamente (cada proceso entera PS/p
MLPs). Es el caso del enunciado. El reparto en chunks balancea igual si no
dividiera.

**P: ¿Cómo calcularon los FLOPs?**
R: A partir de la arquitectura (d, h, c, E, n): forward + backward ≈
$4 E h (d+c) n$ FLOPs por MLP. No usamos hw-counters (no accesibles de forma
portable); contamos las operaciones del costo conocido del MLP. Reportamos
GFLOP/s = FLOPs_total / T_p.

---

## 6. Trampas y errores que el profe caza

- **Decir "broadcast"** en memoria compartida → te para. Decir **"lectura
  concurrente CREW"** o **"memoria compartida de solo lectura"**.
- **Confundir tiempo paralelo con suma de tiempos** → el tiempo paralelo es el
  **máximo** (worker crítico), no la suma (eso es trabajo W).
- **Decir que E(p) siempre vale 1** → falso. Cae con p (Amdahl). Solo se
  mantiene en weak scaling.
- **Decir "usamos OpenMP"** sin aclarar → di "modelo de memoria compartida tipo
  OpenMP, implementado con multiprocessing por el GIL".
- **Olvidar justificar memoria compartida** → te lo pregunta. Respuesta:
  comunicación mínima + datos de solo lectura = la opción más eficiente.

---

## 7. Cómo estructurar la exposición (10 min)

1. **(1 min)** Problema y por qué paralelo (pitch).
2. **(2 min)** PRAM CREW: muestra el Algoritmo 2, señala `pardo`, lectura
   concurrente, reducción arg-máx. **Justifica memoria compartida**.
3. **(2 min)** Métricas: W, T_∞, T_p (con el log), speedup, eficiencia, FLOPs.
   Señala la fórmula con el término $\log_2 p$.
4. **(3 min)** Resultados: dos plataformas (Mac 10 cores → ~5x; **Khipu 32 cores
   → ~10.6**), la comparación valida la escalabilidad. Eficiencia (responde
   "¿E constante?" → strong cae, weak se mantiene), validación teórica (R²=0.989),
   GFLOP/s.
5. **(1 min)** Mejoras propuestas.
6. **(1 min)** Conclusiones.

**Regla de oro:** si te preguntan algo que no sabes, di "no lo medimos / no lo
consideramos, pero..." y ofréce la respuesta cualitativa correcta. No inventes
números.
