# Informe final — PSO con estrategias paralelas y optimización de cartera

## 1. Introducción

Este proyecto implementa Particle Swarm Optimization (PSO) en Python como
banco de pruebas para comparar cinco estrategias de evaluación de fitness:
secuencial (V0), hilos (V1), procesos (V2), asyncio (V3) y vectorización
NumPy (V4). Adicionalmente, se aplica el algoritmo a un problema real de
optimización de cartera de inversión maximizando el ratio de Sharpe.

El algoritmo PSO simula el comportamiento de una bandada de pájaros. Cada
partícula representa una solución candidata que se mueve en el espacio de
búsqueda guiada por su mejor posición histórica (componente cognitiva) y
por el mejor global del enjambre (componente social):

    v = w·v + c1·r1·(pBest - x) + c2·r2·(gBest - x)
    x = x + v

## 2. Metodología experimental

### 2.1 Configuración base

| Parámetro | Valor |
|-----------|-------|
| Partículas | 30 |
| Iteraciones | 200 |
| Inercia w | 0.7 |
| Coef. cognitivo c1 | 1.5 |
| Coef. social c2 | 1.5 |
| Semillas | 42, 123, 456, 789, 1000 |

### 2.2 Funciones benchmark

| Función | Dim | Bounds | Mínimo global |
|---------|-----|--------|---------------|
| Sphere | 2, 10, 30 | [-5.12, 5.12]^d | f(0,...,0) = 0 |
| Rosenbrock | 2, 10, 30 | [-2.048, 2.048]^d | f(1,...,1) = 0 |
| Rastrigin | 2, 10, 30 | [-5.12, 5.12]^d | f(0,...,0) = 0 |
| Ackley | 2, 10, 30 | [-32.768, 32.768]^d | f(0,...,0) = 0 |

### 2.3 Medición de tiempos

Se usa time.perf_counter() con resolución de nanosegundos. Cada
experimento mide por separado: tiempo_evaluacion (dentro del evaluador),
tiempo_actualizacion (velocidades, posiciones y clamp) y overhead.
Todas las mediciones se realizaron en el mismo equipo (Windows, Python 3.11)
para garantizar la comparabilidad.

## 3. Resultados

### 3.1 Tiempos por estrategia (Sphere d=10, 30 partículas, 200 iter.)

| Estrategia | Tiempo total | Speedup vs V0 | Nota |
|------------|-------------|---------------|------|
| V0 Sequential | 0.202s | 1.0× (ref.) | Baseline |
| V1 Threading | 0.784s | 0.26× | GIL limita CPU-bound |
| V2 Process | 5.098s | 0.04× | Overhead de IPC domina |
| V3 Async | 3.234s | 0.06× (lat. sim. 5ms) | Solo útil con I/O real |
| V4 Numpy | 0.067s | 3.01× | Mejor para CPU vectorizable |

Promedios sobre las 12 instancias benchmark (4 funciones × 3 dimensiones):

| Estrategia | Tiempo medio | Speedup medio vs V0 |
|------------|-------------|---------------------|
| V0 Sequential | 0.246s | 1.00× |
| V1 Threading | 0.835s | 0.29× |
| V2 Process | 4.230s | 0.06× |
| V3 Async | 3.500s | 0.07× |
| V4 Numpy | 0.114s | 2.15× |

V4 es la única estrategia paralela que supera al baseline en este hardware.
Las demás introducen un overhead (gestión de hilos, IPC, event loop) que en
funciones matemáticas baratas supera ampliamente el beneficio del paralelismo.

### 3.2 Calidad del fitness (Sphere, V0 secuencial)

| Dimensión | Fitness alcanzado | Converge |
|-----------|------------------|---------|
| d=2 | 3.5×10⁻²² | Sí (prácticamente exacto) |
| d=10 | 1.0×10⁻¹⁰ | Sí |
| d=30 | 4.9×10⁻³ | Parcialmente |

El fitness es idéntico en V0, V1, V2 y V3 con la misma semilla, porque el
evaluador no altera el algoritmo, solo cómo se calcula el fitness. V4 difiere
mínimamente (Sphere d=2: 2.9×10⁻²³) porque el orden de las operaciones
matriciales genera una secuencia distinta de números aleatorios.

El aumento de dimensión dificulta la convergencia: el volumen del espacio
crece exponencialmente (maldición de la dimensionalidad).

### 3.3 Funciones multimodales (Rastrigin, Ackley)

PSO con topología gbest converge prematuramente en Rastrigin:
- d=2: fitness = 0.0 (óptimo exacto). El espacio es pequeño y la diversidad inicial basta.
- d=10: fitness = 8.10. Queda atrapado en un mínimo local.
- d=30: fitness = 103.67. El enjambre pierde diversidad rápidamente.

El problema es estructural de gbest: cuando una partícula cae en un mínimo
local, toda la población se atrae hacia él y se pierde la diversidad necesaria
para escapar. Ackley muestra el mismo patrón aunque más suave (d=30: 3.91).

### 3.4 Análisis detallado de V3 (asyncio)

V3 se ejecutó con una latencia simulada de 5ms por partícula (asyncio.sleep).
Con funciones matemáticas locales, V3 es más lento que V0 porque el overhead
del event loop por 200 iteraciones no se compensa con ningún beneficio (no hay
esperas I/O reales que solapar).

El valor de V3 aparece cuando la evaluación de fitness es realmente I/O-bound.
Análisis teórico con N=30 partículas y latencia de red de 10ms:

| Modo | Tiempo por iteración |
|------|---------------------|
| Secuencial | 30 × 10ms = 300ms |
| asyncio.gather() | max(latencias) ≈ 13ms |
| Speedup | ~23× |

En este escenario (API de precios, servicio de simulación remoto), V3 sería
la estrategia más eficiente con diferencia.

### 3.5 Análisis detallado de V4 (NumPy vectorizado)

| Función (d=30) | Tiempo V0 | Tiempo V4 | Speedup |
|----------------|-----------|-----------|---------|
| Sphere | 0.226s | 0.071s | 3.18× |
| Rosenbrock | 0.275s | 0.129s | 2.13× |
| Rastrigin | 0.265s | 0.120s | 2.21× |
| Ackley | 0.303s | 0.155s | 1.95× |

V4 logra un speedup medio de 2.15× sobre todas las instancias, siendo la única
estrategia paralela que mejora el baseline. El beneficio proviene de eliminar
el bucle Python sobre partículas: todas las operaciones (velocidades,
posiciones, evaluación) se realizan sobre matrices completas que BLAS/LAPACK
ejecuta de forma optimizada.

### 3.6 Grid search de hiperparámetros

Se ejecutó un grid search de 27 combinaciones (w, c1, c2) × 5 semillas = 135
ejecuciones sobre Sphere d=10. Mejores combinaciones por fitness medio:

| w | c1 | c2 | Fitness medio (5 semillas) |
|---|----|----|----------------------------|
| 0.7 | 1.0 | 1.0 | 1.37×10⁻¹⁸ |
| 0.4 | 1.5 | 2.0 | 4.93×10⁻¹⁸ |
| 0.7 | 1.5 | 1.0 | 4.29×10⁻¹⁷ |
| 0.4 | 2.0 | 2.0 | 4.22×10⁻¹⁴ |
| 0.7 | 1.0 | 1.5 | 6.23×10⁻¹³ |

Efecto de la inercia w (media sobre todas las combinaciones c1, c2 y semillas):

| Inercia w | Fitness medio |
|-----------|---------------|
| 0.4 | 5.40×10⁻² |
| 0.7 | 4.83×10⁻⁴ |
| 0.9 | 2.78×10⁰ |

La inercia intermedia (w=0.7) ofrece el mejor equilibrio: suficiente
exploración sin perder capacidad de converger. La inercia alta (w=0.9)
degrada gravemente el rendimiento (fitness medio ~2.78) porque las partículas
mantienen demasiada velocidad y oscilan sin estabilizarse en el óptimo. La
mejor combinación individual encontrada fue w=0.4, c1=2.0, c2=1.5, semilla=42
con fitness 5.6×10⁻²².

## 4. Caso de uso real: optimización de cartera

### 4.1 Formulación

Se construye una cartera de 8 activos (AAPL, MSFT, GOOGL, AMZN, JPM,
JNJ, XOM, BRK-B) maximizando el ratio de Sharpe:

    S(w) = (μ_p - r_f) / σ_p
    μ_p = w^T · μ          (retorno anualizado)
    σ_p = √(w^T · Σ · w)   (volatilidad anualizada)
    r_f = 0.04             (tasa libre de riesgo)

PSO minimiza -S(w). La restricción Σwᵢ = 1 se gestiona normalizando
los pesos en cada evaluación (proyección sobre el simplex estándar).

### 4.2 Resultados (50 partículas, 500 iteraciones, semilla=42)

| Cartera | Retorno | Volatilidad | Sharpe |
|---------|---------|-------------|--------|
| Pesos iguales (1/N) | 17.94% | 16.23% | 0.8589 |
| PSO optimizada | 18.24% | 14.57% | 0.9771 |
| Mejora | +0.30pp | -1.66pp | +13.8% |

Las cinco estrategias (V0-V4) convergen exactamente al mismo Sharpe (0.9771),
confirmando que el evaluador no altera el resultado, solo el tiempo de cálculo:

| Estrategia | Sharpe | Tiempo |
|------------|--------|--------|
| V0 Sequential | 0.9771 | 1.068s |
| V1 Threading | 0.9771 | 3.120s |
| V2 Process | 0.9771 | 8.027s |
| V3 Async | 0.9771 | 8.711s |
| V4 Numpy | 0.9771 | 0.538s ◄ más rápido |

V4 es 2× más rápido que V0 también en este caso de uso, confirmando el patrón
observado en los benchmarks.

Composición de la cartera óptima:

| Activo | Peso | Sector |
|--------|------|--------|
| JNJ | 37.5% | Salud (baja volatilidad) |
| AAPL | 21.5% | Tech (alto retorno) |
| JPM | 17.0% | Financiero (diversificación) |
| AMZN | 14.5% | Tech (alto retorno) |
| XOM | 9.5% | Energía (baja correlación con tech) |
| MSFT, GOOGL, BRK-B | 0.0% | Excluidos |

El PSO concentra peso en activos con buena relación retorno/riesgo y baja
correlación entre sí. Reduce la volatilidad del 16.23% al 14.57% mientras
mantiene (e incluso mejora ligeramente) el retorno, que es exactamente el
objetivo de la optimización de Markowitz.

### 4.3 Comparación con scipy.optimize

| Método | Sharpe | Tiempo | Tipo |
|--------|--------|--------|------|
| PSO V4 | 0.9771 | 0.40s | Metaheurístico global |
| scipy DE | 0.9771 | 2.07s | Metaheurístico global |
| scipy SLSQP | 0.9771 | 0.005s | Local con gradiente |

Los tres métodos encuentran el mismo óptimo. Esto valida que el PSO converge
al óptimo global real, no a un mínimo local. SLSQP es el más rápido al ser un
método local que aprovecha la estructura cuasi-convexa del problema, pero PSO
y DE tienen la ventaja de no requerir gradientes ni un buen punto inicial, lo
que los hace más robustos en problemas con múltiples óptimos locales.

## 5. Discusión crítica

### GIL y threading (V1)

El GIL de CPython hace que V1 sea siempre peor que V0 para funciones
CPU-bound puras. En los benchmarks, V1 fue ~3.4× más lento que V0 de media.
Las operaciones NumPy liberan el GIL parcialmente, pero el overhead de crear
y sincronizar hilos supera ese beneficio para enjambres de 30-50 partículas.

### IPC y multiprocessing (V2)

El overhead de serialización (pickle) es el cuello de botella de V2. Cada
batch de partículas debe serializarse, enviarse por pipe al proceso hijo,
deserializarse, evaluarse y devolverse. En los benchmarks V2 fue ~17× más
lento que V0. Para funciones simples (~1μs por evaluación), el round-trip de
IPC (~1ms) supone un overhead de factor 1000.

V2 sería la estrategia óptima si la función objetivo tardara ≥ 50ms por
evaluación: simulaciones CFD, entrenamiento de redes como fitness, validación
cruzada real.

### asyncio (V3)

asyncio no mejora el rendimiento de CPU, sino el throughput de I/O. En los
benchmarks (sin I/O real) fue de los más lentos, pero su ventaja aparece
cuando la evaluación requiere esperar respuestas externas. En ese contexto un
solo hilo maneja N peticiones concurrentes con latencia ≈ max(tᵢ) en lugar de
Σtᵢ, con speedups teóricos superiores a 20×.

### Vectorización NumPy (V4)

V4 es la estrategia con mejor relación beneficio/complejidad: 2.15× de speedup
medio sin IPC, sin GIL, sin overhead de comunicación. El único requisito es
que la función objetivo sea vectorizable. Fue la única estrategia que mejoró
el baseline tanto en benchmarks como en el caso de uso real.

### Trade-off general

| Función objetivo | Estrategia recomendada |
|-----------------|----------------------|
| Matemática simple (<1ms) | V4 (NumPy) |
| Matemática costosa (>50ms) | V2 (Procesos) |
| I/O externa (API, BD) | V3 (asyncio) |
| I/O + CPU intensiva | V2 + asyncio combinados |

## 6. Recomendaciones

1. Usar V4 por defecto para funciones vectorizables. Fue la única estrategia
   que superó al baseline en este hardware (2.15× de media).

2. V3 para arquitecturas orientadas a servicios: si la función objetivo es un
   microservicio o API externa, asyncio escala a cientos de partículas sin
   aumentar el tiempo de pared.

3. Usar inercia w=0.7 como valor por defecto: el grid search confirma que es
   el mejor equilibrio entre exploración y explotación. Evitar w=0.9.

4. Añadir topología lbest para funciones multimodales: gbest pierde diversidad
   y queda atrapado en mínimos locales (Rastrigin d=30: fitness 103.67).

5. Para cartera real: sustituir los datos sintéticos por datos reales de
   yfinance y añadir restricciones de concentración máxima por activo
   (wᵢ ≤ 0.4) para forzar mayor diversificación.

## 7. Conclusión

El proyecto demuestra que no existe una estrategia de paralelismo
universalmente óptima para PSO. La elección depende del coste y la naturaleza
de la función objetivo:

- Funciones baratas y vectorizables → V4 (NumPy): la única que ganó al baseline.
- Funciones costosas en CPU → V2 (Procesos): paralelismo real que amortiza el IPC.
- Funciones con I/O externa → V3 (asyncio): solapamiento de esperas.
- Funciones baratas no vectorizables → V0 (secuencial): el overhead de paralelizar no compensa.

Los datos experimentales confirman que para las funciones benchmark usadas
(matemáticas y baratas), V4 es la mejor opción con 2.15× de speedup medio,
mientras que V1, V2 y V3 son contraproducentes por su overhead.

El caso de uso de cartera demuestra la aplicabilidad práctica del PSO: mejora
el ratio de Sharpe un 13.8% respecto a la cartera de pesos iguales, reduciendo
la volatilidad del 16.23% al 14.57%. La coincidencia del resultado con
scipy.optimize valida la correctitud de la implementación.