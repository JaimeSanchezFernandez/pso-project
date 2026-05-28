# Documento de diseño — PSO Project

Documento breve de decisiones de arquitectura, trade-offs y limitaciones.
Para el análisis experimental completo, ver `docs/informe_final.md`.

## 1. Arquitectura

El proyecto se organiza en módulos con responsabilidades separadas:

```
core/         Motor PSO: Enjambre, Particle, Topologia, CriterioParada
objectives/   Funciones benchmark (Sphere, Rosenbrock, Rastrigin, Ackley)
              + caso de uso real (PortfolioSharpe)
parallel/     Estrategias de evaluación de fitness V0-V4
experiments/  Runner, GridSearch, BenchmarkSuite (orquestación)
storage/      Persistencia JSON + CSV y carga de resultados
viz/          Visualización: convergencia, speedup, boxplot, enjambre 2D/3D
```

### Principio central: desacoplamiento por inyección de dependencias

El motor PSO (`Enjambre`) recibe la función objetivo y el evaluador como
parámetros del constructor. El core no sabe nada de cómo se evalúa el
fitness ni de qué función se optimiza:

```python
enjambre = Enjambre(
    funcion_objetivo=Sphere(dim=10),   # qué se optimiza
    evaluador=NumpyEvaluator(),         # cómo se evalúa (V0-V4)
    num_particulas=30, w=0.7, c1=1.5, c2=1.5,
)
```

Esto permite comparar las cinco estrategias de paralelismo de forma justa:
el algoritmo es idéntico, solo cambia la estrategia de evaluación inyectada.
Es la implementación del requisito del enunciado: "el core del PSO sea el
mismo y solo cambie la estrategia de evaluación".

### Abstracciones clave

- **FitnessEvaluator**: interfaz común de las cinco variantes. Método
  `evaluate(positions, objective_fn) -> fitnesses`.
- **ObjectiveFunction**: clase base de las funciones objetivo. Son callables
  (`fn(x)`) con bounds y dimensión.
- **Topologia**: define la vecindad del enjambre (implementado: gbest).
- **CriterioParada**: cuándo detener (MaxIteraciones, Tolerancia, Estancamiento).

## 2. Decisiones de diseño y trade-offs

### Estrategia de límites: clamp

Cuando una partícula sale del espacio de búsqueda, se recorta al límite
(`np.clip`) y su velocidad en esa dimensión se anula. Alternativas
consideradas: reflect (puede oscilar) y penalty (requiere calibrar peso).
Se eligió clamp por simplicidad y estabilidad numérica.

### Restricción de cartera: normalización (proyección al simplex)

La restricción Σwᵢ = 1 de la cartera se gestiona normalizando los pesos en
cada evaluación en lugar de añadir una penalización al fitness. Evita
calibrar parámetros adicionales y garantiza que toda solución es válida.

### spawn vs fork en multiprocessing (V2)

Se usa `spawn` porque `fork` copia el estado del proceso padre, incluyendo
mutexes internos de NumPy que pueden quedar bloqueados, causando deadlocks.
`spawn` crea un proceso limpio (más lento de iniciar pero seguro).

### Pool persistente + batching en V2

El `ProcessPoolExecutor` se crea una sola vez y se reutiliza en todas las
iteraciones (crear procesos con spawn es caro). Las partículas se agrupan en
batches para reducir el número de serializaciones pickle.

### Persistencia: JSON + CSV

JSON para el resultado completo (historial por iteración + metadata del
sistema), CSV para métricas finales compatibles con pandas. JSON sobre YAML
para resultados por ser nativo en Python; YAML se reserva para configuración.

### Reproducibilidad

Todas las ejecuciones aceptan `--semilla`. La semilla se pasa a
`np.random.default_rng()` y se registra en los resultados junto con la
versión de Python, SO y procesador.

## 3. Estrategias de evaluación (resumen)

| Versión | Mecanismo | Cuándo conviene |
|---------|-----------|-----------------|
| V0 Sequential | Bucle Python | Baseline; funciones baratas |
| V1 Threading | ThreadPoolExecutor | I/O-bound; libera GIL en NumPy |
| V2 Process | ProcessPoolExecutor + batching | CPU-bound costoso (≥50ms/eval) |
| V3 Async | asyncio.gather() | Evaluación con I/O externa (APIs) |
| V4 Numpy | Operaciones matriciales | Funciones vectorizables (mejor caso general) |

Resultado experimental (ver informe): V4 es la única estrategia que supera
al baseline (2.15× de media) en funciones matemáticas baratas. V1, V2 y V3
introducen overhead que no se compensa salvo en sus casos de uso específicos.

## 4. Caso de uso real: optimización de cartera

`PortfolioSharpe` maximiza el ratio de Sharpe de una cartera de 8 activos
(maximizar retorno ajustado por riesgo). El PSO mejora el Sharpe de 0.859
(cartera de pesos iguales) a 0.977, una mejora del 13.8%, reduciendo la
volatilidad del 16.2% al 14.6%. El resultado coincide con
`scipy.optimize`, lo que valida la correctitud de la implementación.

## 5. Limitaciones y trabajo futuro

- **Topología gbest**: converge prematuramente en funciones multimodales
  (Rastrigin d=30: fitness 103.67). Una topología local (lbest, anillo)
  mantendría mejor la diversidad.
- **V4 y funciones no vectorizables**: el fallback usa
  `np.apply_along_axis`, que es un bucle Python implícito. Funciones que
  acepten matrices directamente obtendrían mayor speedup.
- **Inercia constante**: decrementar w linealmente (0.9 → 0.4) durante la
  ejecución mejoraría el equilibrio exploración/explotación.
- **Datos de cartera**: se usan datos sintéticos realistas. En producción se
  sustituiría por datos reales vía yfinance (dependencia opcional ya incluida).