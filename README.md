# PSO Project — Particle Swarm Optimization

Implementación modular de PSO (Particle Swarm Optimization) en Python,
con comparativa de estrategias de evaluación secuencial, concurrente y paralela.

## Estructura del proyecto

pso_project/
├── core/           # Motor PSO (Swarm, Particle, Topología, Criterios de parada)
├── objectives/     # Funciones benchmark (Sphere, Rosenbrock, Rastrigin, Ackley)
├── parallel/       # Estrategias de evaluación (V0, V1, V2)
├── experiments/    # Runner, Grid Search, Suite de benchmarks
├── storage/        # Persistencia JSON+CSV y carga de resultados
├── viz/            # Visualización de convergencia y animación del enjambre
├── config/         # Configuración YAML
├── tests/          # Tests unitarios (16 tests)
└── results/        # Resultados guardados (generado al ejecutar)
## Instalación
```bash
pip install numpy matplotlib pandas pytest
```

## Uso

### Ejecutar una sola corrida de PSO
```bash
python run_pso.py --fn sphere --dim 10 --evaluator sequential
python run_pso.py --fn rastrigin --dim 30 --evaluator threading
python run_pso.py --fn ackley --dim 10 --evaluator process --save
```

### Ejecutar la suite completa de benchmarks
```bash
python run_benchmarks.py                        # todas las estrategias
python run_benchmarks.py --evaluator sequential # solo V0
python run_benchmarks.py --save                 # guardar resultados
```

### Ejecutar grid search de hiperparámetros
```bash
python run_grid_search.py --fn sphere --dim 10 --evaluator sequential
python run_grid_search.py --fn rastrigin --dim 30 --save
```

### Generar visualizaciones
```bash
python make_viz.py --type convergence --fn sphere --dim 10
python make_viz.py --type speedup --fn rastrigin --dim 10
python make_viz.py --type swarm --fn sphere --dim 2
```

### Ejecutar los tests
```bash
python -m pytest tests/ -v
```

## Estrategias de evaluación

| Versión | Clase | Descripción |
|---------|-------|-------------|
| V0 | `SequentialEvaluator` | Bucle Python estándar. Baseline de referencia. |
| V1 | `ThreadingEvaluator` | `ThreadPoolExecutor`. Limitado por el GIL en CPU-bound puro. Puede mejorar si la función objetivo usa NumPy (libera GIL). |
| V2 | `ProcessEvaluator` | `ProcessPoolExecutor` con batching. Paralelismo real en múltiples cores. Overhead de IPC (pickle) relevante en enjambres pequeños. |

## Funciones benchmark

| Función | Mínimo global | Características |
|---------|--------------|-----------------|
| Sphere | f(0,...,0) = 0 | Unimodal, separable. Verifica correctitud. |
| Rosenbrock | f(1,...,1) = 0 | Valle estrecho y curvo. Difícil de seguir. |
| Rastrigin | f(0,...,0) = 0 | Multimodal. Muchos mínimos locales. |
| Ackley | f(0,...,0) = 0 | Multimodal con mínimo global rodeado de locales. |

## Decisiones de diseño

- **Estrategia de límites**: clamp. Las partículas que salen del espacio
  de búsqueda se recortan al límite más cercano y su velocidad se anula
  en esa dimensión. Simple, estable y predecible.
- **Desacoplamiento**: el core PSO recibe el evaluador por inyección de
  dependencia. Cambiar de V0 a V2 es cambiar un parámetro, no tocar código.
- **Persistencia**: JSON para resultados completos (con historial),
  CSV para métricas finales (compatible con pandas/Excel).
- **Reproducibilidad**: todas las ejecuciones aceptan `--seed`.
  La seed se registra en todos los resultados guardados.