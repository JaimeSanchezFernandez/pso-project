# PSO Project — Particle Swarm Optimization

Implementación modular de PSO (Particle Swarm Optimization) en Python,
con comparativa de cinco estrategias de evaluación (secuencial, hilos,
procesos, asyncio y vectorización NumPy) y un caso de uso real de
optimización de cartera de inversión.

## Estructura del proyecto

pso-project/
├── core/           # Motor PSO (Swarm, Particle, Topología, Criterios de parada)
├── objectives/     # Funciones benchmark + caso de uso real (cartera)
├── parallel/       # Estrategias de evaluación V0–V4
├── experiments/    # Runner, Grid Search, Suite de benchmarks
├── storage/        # Persistencia JSON+CSV y carga de resultados
├── viz/            # Visualización de convergencia y animación del enjambre
├── config/         # Configuración YAML
├── tests/          # Tests unitarios (49 tests)
└── results/        # Resultados guardados (generado al ejecutar)

## Instalación

pip install numpy matplotlib pandas pytest

## Comandos de uso

### Una sola corrida de PSO

python run_pso.py --fn sphere --dim 10 --evaluador sequential
python run_pso.py --fn rastrigin --dim 30 --evaluador threading
python run_pso.py --fn ackley --dim 10 --evaluador process --guardar

### Suite completa de benchmarks

python run_benchmarks.py                          # todas las estrategias
python run_benchmarks.py --evaluador sequential   # solo V0
python run_benchmarks.py --guardar                # guardar resultados

### Caso de uso real — optimización de cartera

python run_portfolio.py                           # todas las estrategias
python run_portfolio.py --evaluador v4            # solo V4 (más rápido)
python run_portfolio.py --particulas 50 --iter 500 --semilla 42
python run_portfolio.py --guardar

### Grid search de hiperparámetros

python run_grid_search.py --fn sphere --dim 10 --evaluador v0
python run_grid_search.py --fn portfolio --evaluador v4
python run_grid_search.py --fn rastrigin --dim 30 --evaluador all --guardar

### Visualizaciones

python make_viz.py --type convergence --fn sphere --dim 10
python make_viz.py --type speedup --fn rastrigin --dim 10
python make_viz.py --type swarm --fn sphere --dim 2

### Tests

python -m pytest tests/ -v

## Estrategias de evaluación (V0–V4)

| Versión | Clase                            | Mecanismo                  | Cuándo usar                                                    |
|---------|----------------------------------|----------------------------|----------------------------------------------------------------|
| V0      | SequentialEvaluator              | Bucle Python puro          | Baseline. Referencia de comparación.                          |
| V1      | ThreadingEvaluator               | ThreadPoolExecutor         | Funciones con I/O o que liberan el GIL (NumPy).               |
| V2      | ProcessEvaluator                 | ProcessPoolExecutor+batch  | Funciones CPU-bound costosas.                                  |
| V3      | AsyncEvaluator                   | asyncio.gather()           | Funciones que requieren llamadas externas (APIs, BD).          |
| V4      | NumpyEvaluator / NumpySwarm      | Operaciones matriciales    | Funciones vectorizables. 2–5x más rápido que V0.              |

## Funciones benchmark

| Función       | Mínimo global      | Características                                    |
|---------------|--------------------|----------------------------------------------------|
| Sphere        | f(0,...,0) = 0     | Unimodal, separable. Verifica correctitud.         |
| Rosenbrock    | f(1,...,1) = 0     | Valle estrecho y curvo. Difícil de seguir.         |
| Rastrigin     | f(0,...,0) = 0     | Multimodal. Muchos mínimos locales.                |
| Ackley        | f(0,...,0) = 0     | Multimodal con mínimo rodeado de locales.          |
| PortfolioSharpe | maximizar Sharpe | 8 activos reales. Restricción de suma unitaria.    |

## Caso de uso real: optimización de cartera

PortfolioSharpe minimiza el negativo del ratio de Sharpe sobre una
cartera de 8 activos (AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, BRK-B).

Ratio de Sharpe: (μ_p - r_f) / σ_p

- μ_p: retorno anualizado de la cartera
- r_f: tasa libre de riesgo (4%)
- σ_p: volatilidad anualizada

Los pesos se normalizan internamente (suman 1, sin posiciones cortas).

Resultado típico con 50 partículas y 500 iteraciones:

| Cartera             | Retorno | Volatilidad | Sharpe |
|---------------------|---------|-------------|--------|
| Pesos iguales (1/N) | 17.9%   | 16.2%       | 0.859  |
| PSO optimizada      | 18.2%   | 14.6%       | 0.977  |
| Mejora              |         |             | +13.8% |

## Decisiones de diseño

- Desacoplamiento: el motor PSO recibe el evaluador por inyección de
  dependencias. Cambiar de V0 a V4 es cambiar un parámetro, no tocar el core.
- Estrategia de límites: clamp. Las partículas que salen del espacio
  de búsqueda se recortan al límite y su velocidad se anula en esa dimensión.
- Restricción de cartera: normalización de pesos (proyección sobre el
  simplex estándar). Evita posiciones cortas sin penalizaciones.
- Persistencia: JSON para resultados completos, CSV para métricas finales.
- Reproducibilidad: todas las ejecuciones aceptan --semilla.
  La semilla se registra junto con versión Python, SO y procesador.