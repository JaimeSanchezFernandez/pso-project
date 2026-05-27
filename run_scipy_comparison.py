# run_scipy_comparison.py
"""
Bonus: comparación de PSO vs scipy.optimize en las funciones benchmark
y en el caso de uso real de optimización de cartera.

scipy.optimize.differential_evolution es el método más comparable a PSO:
ambos son metaheurísticos poblacionales para optimización global sin gradientes.

Uso:
    python run_scipy_comparison.py
    python run_scipy_comparison.py --fn sphere --dim 10
    python run_scipy_comparison.py --caso portfolio
    python run_scipy_comparison.py --caso ambos
"""
import argparse
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

from objectives.functions import Sphere, Rosenbrock, Rastrigin, Ackley
from objectives.portfolio import PortfolioSharpe
from parallel.numpy_eval  import NumpySwarm

FUNCIONES = {
    "sphere":     lambda dim: Sphere(dim=dim),
    "rosenbrock": lambda dim: Rosenbrock(dim=dim),
    "rastrigin":  lambda dim: Rastrigin(dim=dim),
    "ackley":     lambda dim: Ackley(dim=dim),
}


def _separador(char="─", ancho=70):
    print(char * ancho)


def comparar_benchmark(fn_nombre: str, dim: int, semilla: int = 42) -> None:
    fn     = FUNCIONES[fn_nombre](dim)
    bounds = fn.bounds

    print(f"\n{'═'*70}")
    print(f"  {fn_nombre.capitalize()}  d={dim}")
    print(f"{'═'*70}")
    print(f"  {'Método':<35} {'Fitness':>14} {'Tiempo':>9} {'Evals':>8}")
    _separador()

    # --- PSO V4 (NumpySwarm) ---
    t0  = time.perf_counter()
    r   = NumpySwarm(fn, num_particulas=50, max_iter=500, semilla=semilla).ejecutar()
    t   = time.perf_counter() - t0
    n_eval_pso = 50 * 500
    print(f"  {'PSO V4 (NumpySwarm)':<35} {r['fitness_global']:>14.6e} {t:>8.3f}s {n_eval_pso:>8}")

    # --- scipy Differential Evolution ---
    n_eval_de = [0]
    def fn_contador(x):
        n_eval_de[0] += 1
        return fn(x)

    t0  = time.perf_counter()
    res = differential_evolution(
        fn_contador, bounds,
        seed=semilla, maxiter=500, popsize=15, tol=1e-8,
        workers=1,
    )
    t_de = time.perf_counter() - t0
    print(f"  {'scipy DE (differential_evolution)':<35} {res.fun:>14.6e} {t_de:>8.3f}s {n_eval_de[0]:>8}")

    # --- scipy Nelder-Mead (método local, sin gradiente) ---
    x0 = np.zeros(dim)
    t0 = time.perf_counter()
    res_nm = minimize(fn, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    t_nm = time.perf_counter() - t0
    print(f"  {'scipy Nelder-Mead (local, sin grad.)':<35} {res_nm.fun:>14.6e} {t_nm:>8.3f}s {res_nm.nfev:>8}")

    _separador()
    print(f"  Mínimo global conocido: 0.0  (óptimo en {'origen' if fn_nombre != 'rosenbrock' else '(1,...,1)'})")


def comparar_portfolio(semilla: int = 42) -> None:
    fn = PortfolioSharpe()

    print(f"\n{'═'*70}")
    print(f"  Optimización de cartera (maximizar Sharpe ratio)")
    print(f"{'═'*70}")
    print(f"  {'Método':<35} {'Sharpe':>8} {'Retorno':>9} {'Vol':>8} {'Tiempo':>9}")
    _separador()

    # Referencia: pesos iguales
    ref = fn.metricas(np.ones(fn.dim) / fn.dim)
    print(f"  {'Referencia (1/N)':<35} {ref['sharpe']:>8.4f} "
          f"{ref['retorno_anual']:>8.1%} {ref['volatilidad']:>7.1%}  {'—':>8}")

    # PSO V4
    t0  = time.perf_counter()
    r   = NumpySwarm(fn, num_particulas=50, max_iter=500, semilla=semilla).ejecutar()
    t   = time.perf_counter() - t0
    m   = fn.metricas(r["pos_global"])
    print(f"  {'PSO V4 (NumpySwarm)':<35} {m['sharpe']:>8.4f} "
          f"{m['retorno_anual']:>8.1%} {m['volatilidad']:>7.1%} {t:>8.3f}s")

    # scipy DE
    t0  = time.perf_counter()
    res = differential_evolution(
        fn, fn.bounds, seed=semilla, maxiter=500, popsize=15, tol=1e-8, workers=1,
    )
    t_de = time.perf_counter() - t0
    m_de = fn.metricas(res.x)
    print(f"  {'scipy DE (differential_evolution)':<35} {m_de['sharpe']:>8.4f} "
          f"{m_de['retorno_anual']:>8.1%} {m_de['volatilidad']:>7.1%} {t_de:>8.3f}s")

    # scipy SLSQP (método local con restricciones)
    def neg_sharpe(x):
        return fn(x)

    bounds_scipy = [(0, 1)] * fn.dim
    x0 = np.ones(fn.dim) / fn.dim

    t0 = time.perf_counter()
    res_slsqp = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds_scipy,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    t_slsqp = time.perf_counter() - t0
    m_slsqp = fn.metricas(res_slsqp.x)
    print(f"  {'scipy SLSQP (local, con restricc.)':<35} {m_slsqp['sharpe']:>8.4f} "
          f"{m_slsqp['retorno_anual']:>8.1%} {m_slsqp['volatilidad']:>7.1%} {t_slsqp:>8.3f}s")

    _separador()
    mejor_sharpe = max(m['sharpe'], m_de['sharpe'], m_slsqp['sharpe'])
    mejora = (mejor_sharpe - ref['sharpe']) / ref['sharpe'] * 100
    print(f"  Mejor Sharpe encontrado : {mejor_sharpe:.4f}  (+{mejora:.1f}% sobre referencia)")


def parse_args():
    parser = argparse.ArgumentParser(description="PSO vs scipy.optimize")
    parser.add_argument("--caso",    type=str, default="benchmark",
                        choices=["benchmark", "portfolio", "ambos"])
    parser.add_argument("--fn",      type=str, default="sphere",
                        choices=FUNCIONES.keys())
    parser.add_argument("--dim",     type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.caso in ("benchmark", "ambos"):
        if args.caso == "ambos":
            for fn_nombre in FUNCIONES:
                for dim in [2, 10, 30]:
                    comparar_benchmark(fn_nombre, dim, args.semilla)
        else:
            comparar_benchmark(args.fn, args.dim, args.semilla)

    if args.caso in ("portfolio", "ambos"):
        comparar_portfolio(args.semilla)

    print()


if __name__ == "__main__":
    main()