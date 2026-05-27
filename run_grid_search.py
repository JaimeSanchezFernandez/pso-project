# run_grid_search.py
"""
Ejecuta un grid search de hiperparámetros del PSO.

Uso:
    python run_grid_search.py
    python run_grid_search.py --fn sphere --dim 10 --evaluador v0
    python run_grid_search.py --fn portfolio --evaluador v4
    python run_grid_search.py --fn rastrigin --dim 30 --evaluador all --guardar
"""
import argparse
import logging
import itertools
import time
import numpy as np

from objectives.functions    import Sphere, Rosenbrock, Rastrigin, Ackley
from objectives.portfolio    import PortfolioSharpe
from parallel.sequential     import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval   import ProcessEvaluator
from parallel.async_eval     import AsyncEvaluator
from parallel.numpy_eval     import NumpyEvaluator, NumpySwarm
from experiments.runner      import ejecutar_experimento
from storage.persistence     import guardar_resultado

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s")

FUNCIONES = {
    "sphere":     lambda dim: Sphere(dim=dim),
    "rosenbrock": lambda dim: Rosenbrock(dim=dim),
    "rastrigin":  lambda dim: Rastrigin(dim=dim),
    "ackley":     lambda dim: Ackley(dim=dim),
    "portfolio":  lambda dim: PortfolioSharpe(),
}

EVALUADORES = {
    "v0": lambda: SequentialEvaluator(),
    "v1": lambda: ThreadingEvaluator(),
    "v2": lambda: ProcessEvaluator(),
    "v3": lambda: AsyncEvaluator(latency_mean=0.005, seed=42),
    "v4": lambda: NumpyEvaluator(),
}

PARAM_GRID = {
    "w":              [0.4, 0.7, 0.9],
    "c1":             [1.0, 1.5, 2.0],
    "c2":             [1.0, 1.5, 2.0],
    "num_particulas": [30],
}

SEMILLAS = [42, 123, 456, 789, 1000]


def _ejecutar_una(fn, evaluador_nombre, params, semilla, max_iter):
    if evaluador_nombre == "v4":
        t0  = time.perf_counter()
        res = NumpySwarm(
            fn,
            num_particulas=params.get("num_particulas", 30),
            w=params.get("w", 0.7),
            c1=params.get("c1", 1.5),
            c2=params.get("c2", 1.5),
            max_iter=max_iter,
            semilla=semilla,
        ).ejecutar()
        t = time.perf_counter() - t0
        return {
            "funcion_objetivo":     repr(fn),
            "evaluador":            "NumpySwarm()",
            "fitness_global":       res["fitness_global"],
            "num_iteraciones":      res["num_iteraciones"],
            "tiempo_total":         t,
            "tiempo_evaluacion":    res["tiempo_evaluacion"],
            "tiempo_actualizacion": res["tiempo_actualizacion"],
            "overhead":             res["overhead"],
            **params, "semilla": semilla,
        }
    else:
        evaluador = EVALUADORES[evaluador_nombre]()
        resultado = ejecutar_experimento(
            funcion_objetivo=fn, evaluador=evaluador,
            w=params.get("w", 0.7), c1=params.get("c1", 1.5),
            c2=params.get("c2", 1.5),
            num_particulas=params.get("num_particulas", 30),
            max_iter=max_iter, semilla=semilla,
        )
        resultado.update(params)
        resultado["semilla"] = semilla
        return resultado


def busqueda_grid_completa(fn, evaluadores_nombres, param_grid, semillas, max_iter):
    claves        = list(param_grid.keys())
    combinaciones = list(itertools.product(*param_grid.values()))
    total         = len(combinaciones) * len(semillas) * len(evaluadores_nombres)

    print(f"  Combinaciones : {len(combinaciones)}")
    print(f"  Semillas      : {len(semillas)}")
    print(f"  Evaluadores   : {len(evaluadores_nombres)}")
    print(f"  Total runs    : {total}\n")

    resultados = []
    num_run    = 0

    for eval_nombre in evaluadores_nombres:
        for combo in combinaciones:
            params = dict(zip(claves, combo))
            for semilla in semillas:
                num_run += 1
                print(f"  [{num_run:>4}/{total}] {eval_nombre} | "
                      f"w={params['w']} c1={params['c1']} c2={params['c2']} "
                      f"seed={semilla}", end=" ... ", flush=True)
                r = _ejecutar_una(fn, eval_nombre, params, semilla, max_iter)
                r["_evaluador_key"] = eval_nombre
                resultados.append(r)
                print(f"fitness={r['fitness_global']:.4e}  {r['tiempo_total']:.3f}s")

    return resultados


def _imprimir_top(resultados, n=5):
    ordenados = sorted(resultados, key=lambda r: r["fitness_global"])
    print(f"\n  {'─'*72}")
    print(f"  Top {n} combinaciones")
    print(f"  {'─'*72}")
    print(f"  {'Eval':<6} {'w':>5} {'c1':>5} {'c2':>5} "
          f"{'seed':>6} {'fitness':>14} {'tiempo':>9}")
    print(f"  {'─'*72}")
    for r in ordenados[:n]:
        ev = r.get("_evaluador_key", "?").upper()
        print(f"  {ev:<6} {r['w']:>5.2f} {r['c1']:>5.2f} {r['c2']:>5.2f} "
              f"{r['semilla']:>6} {r['fitness_global']:>14.6e} "
              f"{r['tiempo_total']:>8.3f}s")
    print(f"  {'─'*72}")


def _imprimir_resumen_por_evaluador(resultados):
    from collections import defaultdict
    grupos = defaultdict(list)
    for r in resultados:
        grupos[r.get("_evaluador_key", "?")].append(r)

    print(f"\n  {'─'*50}")
    print(f"  Resumen por evaluador")
    print(f"  {'─'*50}")
    print(f"  {'Eval':<6} {'Fitness medio':>16} {'Tiempo medio':>14}")
    print(f"  {'─'*50}")
    for nombre, rs in sorted(grupos.items()):
        fits   = np.mean([r["fitness_global"] for r in rs])
        tiempo = np.mean([r["tiempo_total"]   for r in rs])
        print(f"  {nombre.upper():<6} {fits:>16.6e} {tiempo:>13.3f}s")
    print(f"  {'─'*50}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search PSO")
    parser.add_argument("--fn",        type=str, default="sphere",
                        choices=FUNCIONES.keys())
    parser.add_argument("--dim",       type=int, default=10)
    parser.add_argument("--evaluador", type=str, default="v0",
                        choices=list(EVALUADORES.keys()) + ["all"])
    parser.add_argument("--max_iter",  type=int, default=200)
    parser.add_argument("--guardar",   action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fn   = FUNCIONES[args.fn](args.dim)
    evaluadores_nombres = (
        list(EVALUADORES.keys()) if args.evaluador == "all" else [args.evaluador]
    )

    print(f"\n{'═'*55}")
    print(f"  Grid search PSO")
    print(f"{'═'*55}")
    print(f"  Función    : {repr(fn)}")
    print(f"  Evaluadores: {evaluadores_nombres}")
    print(f"  Grid       : w={PARAM_GRID['w']}  c1={PARAM_GRID['c1']}  c2={PARAM_GRID['c2']}")
    print(f"  Semillas   : {SEMILLAS}")
    print(f"{'═'*55}\n")

    resultados = busqueda_grid_completa(
        fn, evaluadores_nombres, PARAM_GRID, SEMILLAS, args.max_iter
    )
    _imprimir_top(resultados)
    _imprimir_resumen_por_evaluador(resultados)

    if args.guardar:
        for r in resultados:
            guardar_resultado(r)
        print(f"  {len(resultados)} resultados guardados en results/\n")


if __name__ == "__main__":
    main()