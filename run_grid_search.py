# run_grid_search.py
"""
Ejecuta un grid search de hiperparámetros del PSO.

Uso:
    python run_grid_search.py
    python run_grid_search.py --fn sphere --dim 10 --evaluador sequential
    python run_grid_search.py --fn rastrigin --dim 30 --guardar
"""
import argparse
import logging
from objectives.functions import Sphere, Rosenbrock, Rastrigin, Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.grid_search import busqueda_grid
from storage.persistence import guardar_resultado

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

FUNCIONES = {
    "sphere":     Sphere,
    "rosenbrock": Rosenbrock,
    "rastrigin":  Rastrigin,
    "ackley":     Ackley,
}

EVALUADORES = {
    "sequential": SequentialEvaluator,
    "threading":  ThreadingEvaluator,
    "process":    ProcessEvaluator,
}

PARAM_GRID = {
    "w":              [0.4, 0.7, 0.9],
    "c1":             [1.0, 1.5, 2.0],
    "c2":             [1.0, 1.5, 2.0],
    "num_particulas": [30],
}

SEMILLAS = [42, 123, 456, 789, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search de hiperparámetros PSO")
    parser.add_argument("--fn",        type=str, default="sphere",     choices=FUNCIONES.keys())
    parser.add_argument("--dim",       type=int, default=10)
    parser.add_argument("--evaluador", type=str, default="sequential", choices=EVALUADORES.keys())
    parser.add_argument("--guardar",   action="store_true", help="Guardar resultados en results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    funcion_objetivo = FUNCIONES[args.fn](dim=args.dim)
    evaluador = EVALUADORES[args.evaluador]()

    print(f"\nGrid search | fn={args.fn}(d={args.dim}) | evaluador={args.evaluador}")
    print(f"Grid: {PARAM_GRID}")
    print(f"Semillas: {SEMILLAS}\n")

    resultados = busqueda_grid(
        funcion_objetivo=funcion_objetivo,
        evaluador=evaluador,
        param_grid=PARAM_GRID,
        semillas=SEMILLAS,
    )

    resultados_ordenados = sorted(resultados, key=lambda r: r["fitness_global"])

    print(f"\n{'='*70}")
    print(f"Top 5 combinaciones:")
    print(f"{'='*70}")
    print(f"{'w':>6} {'c1':>6} {'c2':>6} {'semilla':>8} {'fitness':>14} {'tiempo':>10}")
    print(f"{'-'*70}")
    for r in resultados_ordenados[:5]:
        print(
            f"{r['w']:>6.2f} {r['c1']:>6.2f} {r['c2']:>6.2f} "
            f"{r['semilla']:>8} {r['fitness_global']:>14.6e} "
            f"{r['tiempo_total']:>9.3f}s"
        )
    print(f"{'='*70}\n")

    if args.guardar:
        for r in resultados:
            guardar_resultado(r)
        print("Resultados guardados en results/")


if __name__ == "__main__":
    main()