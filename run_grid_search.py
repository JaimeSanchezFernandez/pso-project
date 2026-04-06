# run_grid_search.py
"""
Ejecuta un grid search de hiperparámetros del PSO.

Uso:
    python run_grid_search.py
    python run_grid_search.py --fn sphere --dim 10 --evaluator sequential
    python run_grid_search.py --fn rastrigin --dim 30 --save
"""
import argparse
import logging
from objectives.sphere import Sphere
from objectives.rosenbrock import Rosenbrock
from objectives.rastrigin import Rastrigin
from objectives.ackley import Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.grid_search import run_grid_search
from storage.persistence import save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

FUNCTIONS = {
    "sphere":     Sphere,
    "rosenbrock": Rosenbrock,
    "rastrigin":  Rastrigin,
    "ackley":     Ackley,
}

EVALUATORS = {
    "sequential": SequentialEvaluator,
    "threading":  ThreadingEvaluator,
    "process":    ProcessEvaluator,
}

# Grid search reducido (3x3x3 con 5 seeds = 135 runs por función/evaluador)
DEFAULT_PARAM_GRID = {
    "w":           [0.4, 0.7, 0.9],
    "c1":          [1.0, 1.5, 2.0],
    "c2":          [1.0, 1.5, 2.0],
    "n_particles": [30],
}

DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search de hiperparámetros PSO")
    parser.add_argument("--fn",        type=str, default="sphere",     choices=FUNCTIONS.keys())
    parser.add_argument("--dim",       type=int, default=10)
    parser.add_argument("--evaluator", type=str, default="sequential", choices=EVALUATORS.keys())
    parser.add_argument("--save",      action="store_true", help="Guardar resultados en results/")
    return parser.parse_args()


def main():
    args = parse_args()

    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    evaluator = EVALUATORS[args.evaluator]()

    print(f"\nGrid search | fn={args.fn}(d={args.dim}) | evaluator={args.evaluator}")
    print(f"Grid: {DEFAULT_PARAM_GRID}")
    print(f"Seeds: {DEFAULT_SEEDS}\n")

    results = run_grid_search(
        objective_fn=objective_fn,
        evaluator=evaluator,
        param_grid=DEFAULT_PARAM_GRID,
        seeds=DEFAULT_SEEDS,
    )

    # Ordenar por gbest_fit para mostrar los mejores primeros
    results_sorted = sorted(results, key=lambda r: r["gbest_fit"])

    print(f"\n{'='*70}")
    print(f"Top 5 combinaciones:")
    print(f"{'='*70}")
    print(f"{'w':>6} {'c1':>6} {'c2':>6} {'seed':>6} {'gbest_fit':>14} {'tiempo':>10}")
    print(f"{'-'*70}")
    for r in results_sorted[:5]:
        print(
            f"{r['w']:>6.2f} {r['c1']:>6.2f} {r['c2']:>6.2f} "
            f"{r['seed']:>6} {r['gbest_fit']:>14.6e} {r['elapsed_seconds']:>9.3f}s"
        )
    print(f"{'='*70}\n")

    if args.save:
        for r in results:
            save_result(r)
        print(f"Resultados guardados en results/")


if __name__ == "__main__":
    main()