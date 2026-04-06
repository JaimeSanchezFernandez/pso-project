# run_pso.py
"""
Ejecuta una única corrida de PSO con la configuración especificada por CLI.

Uso:
    python run_pso.py --fn sphere --dim 10 --evaluator sequential --seed 42
    python run_pso.py --fn rastrigin --dim 30 --evaluator threading --n_particles 50
    python run_pso.py --fn ackley --dim 10 --evaluator process --w 0.7 --c1 1.5 --c2 1.5
"""
import argparse
import logging
import json
from objectives.sphere import Sphere
from objectives.rosenbrock import Rosenbrock
from objectives.rastrigin import Rastrigin
from objectives.ackley import Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.runner import run_experiment
from io.persistence import save_result

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


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecutar PSO")
    parser.add_argument("--fn",          type=str,   default="sphere",     choices=FUNCTIONS.keys())
    parser.add_argument("--dim",         type=int,   default=10)
    parser.add_argument("--evaluator",   type=str,   default="sequential", choices=EVALUATORS.keys())
    parser.add_argument("--n_particles", type=int,   default=30)
    parser.add_argument("--w",           type=float, default=0.7)
    parser.add_argument("--c1",          type=float, default=1.5)
    parser.add_argument("--c2",          type=float, default=1.5)
    parser.add_argument("--max_iter",    type=int,   default=200)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save",        action="store_true", help="Guardar resultado en results/")
    return parser.parse_args()


def main():
    args = parse_args()

    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    evaluator = EVALUATORS[args.evaluator]()

    result = run_experiment(
        objective_fn=objective_fn,
        evaluator=evaluator,
        n_particles=args.n_particles,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Función:    {result['objective_fn']}")
    print(f"Evaluador:  {result['evaluator']}")
    print(f"gbest_fit:  {result['gbest_fit']:.6e}")
    print(f"iteraciones:{result['n_iterations']}")
    print(f"tiempo:     {result['elapsed_seconds']:.4f}s")
    print(f"{'='*50}\n")

    if args.save:
        path = save_result(result)
        print(f"Resultado guardado en: {path}")


if __name__ == "__main__":
    main()