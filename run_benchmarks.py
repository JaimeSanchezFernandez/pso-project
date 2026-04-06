# run_benchmarks.py
"""
Ejecuta la suite completa de benchmarks para todas las estrategias.

Uso:
    python run_benchmarks.py
    python run_benchmarks.py --evaluator sequential
    python run_benchmarks.py --save
"""
import argparse
import logging
from experiments.benchmark_suite import BENCHMARK_INSTANCES
from experiments.runner import run_experiment
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from storage.persistence import save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

EVALUATORS = {
    "sequential": SequentialEvaluator,
    "threading":  ThreadingEvaluator,
    "process":    ProcessEvaluator,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecutar suite de benchmarks")
    parser.add_argument(
        "--evaluator", type=str, default="all",
        choices=["all", "sequential", "threading", "process"],
        help="Estrategia de evaluación. 'all' ejecuta las tres."
    )
    parser.add_argument("--save", action="store_true", help="Guardar resultados en results/")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.evaluator == "all":
        evaluators_to_run = EVALUATORS
    else:
        evaluators_to_run = {args.evaluator: EVALUATORS[args.evaluator]}

    total = len(BENCHMARK_INSTANCES) * len(evaluators_to_run)
    print(f"\nEjecutando {total} experimentos...\n")

    run_idx = 0
    for eval_name, EvalClass in evaluators_to_run.items():
        evaluator = EvalClass()
        for instance in BENCHMARK_INSTANCES:
            run_idx += 1
            fn = instance["fn"]
            seed = instance["seed"]
            name = instance["name"]

            print(f"[{run_idx}/{total}] {name} | {eval_name}", end=" ... ")

            result = run_experiment(
                objective_fn=fn,
                evaluator=evaluator,
                seed=seed,
            )

            print(f"gbest={result['gbest_fit']:.4e} | {result['elapsed_seconds']:.3f}s")

            if args.save:
                save_result(result)

    print(f"\nListo. {run_idx} experimentos completados.")
    if args.save:
        print("Resultados guardados en results/")


if __name__ == "__main__":
    main()