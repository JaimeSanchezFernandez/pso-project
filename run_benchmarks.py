# run_benchmarks.py
"""
Ejecuta la suite completa de benchmarks para todas las estrategias.

Uso:
    python run_benchmarks.py
    python run_benchmarks.py --evaluador sequential
    python run_benchmarks.py --guardar
"""
import argparse
import logging
from experiments.benchmark_suite import BENCHMARK_INSTANCES
from experiments.runner import ejecutar_experimento
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from storage.persistence import guardar_resultado

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

EVALUADORES = {
    "sequential": SequentialEvaluator,
    "threading":  ThreadingEvaluator,
    "process":    ProcessEvaluator,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutar suite de benchmarks")
    parser.add_argument(
        "--evaluador", type=str, default="all",
        choices=["all", "sequential", "threading", "process"],
        help="Estrategia de evaluación. 'all' ejecuta las tres."
    )
    parser.add_argument("--guardar", action="store_true", help="Guardar resultados en results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.evaluador == "all":
        evaluadores_a_ejecutar = EVALUADORES
    else:
        evaluadores_a_ejecutar = {args.evaluador: EVALUADORES[args.evaluador]}

    total = len(BENCHMARK_INSTANCES) * len(evaluadores_a_ejecutar)
    print(f"\nEjecutando {total} experimentos...\n")

    num_run = 0
    for nombre_eval, ClaseEval in evaluadores_a_ejecutar.items():
        evaluador = ClaseEval()
        for instancia in BENCHMARK_INSTANCES:
            num_run += 1
            fn = instancia["fn"]
            semilla = instancia["seed"]
            nombre = instancia["name"]

            print(f"[{num_run}/{total}] {nombre} | {nombre_eval}", end=" ... ")

            resultado = ejecutar_experimento(
                funcion_objetivo=fn,
                evaluador=evaluador,
                semilla=semilla,
            )

            print(
                f"fitness={resultado['fitness_global']:.4e} | "
                f"{resultado['tiempo_total']:.3f}s"
            )

            if args.guardar:
                guardar_resultado(resultado)

    print(f"\nListo. {num_run} experimentos completados.")
    if args.guardar:
        print("Resultados guardados en results/")


if __name__ == "__main__":
    main()