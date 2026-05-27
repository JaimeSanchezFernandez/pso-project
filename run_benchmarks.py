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
import time
from experiments.benchmark_suite import BENCHMARK_INSTANCES
from experiments.runner import ejecutar_experimento
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from parallel.async_eval import AsyncEvaluator
from parallel.numpy_eval import NumpySwarm
from storage.persistence import guardar_resultado

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

EVALUADORES = {
    "sequential": lambda: SequentialEvaluator(),
    "threading":  lambda: ThreadingEvaluator(),
    "process":    lambda: ProcessEvaluator(),
    "async":      lambda: AsyncEvaluator(latency_mean=0.005, seed=42),
    "numpy":      None,   # usa NumpySwarm directamente, no es un evaluador
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutar suite de benchmarks")
    parser.add_argument(
        "--evaluador", type=str, default="all",
        choices=["all", "sequential", "threading", "process", "async", "numpy"],
        help="Estrategia de evaluación. 'all' ejecuta las cinco."
    )
    parser.add_argument("--guardar", action="store_true", help="Guardar resultados en results/")
    return parser.parse_args()


def _ejecutar_instancia(nombre_eval, fn, semilla):
    """Ejecuta una instancia con cualquier estrategia, incluyendo V4 (NumpySwarm)."""
    if nombre_eval == "numpy":
        t0  = time.perf_counter()
        res = NumpySwarm(fn, num_particulas=30, max_iter=200, semilla=semilla).ejecutar()
        res["tiempo_total"] = time.perf_counter() - t0
        res["funcion_objetivo"] = repr(fn)
        res["evaluador"] = "NumpySwarm()"
        # pos_global es un ndarray; convertir a lista para que sea serializable a JSON
        if hasattr(res.get("pos_global"), "tolist"):
            res["pos_global"] = res["pos_global"].tolist()
        return res
    else:
        evaluador = EVALUADORES[nombre_eval]()
        return ejecutar_experimento(
            funcion_objetivo=fn,
            evaluador=evaluador,
            semilla=semilla,
        )


def main() -> None:
    args = parse_args()

    if args.evaluador == "all":
        nombres_eval = list(EVALUADORES.keys())
    else:
        nombres_eval = [args.evaluador]

    total = len(BENCHMARK_INSTANCES) * len(nombres_eval)
    print(f"\nEjecutando {total} experimentos...\n")

    num_run = 0
    for nombre_eval in nombres_eval:
        for instancia in BENCHMARK_INSTANCES:
            num_run += 1
            fn = instancia["fn"]
            semilla = instancia["seed"]
            nombre = instancia["name"]

            print(f"[{num_run}/{total}] {nombre} | {nombre_eval}", end=" ... ", flush=True)

            resultado = _ejecutar_instancia(nombre_eval, fn, semilla)

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