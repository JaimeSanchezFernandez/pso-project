# run_pso.py
"""
Ejecuta una única corrida de PSO con la configuración especificada por CLI.

Uso:
    python run_pso.py --fn sphere --dim 10 --evaluador sequential --semilla 42
    python run_pso.py --fn rastrigin --dim 30 --evaluador threading --num_particulas 50
    python run_pso.py --fn ackley --dim 10 --evaluador process --w 0.7 --c1 1.5 --c2 1.5
"""
import argparse
import logging
from objectives.functions import Sphere, Rosenbrock, Rastrigin, Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.runner import ejecutar_experimento
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutar PSO")
    parser.add_argument("--fn",             type=str,   default="sphere",     choices=FUNCIONES.keys())
    parser.add_argument("--dim",            type=int,   default=10)
    parser.add_argument("--evaluador",      type=str,   default="sequential", choices=EVALUADORES.keys())
    parser.add_argument("--num_particulas", type=int,   default=30)
    parser.add_argument("--w",              type=float, default=0.7)
    parser.add_argument("--c1",             type=float, default=1.5)
    parser.add_argument("--c2",             type=float, default=1.5)
    parser.add_argument("--max_iter",       type=int,   default=200)
    parser.add_argument("--semilla",        type=int,   default=42)
    parser.add_argument("--guardar",        action="store_true", help="Guardar resultado en results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    funcion_objetivo = FUNCIONES[args.fn](dim=args.dim)
    evaluador = EVALUADORES[args.evaluador]()

    resultado = ejecutar_experimento(
        funcion_objetivo=funcion_objetivo,
        evaluador=evaluador,
        num_particulas=args.num_particulas,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        max_iter=args.max_iter,
        semilla=args.semilla,
    )

    print(f"\n{'='*50}")
    print(f"Función:      {resultado['funcion_objetivo']}")
    print(f"Evaluador:    {resultado['evaluador']}")
    print(f"Fitness:      {resultado['fitness_global']:.6e}")
    print(f"Iteraciones:  {resultado['num_iteraciones']}")
    print(f"Tiempo total: {resultado['tiempo_total']:.4f}s")
    print(f"T. evaluación:{resultado['tiempo_evaluacion']:.4f}s")
    print(f"T. actualiz.: {resultado['tiempo_actualizacion']:.4f}s")
    print(f"Overhead:     {resultado['overhead']:.4f}s")
    print(f"{'='*50}\n")

    if args.guardar:
        ruta = guardar_resultado(resultado)
        print(f"Resultado guardado en: {ruta}")


if __name__ == "__main__":
    main()