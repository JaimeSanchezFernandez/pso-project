# make_viz.py
"""
Genera visualizaciones a partir de los resultados guardados.

Uso:
    python make_viz.py --type convergencia
    python make_viz.py --type speedup
    python make_viz.py --type enjambre --fn sphere --dim 2
    python make_viz.py --type todas_funciones --dim 10
    python make_viz.py --type boxplot --dim 10
"""
import argparse
import logging
import numpy as np
from objectives.functions import Sphere, Rosenbrock, Rastrigin, Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.runner import ejecutar_experimento
from viz.convergence import plot_convergence, plot_speedup, plot_convergence_all_functions, plot_boxplot
from viz.swarm_plot import animate_swarm_2d

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
    "V0_Sequential": SequentialEvaluator,
    "V1_Threading":  ThreadingEvaluator,
    "V2_Process":    ProcessEvaluator,
}

SEMILLAS = [42, 123, 456, 789, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar visualizaciones PSO")
    parser.add_argument(
        "--type", type=str, default="convergencia",
        choices=["convergencia", "speedup", "enjambre", "todas_funciones", "boxplot"],
        help="Tipo de visualización"
    )
    parser.add_argument("--fn",      type=str, default="sphere", choices=FUNCIONES.keys())
    parser.add_argument("--dim",     type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--guardar", action="store_true", help="Guardar figura en results/")
    return parser.parse_args()


def viz_convergencia(args: argparse.Namespace) -> None:
    """Compara curvas de convergencia de V0, V1 y V2."""
    funcion_objetivo = FUNCIONES[args.fn](dim=args.dim)
    historiales: dict[str, list[float]] = {}

    for nombre, ClaseEval in EVALUADORES.items():
        evaluador = ClaseEval()
        resultado = ejecutar_experimento(
            funcion_objetivo=funcion_objetivo,
            evaluador=evaluador,
            semilla=args.semilla,
        )
        historiales[nombre] = resultado["historial_fitness"]
        print(
            f"{nombre}: fitness={resultado['fitness_global']:.4e} | "
            f"t_total={resultado['tiempo_total']:.3f}s | "
            f"t_eval={resultado['tiempo_evaluacion']:.3f}s | "
            f"t_update={resultado['tiempo_actualizacion']:.3f}s | "
            f"overhead={resultado['overhead']:.3f}s"
        )

    ruta = f"results/convergencia_{args.fn}_d{args.dim}.png" if args.guardar else None
    plot_convergence(
        historiales,
        title=f"Convergencia — {args.fn} d={args.dim}",
        output_path=ruta,
    )


def viz_speedup(args: argparse.Namespace) -> None:
    """Compara speedup de V0, V1 y V2."""
    funcion_objetivo = FUNCIONES[args.fn](dim=args.dim)
    tiempos: dict[str, float] = {}

    for nombre, ClaseEval in EVALUADORES.items():
        evaluador = ClaseEval()
        resultado = ejecutar_experimento(
            funcion_objetivo=funcion_objetivo,
            evaluador=evaluador,
            semilla=args.semilla,
        )
        tiempos[nombre] = resultado["tiempo_total"]
        print(f"{nombre}: {resultado['tiempo_total']:.3f}s")

    ruta = f"results/speedup_{args.fn}_d{args.dim}.png" if args.guardar else None
    plot_speedup(
        tiempos,
        title=f"Speedup — {args.fn} d={args.dim}",
        output_path=ruta,
    )


def viz_todas_funciones(args: argparse.Namespace) -> None:
    """Convergencia de las 4 funciones benchmark en un solo plot 2x2."""
    resultados: dict[str, dict[str, list[float]]] = {}

    for nombre_fn, ClaseFn in FUNCIONES.items():
        funcion_objetivo = ClaseFn(dim=args.dim)
        resultados[f"{nombre_fn} d={args.dim}"] = {}

        for nombre_eval, ClaseEval in EVALUADORES.items():
            evaluador = ClaseEval()
            resultado = ejecutar_experimento(
                funcion_objetivo=funcion_objetivo,
                evaluador=evaluador,
                semilla=args.semilla,
            )
            resultados[f"{nombre_fn} d={args.dim}"][nombre_eval] = resultado["historial_fitness"]
            print(f"{nombre_fn} d={args.dim} | {nombre_eval}: fitness={resultado['fitness_global']:.4e}")

    ruta = f"results/todas_funciones_d{args.dim}.png" if args.guardar else None
    plot_convergence_all_functions(
        resultados,
        title=f"Convergencia por función — d={args.dim}",
        output_path=ruta,
    )


def viz_boxplot(args: argparse.Namespace) -> None:
    """Boxplot del fitness final sobre múltiples semillas."""
    resultados_por_estrategia: dict[str, dict[str, list[float]]] = {}

    for nombre_fn, ClaseFn in FUNCIONES.items():
        funcion_objetivo = ClaseFn(dim=args.dim)
        resultados_por_estrategia[f"{nombre_fn} d={args.dim}"] = {
            nombre: [] for nombre in EVALUADORES
        }

        for semilla in SEMILLAS:
            for nombre_eval, ClaseEval in EVALUADORES.items():
                evaluador = ClaseEval()
                resultado = ejecutar_experimento(
                    funcion_objetivo=funcion_objetivo,
                    evaluador=evaluador,
                    semilla=semilla,
                )
                resultados_por_estrategia[f"{nombre_fn} d={args.dim}"][nombre_eval].append(
                    resultado["fitness_global"]
                )
                print(f"{nombre_fn} | {nombre_eval} | semilla={semilla}: fitness={resultado['fitness_global']:.4e}")

    ruta = f"results/boxplot_d{args.dim}.png" if args.guardar else None
    plot_boxplot(
        resultados_por_estrategia,
        title=f"Distribución fitness final — d={args.dim} ({len(SEMILLAS)} semillas)",
        output_path=ruta,
    )


def viz_enjambre(args: argparse.Namespace) -> None:
    """Genera animación 2D del enjambre (solo para dim=2)."""
    if args.dim != 2:
        print("La animación del enjambre solo está disponible para dim=2.")
        return

    from core.swarm import Enjambre
    from core.stopcriteria import Estancamiento

    funcion_objetivo = FUNCIONES[args.fn](dim=2)
    evaluador = SequentialEvaluator()

    enjambre = Enjambre(
        funcion_objetivo=funcion_objetivo,
        evaluador=evaluador,
        num_particulas=30,
        semilla=args.semilla,
        criterio_parada=Estancamiento(paciencia=50),
    )

    resultado = enjambre.ejecutar()

    ruta = f"results/enjambre_{args.fn}_d2.gif" if args.guardar else None
    animate_swarm_2d(
        objective_fn=funcion_objetivo,
        position_history=resultado["historial_posiciones"],
        gbest_history=resultado["historial_mejor_global"],
        output_path=ruta,
    )


def main() -> None:
    args = parse_args()

    if args.type == "convergencia":
        viz_convergencia(args)
    elif args.type == "speedup":
        viz_speedup(args)
    elif args.type == "enjambre":
        viz_enjambre(args)
    elif args.type == "todas_funciones":
        viz_todas_funciones(args)
    elif args.type == "boxplot":
        viz_boxplot(args)


if __name__ == "__main__":
    main()