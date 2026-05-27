# make_viz.py
"""
Genera visualizaciones a partir de resultados del PSO.

Uso:
    python make_viz.py --type convergencia
    python make_viz.py --type speedup
    python make_viz.py --type enjambre --fn sphere --dim 2
    python make_viz.py --type superficie --fn rastrigin --dim 2
    python make_viz.py --type todas_funciones --dim 10
    python make_viz.py --type boxplot --dim 10
    python make_viz.py --type portfolio
"""
import argparse
import logging
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
from viz.convergence         import (
    plot_convergence,
    plot_speedup,
    plot_convergence_all_functions,
    plot_boxplot,
    plot_portfolio,
)
from viz.swarm_plot import animate_swarm_2d, animate_swarm_3d

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
    "V0_Sequential": lambda: SequentialEvaluator(),
    "V1_Threading":  lambda: ThreadingEvaluator(),
    "V2_Process":    lambda: ProcessEvaluator(),
    "V3_Async":      lambda: AsyncEvaluator(latency_mean=0.005, seed=42),
    "V4_Numpy":      None,   # usa NumpySwarm directamente
}

SEMILLAS = [42, 123, 456, 789, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar visualizaciones PSO")
    parser.add_argument(
        "--type", type=str, default="convergencia",
        choices=["convergencia", "speedup", "enjambre",
                 "todas_funciones", "boxplot", "portfolio", "superficie"],
    )
    parser.add_argument("--fn",      type=str, default="sphere", choices=FUNCIONES.keys())
    parser.add_argument("--dim",     type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--guardar", action="store_true")
    return parser.parse_args()


def _ejecutar(fn, nombre_eval, semilla, max_iter=200):
    """Ejecuta un experimento con cualquier evaluador incluyendo V4."""
    if nombre_eval == "V4_Numpy":
        t0  = time.perf_counter()
        res = NumpySwarm(fn, num_particulas=30, max_iter=max_iter, semilla=semilla).ejecutar()
        res["tiempo_total"] = time.perf_counter() - t0
        return res
    else:
        evaluador = EVALUADORES[nombre_eval]()
        return ejecutar_experimento(
            funcion_objetivo=fn, evaluador=evaluador,
            semilla=semilla, max_iter=max_iter,
        )


def viz_convergencia(args):
    fn = FUNCIONES[args.fn](dim=args.dim)
    historiales = {}
    for nombre in EVALUADORES:
        res = _ejecutar(fn, nombre, args.semilla)
        historiales[nombre] = res["historial_fitness"]
        print(f"{nombre}: fitness={res['fitness_global']:.4e}  t={res['tiempo_total']:.3f}s")

    ruta = f"results/convergencia_{args.fn}_d{args.dim}.png" if args.guardar else None
    plot_convergence(historiales, title=f"Convergencia — {args.fn} d={args.dim}", output_path=ruta)


def viz_speedup(args):
    fn = FUNCIONES[args.fn](dim=args.dim)
    tiempos = {}
    for nombre in EVALUADORES:
        res = _ejecutar(fn, nombre, args.semilla)
        tiempos[nombre] = res["tiempo_total"]
        print(f"{nombre}: {res['tiempo_total']:.3f}s")

    ruta = f"results/speedup_{args.fn}_d{args.dim}.png" if args.guardar else None
    plot_speedup(tiempos, title=f"Speedup — {args.fn} d={args.dim}", output_path=ruta)


def viz_todas_funciones(args):
    resultados = {}
    for nombre_fn, ClaseFn in FUNCIONES.items():
        fn  = ClaseFn(dim=args.dim)
        key = f"{nombre_fn} d={args.dim}"
        resultados[key] = {}
        for nombre_eval in EVALUADORES:
            res = _ejecutar(fn, nombre_eval, args.semilla)
            resultados[key][nombre_eval] = res["historial_fitness"]
            print(f"{key} | {nombre_eval}: fitness={res['fitness_global']:.4e}")

    ruta = f"results/todas_funciones_d{args.dim}.png" if args.guardar else None
    plot_convergence_all_functions(resultados, title=f"Convergencia — d={args.dim}", output_path=ruta)


def viz_boxplot(args):
    resultados = {}
    for nombre_fn, ClaseFn in FUNCIONES.items():
        fn  = ClaseFn(dim=args.dim)
        key = f"{nombre_fn} d={args.dim}"
        resultados[key] = {n: [] for n in EVALUADORES}
        for semilla in SEMILLAS:
            for nombre_eval in EVALUADORES:
                res = _ejecutar(fn, nombre_eval, semilla)
                resultados[key][nombre_eval].append(res["fitness_global"])
                print(f"{key} | {nombre_eval} | s={semilla}: {res['fitness_global']:.4e}")

    ruta = f"results/boxplot_d{args.dim}.png" if args.guardar else None
    plot_boxplot(resultados, title=f"Distribución fitness — d={args.dim}", output_path=ruta)


def viz_portfolio(args):
    """Convergencia y composición de la cartera óptima para V0 y V4."""
    fn = PortfolioSharpe()
    historiales = {}
    metricas_por_eval = {}

    for nombre_eval in ["V0_Sequential", "V4_Numpy"]:
        res = _ejecutar(fn, nombre_eval, args.semilla, max_iter=500)
        historiales[nombre_eval] = [-f for f in res["historial_fitness"]]
        metricas_por_eval[nombre_eval] = fn.metricas(res["pos_global"])
        m = metricas_por_eval[nombre_eval]
        print(f"{nombre_eval}: Sharpe={m['sharpe']:.4f}  "
              f"retorno={m['retorno_anual']:.1%}  vol={m['volatilidad']:.1%}  "
              f"t={res['tiempo_total']:.3f}s")

    ruta = f"results/portfolio_convergencia.png" if args.guardar else None
    plot_portfolio(
        historiales,
        metricas_por_eval,
        referencia_sharpe=fn.metricas(np.ones(fn.dim)/fn.dim)["sharpe"],
        asset_names=fn.asset_names,
        output_path=ruta,
    )


def viz_enjambre(args):
    if args.dim != 2:
        print("La animación solo está disponible para dim=2.")
        return

    from core.swarm import Enjambre
    from core.stopcriteria import Estancamiento

    fn       = FUNCIONES[args.fn](dim=2)
    enjambre = Enjambre(
        funcion_objetivo=fn,
        evaluador=SequentialEvaluator(),
        num_particulas=30,
        semilla=args.semilla,
        criterio_parada=Estancamiento(paciencia=50),
    )
    resultado = enjambre.ejecutar()

    ruta = f"results/enjambre_{args.fn}_d2.gif" if args.guardar else None
    animate_swarm_2d(
        objective_fn=fn,
        position_history=resultado["historial_posiciones"],
        gbest_history=resultado["historial_mejor_global"],
        output_path=ruta,
    )


def viz_superficie(args):
    if args.dim != 2:
        print("La animación 3D solo está disponible para dim=2.")
        return

    from core.swarm import Enjambre
    from core.stopcriteria import Estancamiento

    fn       = FUNCIONES[args.fn](dim=2)
    enjambre = Enjambre(
        funcion_objetivo=fn,
        evaluador=SequentialEvaluator(),
        num_particulas=30,
        semilla=args.semilla,
        criterio_parada=Estancamiento(paciencia=50),
    )
    resultado = enjambre.ejecutar()

    ruta = f"results/superficie_{args.fn}_d2.gif" if args.guardar else None
    animate_swarm_3d(
        objective_fn=fn,
        position_history=resultado["historial_posiciones"],
        gbest_history=resultado["historial_mejor_global"],
        output_path=ruta,
    )


def main():
    args = parse_args()
    dispatch = {
        "convergencia":    viz_convergencia,
        "speedup":         viz_speedup,
        "enjambre":        viz_enjambre,
        "todas_funciones": viz_todas_funciones,
        "boxplot":         viz_boxplot,
        "portfolio":       viz_portfolio,
        "superficie":      viz_superficie,
    }
    dispatch[args.type](args)


if __name__ == "__main__":
    main()