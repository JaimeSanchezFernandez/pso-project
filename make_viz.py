# make_viz.py
"""
Genera visualizaciones a partir de los resultados guardados.

Uso:
    python make_viz.py --type convergence
    python make_viz.py --type speedup
    python make_viz.py --type swarm --fn sphere --dim 2
"""
import argparse
import logging
import numpy as np
from objectives.sphere import Sphere
from objectives.rosenbrock import Rosenbrock
from objectives.rastrigin import Rastrigin
from objectives.ackley import Ackley
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator
from experiments.runner import run_experiment
from viz.convergence import plot_convergence, plot_speedup
from viz.swarm_plot import animate_swarm_2d

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
    "V0_Sequential": SequentialEvaluator,
    "V1_Threading":  ThreadingEvaluator,
    "V2_Process":    ProcessEvaluator,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar visualizaciones PSO")
    parser.add_argument(
        "--type", type=str, default="convergence",
        choices=["convergence", "speedup", "swarm"],
        help="Tipo de visualización"
    )
    parser.add_argument("--fn",   type=str, default="sphere", choices=FUNCTIONS.keys())
    parser.add_argument("--dim",  type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="Guardar figura en results/")
    return parser.parse_args()


def viz_convergence(args: argparse.Namespace) -> None:
    """Compara curvas de convergencia de V0, V1 y V2."""
    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    histories: dict[str, list[float]] = {}
    times: dict[str, float] = {}

    for name, EvalClass in EVALUATORS.items():
        evaluator = EvalClass()
        result = run_experiment(
            objective_fn=objective_fn,
            evaluator=evaluator,
            seed=args.seed,
        )
        histories[name] = result["fitness_history"]
        times[name] = result["elapsed_seconds"]
        print(
            f"{name}: gbest={result['gbest_fit']:.4e} | "
            f"t_total={result['elapsed_seconds']:.3f}s | "
            f"t_eval={result['time_eval']:.3f}s | "
            f"t_update={result['time_update']:.3f}s | "
            f"overhead={result['overhead']:.3f}s"
        )

    output_path = f"results/convergence_{args.fn}_d{args.dim}.png" if args.save else None
    plot_convergence(
        histories,
        title=f"Convergencia — {args.fn} d={args.dim}",
        output_path=output_path,
    )


def viz_speedup(args: argparse.Namespace) -> None:
    """Compara speedup de V0, V1 y V2."""
    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    times: dict[str, float] = {}

    for name, EvalClass in EVALUATORS.items():
        evaluator = EvalClass()
        result = run_experiment(
            objective_fn=objective_fn,
            evaluator=evaluator,
            seed=args.seed,
        )
        times[name] = result["elapsed_seconds"]
        print(f"{name}: {result['elapsed_seconds']:.3f}s")

    output_path = f"results/speedup_{args.fn}_d{args.dim}.png" if args.save else None
    plot_speedup(
        times,
        title=f"Speedup — {args.fn} d={args.dim}",
        output_path=output_path,
    )


def viz_swarm(args: argparse.Namespace) -> None:
    """Genera animación 2D del enjambre (solo para dim=2)."""
    if args.dim != 2:
        print("La animación del enjambre solo está disponible para dim=2.")
        return

    from core.swarm import Swarm
    from core.stopcriteria import Stagnation

    objective_fn = FUNCTIONS[args.fn](dim=2)
    evaluator = SequentialEvaluator()

    swarm = Swarm(
        objective_fn=objective_fn,
        evaluator=evaluator,
        n_particles=30,
        seed=args.seed,
        stop_criterion=Stagnation(patience=50),
    )

    result = swarm.run()

    output_path = f"results/swarm_{args.fn}_d2.gif" if args.save else None
    animate_swarm_2d(
        objective_fn=objective_fn,
        position_history=result["position_history"],
        gbest_history=result["gbest_history"],
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()

    if args.type == "convergence":
        viz_convergence(args)
    elif args.type == "speedup":
        viz_speedup(args)
    elif args.type == "swarm":
        viz_swarm(args)


if __name__ == "__main__":
    main()