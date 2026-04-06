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


def parse_args():
    parser = argparse.ArgumentParser(description="Generar visualizaciones PSO")
    parser.add_argument(
        "--type", type=str, default="convergence",
        choices=["convergence", "speedup", "swarm"],
        help="Tipo de visualización"
    )
    parser.add_argument("--fn",    type=str, default="sphere", choices=FUNCTIONS.keys())
    parser.add_argument("--dim",   type=int, default=10)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--save",  action="store_true", help="Guardar figura en results/")
    return parser.parse_args()


def viz_convergence(args):
    """Compara curvas de convergencia de V0, V1 y V2."""
    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    histories = {}
    times = {}

    for name, EvalClass in EVALUATORS.items():
        evaluator = EvalClass()
        result = run_experiment(
            objective_fn=objective_fn,
            evaluator=evaluator,
            seed=args.seed,
        )
        histories[name] = result["fitness_history"]
        times[name] = result["elapsed_seconds"]
        print(f"{name}: gbest={result['gbest_fit']:.4e} | {result['elapsed_seconds']:.3f}s")

    output_path = f"results/convergence_{args.fn}_d{args.dim}.png" if args.save else None
    plot_convergence(
        histories,
        title=f"Convergencia — {args.fn} d={args.dim}",
        output_path=output_path,
    )


def viz_speedup(args):
    """Compara speedup de V0, V1 y V2."""
    objective_fn = FUNCTIONS[args.fn](dim=args.dim)
    times = {}

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


def viz_swarm(args):
    """Genera animación 2D del enjambre (solo para dim=2)."""
    if args.dim != 2:
        print("La animación del enjambre solo está disponible para dim=2.")
        return

    objective_fn = FUNCTIONS[args.fn](dim=2)

    # Ejecutar PSO recogiendo historial de posiciones
    from core.swarm import Swarm
    from core.stopcriteria import Stagnation

    evaluator = SequentialEvaluator()
    swarm = Swarm(
        objective_fn=objective_fn,
        evaluator=evaluator,
        n_particles=30,
        seed=args.seed,
        stop_criterion=Stagnation(patience=50),
    )

    # Monkey-patch para recoger posiciones por iteración
    position_history = []
    gbest_history = []
    original_run = swarm.run

    def run_with_tracking():
        from core.stopcriteria import Stagnation
        swarm.stop_criterion.reset()
        swarm._initialize()

        positions = np.array([p.position for p in swarm.particles])
        fitnesses = evaluator.evaluate(positions, objective_fn)
        for particle, fit in zip(swarm.particles, fitnesses):
            particle.update_personal_best(fit)

        swarm.gbest_pos = swarm.topology.get_best_position(swarm.particles)
        swarm.gbest_fit = min(p.pbest_fit for p in swarm.particles)
        swarm.fitness_history = [swarm.gbest_fit]

        iteration = 0
        while not swarm.stop_criterion.should_stop(iteration, swarm.gbest_fit, swarm.fitness_history):
            position_history.append(np.array([p.position.copy() for p in swarm.particles]))
            gbest_history.append(swarm.gbest_pos.copy())

            for particle in swarm.particles:
                swarm._update_velocity(particle, swarm.gbest_pos)
                swarm._update_position(particle)

            positions = np.array([p.position for p in swarm.particles])
            fitnesses = evaluator.evaluate(positions, objective_fn)
            for particle, fit in zip(swarm.particles, fitnesses):
                particle.update_personal_best(fit)

            swarm.gbest_pos = swarm.topology.get_best_position(swarm.particles)
            swarm.gbest_fit = min(p.pbest_fit for p in swarm.particles)
            swarm.fitness_history.append(swarm.gbest_fit)
            iteration += 1

        return {
            "gbest_fit": swarm.gbest_fit,
            "gbest_pos": swarm.gbest_pos,
            "fitness_history": swarm.fitness_history,
            "n_iterations": iteration,
            "seed": swarm.seed,
        }

    run_with_tracking()

    output_path = f"results/swarm_{args.fn}_d2.gif" if args.save else None
    animate_swarm_2d(
        objective_fn=objective_fn,
        position_history=position_history,
        gbest_history=gbest_history,
        output_path=output_path,
    )


def main():
    args = parse_args()

    if args.type == "convergence":
        viz_convergence(args)
    elif args.type == "speedup":
        viz_speedup(args)
    elif args.type == "swarm":
        viz_swarm(args)


if __name__ == "__main__":
    main()