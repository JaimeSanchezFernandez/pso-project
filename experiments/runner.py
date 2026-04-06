# experiments/runner.py
import time
import logging
from core.swarm import Swarm
from core.topology import GlobalBest
from core.stopcriteria import MaxIterations, Tolerance, Stagnation
from objectives.base import ObjectiveFunction
from parallel.base_evaluator import FitnessEvaluator

logger = logging.getLogger(__name__)


def run_experiment(
    objective_fn: ObjectiveFunction,
    evaluator: FitnessEvaluator,
    n_particles: int = 30,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """
    Ejecuta una única corrida de PSO y devuelve los resultados.

    Returns
    -------
    dict con configuración, métricas y tiempos
    """
    stop = Stagnation(patience=50, tol=1e-8)

    swarm = Swarm(
        objective_fn=objective_fn,
        evaluator=evaluator,
        n_particles=n_particles,
        w=w,
        c1=c1,
        c2=c2,
        topology=GlobalBest(),
        stop_criterion=stop,
        seed=seed,
    )

    logger.info(
        f"Experimento | fn={objective_fn} | evaluator={evaluator} | "
        f"n_particles={n_particles} | w={w} | c1={c1} | c2={c2} | seed={seed}"
    )

    t_start = time.perf_counter()
    result = swarm.run()
    t_end = time.perf_counter()

    elapsed = t_end - t_start

    return {
        "objective_fn": repr(objective_fn),
        "evaluator": repr(evaluator),
        "n_particles": n_particles,
        "w": w,
        "c1": c1,
        "c2": c2,
        "max_iter": max_iter,
        "seed": seed,
        "gbest_fit": result["gbest_fit"],
        "gbest_pos": result["gbest_pos"].tolist(),
        "fitness_history": result["fitness_history"],
        "n_iterations": result["n_iterations"],
        "elapsed_seconds": elapsed,
    }