# experiments/runner.py
import time
import logging
from core.swarm import Swarm
from core.topology import GlobalBest
from core.stopcriteria import MaxIterations, Stagnation
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

    Criterio de parada: combinación de MaxIterations + Stagnation.
    MaxIterations garantiza que max_iter se respeta siempre.
    Stagnation para antes si el algoritmo se ha estancado.

    Returns
    -------
    dict con configuración, métricas y tiempos desglosados
    """
    from core.stopcriteria import MaxIterations, Stagnation

    class CombinedStop:
        """Para cuando se cumple cualquiera de los dos criterios."""
        def __init__(self) -> None:
            self.max_iter = MaxIterations(max_iter)
            self.stagnation = Stagnation(patience=50, tol=1e-8)

        def should_stop(self, iteration: int, best_fit: float, fitness_history: list[float]) -> bool:
            return (
                self.max_iter.should_stop(iteration, best_fit, fitness_history) or
                self.stagnation.should_stop(iteration, best_fit, fitness_history)
            )

        def reset(self) -> None:
            self.max_iter.reset()
            self.stagnation.reset()

    swarm = Swarm(
        objective_fn=objective_fn,
        evaluator=evaluator,
        n_particles=n_particles,
        w=w,
        c1=c1,
        c2=c2,
        topology=GlobalBest(),
        stop_criterion=CombinedStop(),
        seed=seed,
    )

    logger.info(
        f"Experimento | fn={objective_fn} | evaluator={evaluator} | "
        f"n_particles={n_particles} | w={w} | c1={c1} | c2={c2} | "
        f"max_iter={max_iter} | seed={seed}"
    )

    t_start = time.perf_counter()
    try:
        result = swarm.run()
    finally:
        if hasattr(evaluator, "shutdown"):
            evaluator.shutdown()
    t_end = time.perf_counter()

    elapsed = t_end - t_start

    logger.info(
        f"Resultado | gbest_fit={result['gbest_fit']:.6e} | "
        f"iter={result['n_iterations']} | elapsed={elapsed:.4f}s"
    )

    return {
        "objective_fn":     repr(objective_fn),
        "evaluator":        repr(evaluator),
        "n_particles":      n_particles,
        "w":                w,
        "c1":               c1,
        "c2":               c2,
        "max_iter":         max_iter,
        "seed":             seed,
        "gbest_fit":        result["gbest_fit"],
        "gbest_pos":        result["gbest_pos"].tolist(),
        "fitness_history":  result["fitness_history"],
        "n_iterations":     result["n_iterations"],
        "elapsed_seconds":  elapsed,
        "time_eval":        result["time_eval"],
        "time_update":      result["time_update"],
        "time_total":       result["time_total"],
        "overhead":         result["overhead"],
    }