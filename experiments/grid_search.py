# experiments/grid_search.py
import itertools
import logging
from experiments.runner import run_experiment
from objectives.base import ObjectiveFunction
from parallel.base_evaluator import FitnessEvaluator

logger = logging.getLogger(__name__)


def run_grid_search(
    objective_fn: ObjectiveFunction,
    evaluator: FitnessEvaluator,
    param_grid: dict,
    seeds: list[int] = None,
) -> list[dict]:
    """
    Ejecuta un grid search sobre los hiperparámetros del PSO.

    Parameters
    ----------
    objective_fn : función objetivo a optimizar
    evaluator    : estrategia de evaluación (V0, V1, V2)
    param_grid   : diccionario con listas de valores por parámetro.
                   Claves soportadas: w, c1, c2, n_particles
                   Ejemplo:
                   {
                       "w":           [0.4, 0.7, 0.9],
                       "c1":          [1.0, 1.5, 2.0],
                       "c2":          [1.0, 1.5, 2.0],
                       "n_particles": [20, 30]
                   }
    seeds        : lista de seeds para repetir cada combinación.
                   Permite calcular media y desviación típica.

    Returns
    -------
    Lista de dicts, uno por cada combinación (parámetros + métricas).
    """
    if seeds is None:
        seeds = [42]

    # Genera todas las combinaciones posibles
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    total = len(combinations) * len(seeds)
    logger.info(f"Grid search | combinaciones={len(combinations)} | seeds={len(seeds)} | total runs={total}")

    results = []
    run_idx = 0

    for combo in combinations:
        params = dict(zip(keys, combo))

        for seed in seeds:
            run_idx += 1
            logger.info(f"Run {run_idx}/{total} | params={params} | seed={seed}")

            result = run_experiment(
                objective_fn=objective_fn,
                evaluator=evaluator,
                w=params.get("w", 0.7),
                c1=params.get("c1", 1.5),
                c2=params.get("c2", 1.5),
                n_particles=params.get("n_particles", 30),
                seed=seed,
            )

            result.update(params)
            result["seed"] = seed
            results.append(result)

    return results