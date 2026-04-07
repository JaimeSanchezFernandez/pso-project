# experiments/grid_search.py
import itertools
import logging
from experiments.runner import ejecutar_experimento
from objectives.functions import ObjectiveFunction
from parallel.sequential import FitnessEvaluator

logger = logging.getLogger(__name__)


def busqueda_grid(
    funcion_objetivo: ObjectiveFunction,
    evaluador: FitnessEvaluator,
    param_grid: dict[str, list],
    semillas: list[int] | None = None,
) -> list[dict]:
    """
    Ejecuta un grid search sobre los hiperparámetros del PSO.

    Parameters
    ----------
    funcion_objetivo : función objetivo a optimizar
    evaluador        : estrategia de evaluación (V0, V1, V2)
    param_grid       : diccionario con listas de valores por parámetro.
                       Ejemplo:
                       {
                           "w":            [0.4, 0.7, 0.9],
                           "c1":           [1.0, 1.5, 2.0],
                           "c2":           [1.0, 1.5, 2.0],
                           "num_particulas": [20, 30]
                       }
    semillas         : lista de semillas para repetir cada combinación.

    Returns
    -------
    Lista de dicts, uno por cada combinación (parámetros + métricas).
    """
    if semillas is None:
        semillas = [42]

    claves = list(param_grid.keys())
    valores = list(param_grid.values())
    combinaciones = list(itertools.product(*valores))

    total = len(combinaciones) * len(semillas)
    logger.info(
        f"Grid search | combinaciones={len(combinaciones)} | "
        f"semillas={len(semillas)} | total={total}"
    )

    resultados = []
    num_run = 0

    for combo in combinaciones:
        params = dict(zip(claves, combo))

        for semilla in semillas:
            num_run += 1
            logger.info(f"Run {num_run}/{total} | params={params} | semilla={semilla}")

            resultado = ejecutar_experimento(
                funcion_objetivo=funcion_objetivo,
                evaluador=evaluador,
                w=params.get("w", 0.7),
                c1=params.get("c1", 1.5),
                c2=params.get("c2", 1.5),
                num_particulas=params.get("num_particulas", 30),
                semilla=semilla,
            )

            resultado.update(params)
            resultado["semilla"] = semilla
            resultados.append(resultado)

    return resultados