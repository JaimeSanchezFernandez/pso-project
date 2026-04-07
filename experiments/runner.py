# experiments/runner.py
import time
import logging
from core.swarm import Enjambre
from core.topology import MejorGlobal
from core.stopcriteria import MaxIteraciones, Estancamiento
from objectives.functions import ObjectiveFunction
from parallel.sequential import FitnessEvaluator

logger = logging.getLogger(__name__)


def ejecutar_experimento(
    funcion_objetivo: ObjectiveFunction,
    evaluador: FitnessEvaluator,
    num_particulas: int = 30,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    max_iter: int = 200,
    tol: float = 1e-6,
    semilla: int = 42,
) -> dict:
    """Ejecuta una corrida del PSO y devuelve los resultados."""

    # Usamos MaxIteraciones como criterio de parada principal
    # y Estancamiento para parar antes si no hay mejora
    criterio = MaxIteraciones(max_iter)

    enjambre = Enjambre(
        funcion_objetivo=funcion_objetivo,
        evaluador=evaluador,
        num_particulas=num_particulas,
        w=w,
        c1=c1,
        c2=c2,
        topologia=MejorGlobal(),
        criterio_parada=criterio,
        semilla=semilla,
    )

    logger.info(
        f"Experimento | fn={funcion_objetivo} | evaluador={evaluador} | "
        f"num_particulas={num_particulas} | w={w} | c1={c1} | c2={c2} | "
        f"max_iter={max_iter} | semilla={semilla}"
    )

    t_inicio = time.perf_counter()
    resultado = enjambre.ejecutar()
    t_fin = time.perf_counter()

    tiempo_total = t_fin - t_inicio

    logger.info(
        f"Resultado | fitness_global={resultado['fitness_global']:.6e} | "
        f"iter={resultado['num_iteraciones']} | elapsed={tiempo_total:.4f}s"
    )

    return {
        "funcion_objetivo":     repr(funcion_objetivo),
        "evaluador":            repr(evaluador),
        "num_particulas":       num_particulas,
        "w":                    w,
        "c1":                   c1,
        "c2":                   c2,
        "max_iter":             max_iter,
        "semilla":              semilla,
        "fitness_global":       resultado["fitness_global"],
        "pos_global":           resultado["pos_global"].tolist(),
        "historial_fitness":    resultado["historial_fitness"],
        "num_iteraciones":      resultado["num_iteraciones"],
        "tiempo_total":         tiempo_total,
        "tiempo_evaluacion":    resultado["tiempo_evaluacion"],
        "tiempo_actualizacion": resultado["tiempo_actualizacion"],
        "overhead":             resultado["overhead"],
    }