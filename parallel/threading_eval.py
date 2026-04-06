# parallel/threading_eval.py — V1
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .base_evaluator import FitnessEvaluator
from objectives.base import ObjectiveFunction


class ThreadingEvaluator(FitnessEvaluator):
    """
    V1 — Evaluación paralela con hilos (ThreadPoolExecutor).

    Cada partícula se evalúa en un hilo separado.

    Nota sobre el GIL:
    Python tiene el GIL (Global Interpreter Lock), que impide que
    dos hilos ejecuten bytecode Python puro al mismo tiempo. Por tanto,
    para funciones objetivo implementadas en Python puro, esta estrategia
    NO mejora el rendimiento respecto a V0 — el GIL serializa la ejecución.

    Sin embargo, NumPy libera el GIL durante sus operaciones internas
    (están implementadas en C). Si la función objetivo hace uso intensivo
    de NumPy (como Rastrigin o Ackley), los hilos pueden solaparse
    parcialmente y sí puede haber ganancia real.

    En cualquier caso, el overhead de crear y gestionar hilos puede
    superar el beneficio para enjambres pequeños.

    Parameters
    ----------
    max_workers : número máximo de hilos. None = usa os.cpu_count()
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitnesses = list(executor.map(objective_fn, positions))
        return np.array(fitnesses)