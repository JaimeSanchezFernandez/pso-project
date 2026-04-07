# parallel/sequential.py
import numpy as np
from objectives.functions import ObjectiveFunction


class FitnessEvaluator:
    """
    Clase base para todas las estrategias de evaluación de fitness.
    Recibe positions (n_particles, dim) y devuelve fitness (n_particles,).
    """

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SequentialEvaluator(FitnessEvaluator):
    """
    V0 — Evaluación secuencial (baseline).
    Evalúa cada partícula una a una en un bucle Python estándar.
    Sin overhead, sin paralelismo. Punto de referencia contra el
    que se miden todas las demás estrategias.
    """

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        fitnesses = np.empty(len(positions))
        for i, pos in enumerate(positions):
            fitnesses[i] = objective_fn(pos)
        return fitnesses