# parallel/sequential.py — V0
import numpy as np
from .base_evaluator import FitnessEvaluator
from objectives.base import ObjectiveFunction


class SequentialEvaluator(FitnessEvaluator):
    """
    V0 — Evaluación secuencial (baseline).

    Evalúa cada partícula una a una en un bucle Python estándar.
    Es el punto de referencia contra el que se miden todas las
    demás estrategias. Sin overhead, sin paralelismo.
    """

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        fitnesses = np.empty(len(positions))
        for i, pos in enumerate(positions):
            fitnesses[i] = objective_fn(pos)
        return fitnesses