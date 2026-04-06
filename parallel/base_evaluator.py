# parallel/base_evaluator.py
from abc import ABC, abstractmethod
import numpy as np
from objectives.base import ObjectiveFunction


class FitnessEvaluator(ABC):
    """
    Contrato base para todas las estrategias de evaluación de fitness.

    El Swarm llama siempre a evaluate() sin saber si por debajo
    hay un bucle secuencial, hilos o procesos. Eso es todo el desacoplamiento.

    Parameters
    ----------
    positions    : array de shape (n_particles, dim)
    objective_fn : función objetivo a evaluar

    Returns
    -------
    np.ndarray de shape (n_particles,) con el fitness de cada partícula
    """

    @abstractmethod
    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"