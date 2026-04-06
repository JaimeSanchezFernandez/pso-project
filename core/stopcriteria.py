# core/stopcriteria.py
from abc import ABC, abstractmethod
import numpy as np


class StopCriterion(ABC):
    """Contrato base para los criterios de parada."""

    @abstractmethod
    def should_stop(self, iteration: int, best_fit: float, fitness_history: list[float]) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resetea el estado interno (necesario entre ejecuciones)."""
        ...


class MaxIterations(StopCriterion):
    """Para cuando se alcanza el número máximo de iteraciones."""

    def __init__(self, max_iter: int):
        self.max_iter = max_iter

    def should_stop(self, iteration: int, best_fit: float, fitness_history: list[float]) -> bool:
        return iteration >= self.max_iter

    def reset(self) -> None:
        pass


class Tolerance(StopCriterion):
    """Para cuando el mejor fitness cae por debajo de un umbral."""

    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def should_stop(self, iteration: int, best_fit: float, fitness_history: list[float]) -> bool:
        return best_fit < self.tol

    def reset(self) -> None:
        pass


class Stagnation(StopCriterion):
    """
    Para cuando el mejor fitness no mejora más de `tol`
    durante `patience` iteraciones consecutivas.
    """

    def __init__(self, patience: int = 50, tol: float = 1e-8):
        self.patience = patience
        self.tol = tol
        self._counter = 0
        self._last_best = float("inf")

    def should_stop(self, iteration: int, best_fit: float, fitness_history: list[float]) -> bool:
        if abs(self._last_best - best_fit) < self.tol:
            self._counter += 1
        else:
            self._counter = 0
            self._last_best = best_fit
        return self._counter >= self.patience

    def reset(self) -> None:
        self._counter = 0
        self._last_best = float("inf")