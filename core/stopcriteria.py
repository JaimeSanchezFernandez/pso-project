# core/stopcriteria.py
from abc import ABC, abstractmethod
import numpy as np


class CriterioParada(ABC):
    """Clase base para los criterios de parada."""

    @abstractmethod
    def debe_parar(self, iteracion: int, mejor_fitness: float, historial: list[float]) -> bool:
        ...

    @abstractmethod
    def reiniciar(self) -> None:
        """Resetea el estado interno entre ejecuciones."""
        ...


class MaxIteraciones(CriterioParada):
    """Para cuando se alcanza el número máximo de iteraciones."""

    def __init__(self, max_iter: int) -> None:
        self.max_iter = max_iter

    def debe_parar(self, iteracion: int, mejor_fitness: float, historial: list[float]) -> bool:
        return iteracion >= self.max_iter

    def reiniciar(self) -> None:
        pass


class Tolerancia(CriterioParada):
    """Para cuando el mejor fitness cae por debajo de un umbral."""

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def debe_parar(self, iteracion: int, mejor_fitness: float, historial: list[float]) -> bool:
        return mejor_fitness < self.tol

    def reiniciar(self) -> None:
        pass


class Estancamiento(CriterioParada):
    """
    Para cuando el mejor fitness no mejora más de `tol`
    durante `paciencia` iteraciones consecutivas.
    """

    def __init__(self, paciencia: int = 50, tol: float = 1e-8) -> None:
        self.paciencia = paciencia
        self.tol = tol
        self._contador = 0
        self._ultimo_mejor = float("inf")

    def debe_parar(self, iteracion: int, mejor_fitness: float, historial: list[float]) -> bool:
        if abs(self._ultimo_mejor - mejor_fitness) < self.tol:
            self._contador += 1
        else:
            self._contador = 0
            self._ultimo_mejor = mejor_fitness
        return self._contador >= self.paciencia

    def reiniciar(self) -> None:
        self._contador = 0
        self._ultimo_mejor = float("inf")