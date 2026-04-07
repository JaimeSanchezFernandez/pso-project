# core/topology.py
from abc import ABC, abstractmethod
import numpy as np
from .particle import Particle


class Topologia(ABC):
    """Clase base para las topologías del enjambre."""

    @abstractmethod
    def obtener_mejor_posicion(self, particulas: list[Particle]) -> np.ndarray:
        """
        Devuelve la posición del mejor vecino para guiar la actualización.

        Parameters
        ----------
        particulas : lista completa de partículas del enjambre

        Returns
        -------
        np.ndarray : posición del mejor conocido según la topología
        """
        ...


class MejorGlobal(Topologia):
    """
    Topología global (gbest): todas las partículas comparten
    el mismo mejor global. Converge rápido pero es más propensa
    a quedar atrapada en mínimos locales.
    """

    def obtener_mejor_posicion(self, particulas: list[Particle]) -> np.ndarray:
        mejor = min(particulas, key=lambda p: p.mejor_fitness)
        return mejor.mejor_pos.copy()