# core/topology.py
from abc import ABC, abstractmethod
import numpy as np
from .particle import Particle


class Topology(ABC):
    """Contrato base para las topologías del enjambre."""

    @abstractmethod
    def get_best_position(self, particles: list[Particle]) -> np.ndarray:
        """
        Devuelve la posición del mejor vecino para guiar la actualización.

        Parameters
        ----------
        particles : lista completa de partículas del enjambre

        Returns
        -------
        np.ndarray : posición del mejor conocido según la topología
        """
        ...


class GlobalBest(Topology):
    """
    Topología global (gbest): todas las partículas comparten
    el mismo mejor global. Es la topología más simple y converge
    rápido, pero es más propensa a quedar atrapada en mínimos locales.
    """

    def get_best_position(self, particles: list[Particle]) -> np.ndarray:
        best = min(particles, key=lambda p: p.pbest_fit)
        return best.pbest_pos.copy()