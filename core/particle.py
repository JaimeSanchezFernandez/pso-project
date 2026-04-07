# core/particle.py
from dataclasses import dataclass
import numpy as np


@dataclass
class Particle:
    """
    Representa el estado completo de una partícula del enjambre.

    Attributes
    ----------
    posicion      : posición actual en el espacio de búsqueda, shape (dim,)
    velocidad     : velocidad actual, shape (dim,)
    mejor_pos     : mejor posición personal encontrada hasta ahora, shape (dim,)
    mejor_fitness : fitness de la mejor posición personal
    """

    posicion: np.ndarray
    velocidad: np.ndarray
    mejor_pos: np.ndarray
    mejor_fitness: float = float("inf")

    def actualizar_mejor(self, fitness: float) -> None:
        """Actualiza mejor_pos si el fitness actual es mejor (menor)."""
        if fitness < self.mejor_fitness:
            self.mejor_fitness = fitness
            self.mejor_pos = self.posicion.copy()

    def __repr__(self) -> str:
        return (
            f"Particle(mejor_fitness={self.mejor_fitness:.6f}, "
            f"posicion={self.posicion})"
        )