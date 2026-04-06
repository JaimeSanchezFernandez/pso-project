# core/particle.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Particle:
    """
    Representa el estado completo de una partícula del enjambre.

    Attributes
    ----------
    position  : posición actual en el espacio de búsqueda, shape (dim,)
    velocity  : velocidad actual, shape (dim,)
    pbest_pos : mejor posición personal encontrada hasta ahora, shape (dim,)
    pbest_fit : fitness de la mejor posición personal
    """

    position: np.ndarray
    velocity: np.ndarray
    pbest_pos: np.ndarray
    pbest_fit: float = float("inf")

    def update_personal_best(self, fitness: float) -> None:
        """Actualiza pbest si el fitness actual es mejor (menor)."""
        if fitness < self.pbest_fit:
            self.pbest_fit = fitness
            self.pbest_pos = self.position.copy()

    def __repr__(self) -> str:
        return (
            f"Particle(pbest_fit={self.pbest_fit:.6f}, "
            f"pos={self.position})"
        )