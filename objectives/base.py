# objectives/base.py
from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    """
    Contrato base para todas las funciones objetivo.

    Toda función benchmark hereda de esta clase e implementa __call__.
    La interfaz es intencionadamente simple: recibe un array de posiciones
    y devuelve un array de fitness values.
    """

    def __init__(self, dim: int, bounds: list[tuple[float, float]]):
        """
        Parameters
        ----------
        dim    : dimensionalidad del espacio de búsqueda
        bounds : lista de (min, max) por dimensión
        """
        assert len(bounds) == dim, "bounds debe tener una entrada por dimensión"
        self.dim = dim
        self.bounds = bounds  # lista de (min, max)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        Evalúa la función en el punto x.

        Parameters
        ----------
        x : array de shape (dim,)  — posición de UNA partícula

        Returns
        -------
        float : valor de fitness (a minimizar)
        """
        ...

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds])

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"