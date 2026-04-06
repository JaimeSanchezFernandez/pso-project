# objectives/rastrigin.py
import numpy as np
from .base import ObjectiveFunction


class Rastrigin(ObjectiveFunction):
    """
    Función Rastrigin: f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))

    - Mínimo global: f(0, 0, ..., 0) = 0
    - Separable, altamente multimodal (muchos mínimos locales).
    - Es una de las funciones más exigentes para PSO porque el enjambre
      puede quedar atrapado fácilmente en mínimos locales.
    """

    def __init__(self, dim: int):
        bounds = [(-5.12, 5.12)] * dim
        super().__init__(dim, bounds)

    def __call__(self, x: np.ndarray) -> float:
        d = self.dim
        return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))