# objectives/sphere.py
import numpy as np
from .base import ObjectiveFunction


class Sphere(ObjectiveFunction):
    """
    Función Sphere: f(x) = sum(x_i^2)

    - Mínimo global: f(0, 0, ..., 0) = 0
    - Separable, unimodal, convexa.
    - Usada principalmente para verificar correctitud del PSO
      ya que es la más simple posible.
    """

    def __init__(self, dim: int):
        bounds = [(-5.12, 5.12)] * dim
        super().__init__(dim, bounds)

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))