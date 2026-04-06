# objectives/rosenbrock.py
import numpy as np
from .base import ObjectiveFunction


class Rosenbrock(ObjectiveFunction):
    """
    Función Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    - Mínimo global: f(1, 1, ..., 1) = 0
    - No separable, unimodal pero con un valle muy estrecho y curvo.
    - Es una de las funciones más difíciles para PSO porque el valle
      que lleva al mínimo es muy difícil de seguir.
    """

    def __init__(self, dim: int):
        bounds = [(-2.048, 2.048)] * dim
        super().__init__(dim, bounds)

    def __call__(self, x: np.ndarray) -> float:
        xi = x[:-1]
        xi1 = x[1:]
        return float(np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1.0 - xi) ** 2))