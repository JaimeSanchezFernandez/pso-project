# objectives/ackley.py
import numpy as np
from .base import ObjectiveFunction


class Ackley(ObjectiveFunction):
    """
    Función Ackley: 
    f(x) = -a*exp(-b*sqrt(1/d * sum(x_i^2))) 
           - exp(1/d * sum(cos(c*x_i))) + a + exp(1)

    - Mínimo global: f(0, 0, ..., 0) = 0
    - No separable, multimodal con un mínimo global rodeado de 
      muchos mínimos locales poco profundos.
    - Los parámetros estándar son a=20, b=0.2, c=2*pi.
    """

    def __init__(self, dim: int, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        bounds = [(-32.768, 32.768)] * dim
        super().__init__(dim, bounds)
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: np.ndarray) -> float:
        d = self.dim
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return float(term1 + term2 + self.a + np.e)