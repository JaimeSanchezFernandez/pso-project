# objectives/functions.py
import numpy as np


class ObjectiveFunction:
    """
    Clase base para todas las funciones objetivo.
    Recibe un array de shape (dim,) y devuelve un float.
    """

    def __init__(self, dim: int, bounds: list[tuple[float, float]]) -> None:
        assert len(bounds) == dim, "bounds debe tener una entrada por dimensión"
        self.dim = dim
        self.bounds = bounds

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds])

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


class Sphere(ObjectiveFunction):
    """
    f(x) = sum(x_i^2)
    Mínimo global: f(0,...,0) = 0
    Unimodal, separable. Usada para verificar correctitud del PSO.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim, [(-5.12, 5.12)] * dim)

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))


class Rosenbrock(ObjectiveFunction):
    """
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Mínimo global: f(1,...,1) = 0
    Valle estrecho y curvo, difícil de seguir para el enjambre.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim, [(-2.048, 2.048)] * dim)

    def __call__(self, x: np.ndarray) -> float:
        xi = x[:-1]
        xi1 = x[1:]
        return float(np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1.0 - xi) ** 2))


class Rastrigin(ObjectiveFunction):
    """
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    Mínimo global: f(0,...,0) = 0
    Multimodal con muchos mínimos locales. Difícil para PSO.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim, [(-5.12, 5.12)] * dim)

    def __call__(self, x: np.ndarray) -> float:
        return float(10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


class Ackley(ObjectiveFunction):
    """
    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2)))
           - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    Mínimo global: f(0,...,0) = 0
    Multimodal con mínimo global rodeado de locales poco profundos.
    """

    def __init__(self, dim: int, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> None:
        super().__init__(dim, [(-32.768, 32.768)] * dim)
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