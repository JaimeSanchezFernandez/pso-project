# tests/test_objectives.py
import numpy as np
import pytest
from objectives.functions import Sphere, Rosenbrock, Rastrigin, Ackley


def test_sphere_minimum():
    """Sphere debe devolver 0 en el origen."""
    fn = Sphere(dim=3)
    assert fn(np.zeros(3)) == pytest.approx(0.0)


def test_sphere_positive():
    """Sphere siempre debe devolver valores no negativos."""
    fn = Sphere(dim=5)
    x = np.random.default_rng(42).uniform(-5, 5, 5)
    assert fn(x) >= 0.0


def test_rosenbrock_minimum():
    """Rosenbrock debe devolver 0 en el vector de unos."""
    fn = Rosenbrock(dim=4)
    assert fn(np.ones(4)) == pytest.approx(0.0, abs=1e-10)


def test_rastrigin_minimum():
    """Rastrigin debe devolver 0 en el origen."""
    fn = Rastrigin(dim=5)
    assert fn(np.zeros(5)) == pytest.approx(0.0)


def test_ackley_minimum():
    """Ackley debe devolver 0 en el origen."""
    fn = Ackley(dim=3)
    assert fn(np.zeros(3)) == pytest.approx(0.0, abs=1e-10)


def test_dimensions_match():
    """Todas las funciones deben respetar la dimensión declarada."""
    for FnClass in [Sphere, Rastrigin, Ackley]:
        for dim in [2, 10, 30]:
            fn = FnClass(dim=dim)
            assert fn.dim == dim
            assert len(fn.bounds) == dim


def test_bounds_defined():
    """Todas las funciones deben tener límites bien definidos."""
    for FnClass in [Sphere, Rosenbrock, Rastrigin, Ackley]:
        fn = FnClass(dim=5)
        assert len(fn.lower_bounds) == 5
        assert len(fn.upper_bounds) == 5
        assert np.all(fn.lower_bounds < fn.upper_bounds)