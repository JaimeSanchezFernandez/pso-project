# tests/test_core.py
import numpy as np
import pytest
from core.swarm import Swarm
from core.stopcriteria import MaxIterations
from objectives.sphere import Sphere
from parallel.sequential import SequentialEvaluator


def make_swarm(seed=42, max_iter=100):
    return Swarm(
        objective_fn=Sphere(dim=2),
        evaluator=SequentialEvaluator(),
        n_particles=20,
        w=0.7, c1=1.5, c2=1.5,
        stop_criterion=MaxIterations(max_iter),
        seed=seed,
    )


def test_reproducibility():
    """Dos ejecuciones con la misma seed deben dar exactamente el mismo resultado."""
    result1 = make_swarm(seed=42).run()
    result2 = make_swarm(seed=42).run()
    assert result1["gbest_fit"] == result2["gbest_fit"]
    assert np.allclose(result1["gbest_pos"], result2["gbest_pos"])


def test_different_seeds_differ():
    """Seeds distintas deben dar resultados distintos."""
    result1 = make_swarm(seed=42).run()
    result2 = make_swarm(seed=99).run()
    assert result1["gbest_fit"] != result2["gbest_fit"]


def test_monotonic_gbest():
    """El mejor global nunca debe empeorar entre iteraciones."""
    result = make_swarm(seed=42, max_iter=200).run()
    history = result["fitness_history"]
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-12, (
            f"gbest empeoró en iteración {i}: {history[i-1]} → {history[i]}"
        )


def test_sphere_converges():
    """PSO debe converger cerca de 0 en Sphere(d=2) con parámetros razonables."""
    result = make_swarm(seed=42, max_iter=500).run()
    assert result["gbest_fit"] < 1e-3, (
        f"Sphere no convergió suficiente: gbest_fit={result['gbest_fit']}"
    )


def test_bounds_respected():
    """Las posiciones finales deben estar dentro de los límites."""
    swarm = make_swarm(seed=42, max_iter=100)
    swarm._initialize()
    lb = swarm.objective_fn.lower_bounds
    ub = swarm.objective_fn.upper_bounds
    for p in swarm.particles:
        assert np.all(p.position >= lb), "Partícula fuera del límite inferior"
        assert np.all(p.position <= ub), "Partícula fuera del límite superior"