# tests/test_numpy_eval.py
import numpy as np
import pytest
from objectives.functions import Sphere, Rastrigin, Ackley
from parallel.sequential import SequentialEvaluator
from parallel.numpy_eval import NumpyEvaluator, NumpySwarm


@pytest.fixture
def positions():
    return np.random.default_rng(42).uniform(-5, 5, (30, 10))

@pytest.fixture
def fn():
    return Sphere(dim=10)


def test_shape_correcto(positions, fn):
    assert NumpyEvaluator().evaluate(positions, fn).shape == (30,)

def test_mismo_resultado_que_v0(positions, fn):
    assert np.allclose(
        SequentialEvaluator().evaluate(positions, fn),
        NumpyEvaluator().evaluate(positions, fn),
    )

def test_valores_no_nan(positions, fn):
    assert not np.any(np.isnan(NumpyEvaluator().evaluate(positions, fn)))

def test_funciona_con_rastrigin():
    fn  = Rastrigin(dim=5)
    pos = np.random.default_rng(0).uniform(-5, 5, (20, 5))
    assert np.allclose(
        NumpyEvaluator().evaluate(pos, fn),
        SequentialEvaluator().evaluate(pos, fn),
    )

def test_funciona_con_ackley():
    fn  = Ackley(dim=8)
    pos = np.random.default_rng(1).uniform(-32, 32, (15, 8))
    assert np.allclose(
        NumpyEvaluator().evaluate(pos, fn),
        SequentialEvaluator().evaluate(pos, fn),
    )

def test_numpy_swarm_ejecutar_devuelve_claves(fn):
    r = NumpySwarm(fn, num_particulas=20, max_iter=10, semilla=42).ejecutar()
    assert {"fitness_global","pos_global","historial_fitness",
            "num_iteraciones","tiempo_evaluacion","tiempo_actualizacion","overhead"
            }.issubset(r.keys())

def test_numpy_swarm_pos_global_shape(fn):
    r = NumpySwarm(fn, num_particulas=20, max_iter=10, semilla=42).ejecutar()
    assert r["pos_global"].shape == (fn.dim,)

def test_numpy_swarm_historial_monotono(fn):
    h = NumpySwarm(fn, num_particulas=20, max_iter=100, semilla=42).ejecutar()["historial_fitness"]
    for i in range(1, len(h)):
        assert h[i] <= h[i-1] + 1e-12

def test_numpy_swarm_reproducibilidad(fn):
    r1 = NumpySwarm(fn, num_particulas=20, max_iter=50, semilla=7).ejecutar()
    r2 = NumpySwarm(fn, num_particulas=20, max_iter=50, semilla=7).ejecutar()
    assert r1["fitness_global"] == r2["fitness_global"]
    assert np.allclose(r1["pos_global"], r2["pos_global"])

def test_numpy_swarm_sphere_converge():
    r = NumpySwarm(Sphere(dim=2), num_particulas=30, max_iter=500, semilla=42).ejecutar()
    assert r["fitness_global"] < 1e-3

def test_numpy_swarm_limites_respetados(fn):
    r   = NumpySwarm(fn, num_particulas=20, max_iter=50, semilla=42).ejecutar()
    pos = r["pos_global"]
    assert np.all(pos >= fn.lower_bounds - 1e-10)
    assert np.all(pos <= fn.upper_bounds + 1e-10)

def test_numpy_swarm_tiempos_positivos(fn):
    r = NumpySwarm(fn, num_particulas=20, max_iter=20, semilla=42).ejecutar()
    assert r["tiempo_evaluacion"]    >= 0.0
    assert r["tiempo_actualizacion"] >= 0.0

def test_numpy_swarm_es_mas_rapido_que_v0():
    import time
    from core.swarm import Enjambre
    from core.stopcriteria import MaxIteraciones

    fn = Sphere(dim=50)
    t0 = time.perf_counter()
    Enjambre(fn, SequentialEvaluator(), num_particulas=50,
             criterio_parada=MaxIteraciones(200), semilla=42).ejecutar()
    t_v0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    NumpySwarm(fn, num_particulas=50, max_iter=200, semilla=42).ejecutar()
    t_v4 = time.perf_counter() - t0

    assert t_v4 < t_v0