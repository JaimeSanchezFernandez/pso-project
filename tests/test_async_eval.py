# tests/test_async_eval.py
import numpy as np
import pytest
from objectives.functions import Sphere, Rastrigin
from parallel.sequential import SequentialEvaluator
from parallel.async_eval import AsyncEvaluator


@pytest.fixture
def positions():
    return np.random.default_rng(42).uniform(-5, 5, (20, 5))

@pytest.fixture
def fn():
    return Sphere(dim=5)


def test_shape_correcto(positions, fn):
    ev = AsyncEvaluator(latency_mean=0.001, seed=42)
    assert ev.evaluate(positions, fn).shape == (20,)

def test_mismo_resultado_que_v0(positions, fn):
    fits0 = SequentialEvaluator().evaluate(positions, fn)
    fits3 = AsyncEvaluator(latency_mean=0.001, seed=42).evaluate(positions, fn)
    assert np.allclose(fits0, fits3)

def test_valores_no_nan(positions, fn):
    ev = AsyncEvaluator(latency_mean=0.001, seed=42)
    assert not np.any(np.isnan(ev.evaluate(positions, fn)))

def test_valores_positivos_sphere(positions, fn):
    ev = AsyncEvaluator(latency_mean=0.001, seed=42)
    assert np.all(ev.evaluate(positions, fn) >= 0.0)

def test_funciona_con_rastrigin():
    fn  = Rastrigin(dim=3)
    pos = np.random.default_rng(7).uniform(-5, 5, (10, 3))
    assert np.allclose(
        AsyncEvaluator(latency_mean=0.001, seed=42).evaluate(pos, fn),
        SequentialEvaluator().evaluate(pos, fn),
    )

def test_pso_completo_mismo_resultado():
    from core.swarm import Enjambre
    from core.stopcriteria import MaxIteraciones

    fn     = Sphere(dim=5)
    kwargs = dict(funcion_objetivo=fn, num_particulas=20,
                  criterio_parada=MaxIteraciones(30), semilla=42)
    r0 = Enjambre(evaluador=SequentialEvaluator(), **kwargs).ejecutar()
    r3 = Enjambre(evaluador=AsyncEvaluator(latency_mean=0.0, seed=42), **kwargs).ejecutar()
    assert r0["fitness_global"] == pytest.approx(r3["fitness_global"], rel=1e-8)

def test_repr():
    ev = AsyncEvaluator(latency_mean=0.01, latency_std=0.003)
    assert "AsyncEvaluator" in repr(ev)
    assert "0.01" in repr(ev)