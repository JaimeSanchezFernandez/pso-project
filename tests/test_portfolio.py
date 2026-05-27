# tests/test_portfolio.py
import numpy as np
import pytest
from objectives.portfolio import PortfolioSharpe


@pytest.fixture
def fn():
    return PortfolioSharpe()


def test_dimensiones(fn):
    assert fn.dim == 8
    assert len(fn.asset_names) == 8
    assert len(fn.bounds) == 8

def test_bounds_son_0_1(fn):
    for lb, ub in fn.bounds:
        assert lb == 0.0
        assert ub == 1.0

def test_normalizar_pesos_suma_1(fn):
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(0, 1, 8)
        w = fn.normalizar_pesos(x)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

def test_normalizar_pesos_no_negativos(fn):
    x = np.array([-1.0, 0.5, 0.0, 2.0, -0.3, 1.0, 0.0, 0.8])
    w = fn.normalizar_pesos(x)
    assert np.all(w >= 0.0)

def test_normalizar_pesos_todos_cero(fn):
    w = fn.normalizar_pesos(np.zeros(8))
    assert w == pytest.approx(np.ones(8) / 8, abs=1e-10)

def test_fitness_es_negativo(fn):
    assert fn(np.ones(8) / 8) < 0.0

def test_reproducibilidad(fn):
    x = np.array([0.2, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1])
    assert fn(x) == fn(x)

def test_metricas_estructura(fn):
    m = fn.metricas(np.ones(8) / 8)
    assert "pesos" in m
    assert "retorno_anual" in m
    assert "volatilidad" in m
    assert "sharpe" in m

def test_metricas_sharpe_positivo(fn):
    assert fn.metricas(np.ones(8) / 8)["sharpe"] > 0.0

def test_metricas_consistencia_con_call(fn):
    x = np.ones(8) / 8
    m = fn.metricas(x)
    assert m["sharpe"] == pytest.approx(-fn(x), abs=1e-4)

def test_datos_reproducibles():
    fn1 = PortfolioSharpe(data_seed=0)
    fn2 = PortfolioSharpe(data_seed=0)
    assert np.allclose(fn1.mu_anual, fn2.mu_anual)
    assert np.allclose(fn1.cov_anual, fn2.cov_anual)

def test_datos_distintos_seed():
    fn1 = PortfolioSharpe(data_seed=0)
    fn2 = PortfolioSharpe(data_seed=99)
    assert not np.allclose(fn1.mu_anual, fn2.mu_anual)

def test_pso_mejora_cartera_uniforme():
    from core.swarm import Enjambre
    from core.stopcriteria import MaxIteraciones
    from parallel.sequential import SequentialEvaluator

    fn = PortfolioSharpe()
    sharpe_uniforme = fn.metricas(np.ones(8) / 8)["sharpe"]
    resultado = Enjambre(
        funcion_objetivo=fn,
        evaluador=SequentialEvaluator(),
        num_particulas=40,
        criterio_parada=MaxIteraciones(300),
        semilla=42,
    ).ejecutar()
    assert fn.metricas(resultado["pos_global"])["sharpe"] > sharpe_uniforme