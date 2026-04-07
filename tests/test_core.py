# tests/test_core.py
import numpy as np
import pytest
from core.swarm import Enjambre
from core.stopcriteria import MaxIteraciones
from objectives.functions import Sphere
from parallel.sequential import SequentialEvaluator


def crear_enjambre(semilla=42, max_iter=100):
    return Enjambre(
        funcion_objetivo=Sphere(dim=2),
        evaluador=SequentialEvaluator(),
        num_particulas=20,
        w=0.7, c1=1.5, c2=1.5,
        criterio_parada=MaxIteraciones(max_iter),
        semilla=semilla,
    )


def test_reproducibilidad():
    """Dos ejecuciones con la misma semilla deben dar exactamente el mismo resultado."""
    resultado1 = crear_enjambre(semilla=42).ejecutar()
    resultado2 = crear_enjambre(semilla=42).ejecutar()
    assert resultado1["fitness_global"] == resultado2["fitness_global"]
    assert np.allclose(resultado1["pos_global"], resultado2["pos_global"])


def test_semillas_distintas():
    """Semillas distintas deben dar resultados distintos."""
    resultado1 = crear_enjambre(semilla=42).ejecutar()
    resultado2 = crear_enjambre(semilla=99).ejecutar()
    assert resultado1["fitness_global"] != resultado2["fitness_global"]


def test_monotonicidad_fitness_global():
    """El mejor global nunca debe empeorar entre iteraciones."""
    resultado = crear_enjambre(semilla=42, max_iter=200).ejecutar()
    historial = resultado["historial_fitness"]
    for i in range(1, len(historial)):
        assert historial[i] <= historial[i - 1] + 1e-12, (
            f"fitness_global empeoró en iteración {i}: {historial[i-1]} → {historial[i]}"
        )


def test_sphere_converge():
    """PSO debe converger cerca de 0 en Sphere(d=2) con parámetros razonables."""
    resultado = crear_enjambre(semilla=42, max_iter=500).ejecutar()
    assert resultado["fitness_global"] < 1e-3, (
        f"Sphere no convergió suficiente: fitness_global={resultado['fitness_global']}"
    )


def test_limites_respetados():
    """Las posiciones deben estar dentro de los límites."""
    enjambre = crear_enjambre(semilla=42, max_iter=100)
    enjambre._inicializar()
    lb = enjambre.funcion_objetivo.lower_bounds
    ub = enjambre.funcion_objetivo.upper_bounds
    for p in enjambre.particulas:
        assert np.all(p.posicion >= lb), "Partícula fuera del límite inferior"
        assert np.all(p.posicion <= ub), "Partícula fuera del límite superior"