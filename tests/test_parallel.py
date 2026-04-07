# tests/test_parallel.py
import numpy as np
import pytest
from core.swarm import Enjambre
from core.stopcriteria import MaxIteraciones
from objectives.functions import Sphere, Rastrigin
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator


def ejecutar_con_evaluador(evaluador, fn, semilla=42, max_iter=50):
    enjambre = Enjambre(
        funcion_objetivo=fn,
        evaluador=evaluador,
        num_particulas=20,
        w=0.7, c1=1.5, c2=1.5,
        criterio_parada=MaxIteraciones(max_iter),
        semilla=semilla,
    )
    return enjambre.ejecutar()


def test_v0_v1_mismo_resultado():
    """V0 y V1 deben producir exactamente el mismo fitness con la misma semilla."""
    fn = Sphere(dim=5)
    r0 = ejecutar_con_evaluador(SequentialEvaluator(), fn, semilla=42)
    r1 = ejecutar_con_evaluador(ThreadingEvaluator(), fn, semilla=42)
    assert r0["fitness_global"] == pytest.approx(r1["fitness_global"], rel=1e-10)


def test_v0_v2_mismo_resultado():
    """V0 y V2 deben producir exactamente el mismo fitness con la misma semilla."""
    fn = Sphere(dim=5)
    r0 = ejecutar_con_evaluador(SequentialEvaluator(), fn, semilla=42)
    r2 = ejecutar_con_evaluador(ProcessEvaluator(), fn, semilla=42)
    assert r0["fitness_global"] == pytest.approx(r2["fitness_global"], rel=1e-10)


def test_todos_evaluadores_mismo_resultado():
    """V0, V1 y V2 deben converger al mismo resultado en Rastrigin."""
    fn = Rastrigin(dim=3)
    r0 = ejecutar_con_evaluador(SequentialEvaluator(), fn, semilla=99)
    r1 = ejecutar_con_evaluador(ThreadingEvaluator(), fn, semilla=99)
    r2 = ejecutar_con_evaluador(ProcessEvaluator(), fn, semilla=99)
    assert r0["fitness_global"] == pytest.approx(r1["fitness_global"], rel=1e-10)
    assert r0["fitness_global"] == pytest.approx(r2["fitness_global"], rel=1e-10)


def test_evaluadores_forma_correcta():
    """Cada evaluador debe devolver un array de shape (num_particulas,)."""
    fn = Sphere(dim=5)
    posiciones = np.random.default_rng(42).uniform(-5, 5, (30, 5))
    for ClaseEval in [SequentialEvaluator, ThreadingEvaluator, ProcessEvaluator]:
        evaluador = ClaseEval()
        evaluaciones = evaluador.evaluate(posiciones, fn)
        assert evaluaciones.shape == (30,), f"{ClaseEval.__name__} devolvió shape incorrecto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])