# tests/test_parallel.py
import numpy as np
import pytest
from core.swarm import Swarm
from core.stopcriteria import MaxIterations
from objectives.sphere import Sphere
from objectives.rastrigin import Rastrigin
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval import ProcessEvaluator


def run_with_evaluator(evaluator, fn, seed=42, max_iter=50):
    swarm = Swarm(
        objective_fn=fn,
        evaluator=evaluator,
        n_particles=20,
        w=0.7, c1=1.5, c2=1.5,
        stop_criterion=MaxIterations(max_iter),
        seed=seed,
    )
    return swarm.run()


def test_v0_v1_same_result():
    """V0 y V1 deben producir exactamente el mismo fitness con la misma seed."""
    fn = Sphere(dim=5)
    r0 = run_with_evaluator(SequentialEvaluator(), fn, seed=42)
    r1 = run_with_evaluator(ThreadingEvaluator(), fn, seed=42)
    assert r0["gbest_fit"] == pytest.approx(r1["gbest_fit"], rel=1e-10)


def test_v0_v2_same_result():
    """V0 y V2 deben producir exactamente el mismo fitness con la misma seed."""
    fn = Sphere(dim=5)
    r0 = run_with_evaluator(SequentialEvaluator(), fn, seed=42)
    r2 = run_with_evaluator(ProcessEvaluator(), fn, seed=42)
    assert r0["gbest_fit"] == pytest.approx(r2["gbest_fit"], rel=1e-10)


def test_all_evaluators_same_result():
    """V0, V1 y V2 deben converger al mismo resultado en Rastrigin."""
    fn = Rastrigin(dim=3)
    r0 = run_with_evaluator(SequentialEvaluator(), fn, seed=99)
    r1 = run_with_evaluator(ThreadingEvaluator(), fn, seed=99)
    r2 = run_with_evaluator(ProcessEvaluator(), fn, seed=99)
    assert r0["gbest_fit"] == pytest.approx(r1["gbest_fit"], rel=1e-10)
    assert r0["gbest_fit"] == pytest.approx(r2["gbest_fit"], rel=1e-10)


def test_evaluators_return_correct_shape():
    """Cada evaluador debe devolver un array de shape (n_particles,)."""
    fn = Sphere(dim=5)
    positions = np.random.default_rng(42).uniform(-5, 5, (30, 5))
    for EvalClass in [SequentialEvaluator, ThreadingEvaluator, ProcessEvaluator]:
        evaluator = EvalClass()
        fitnesses = evaluator.evaluate(positions, fn)
        assert fitnesses.shape == (30,), f"{EvalClass.__name__} devolvió shape incorrecto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])