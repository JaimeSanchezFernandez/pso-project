# parallel/__init__.py
from .sequential import FitnessEvaluator, SequentialEvaluator
from .threading_eval import ThreadingEvaluator
from .process_eval import ProcessEvaluator

__all__ = ["FitnessEvaluator", "SequentialEvaluator", "ThreadingEvaluator", "ProcessEvaluator"]