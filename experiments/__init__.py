# experiments/__init__.py
from .runner import ejecutar_experimento
from .grid_search import busqueda_grid
from .benchmark_suite import BENCHMARK_INSTANCES

__all__ = ["ejecutar_experimento", "busqueda_grid", "BENCHMARK_INSTANCES"]