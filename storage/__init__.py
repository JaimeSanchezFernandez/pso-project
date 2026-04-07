# storage/__init__.py
from .persistence import guardar_resultado
from .loader import cargar_resultado, cargar_todos_resultados, cargar_resumen

__all__ = ["guardar_resultado", "cargar_resultado", "cargar_todos_resultados", "cargar_resumen"]