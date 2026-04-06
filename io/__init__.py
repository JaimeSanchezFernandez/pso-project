# io/__init__.py
from .persistence import save_result
from .loader import load_result, load_all_results, load_summary

__all__ = ["save_result", "load_result", "load_all_results", "load_summary"]