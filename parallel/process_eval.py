# parallel/process_eval.py — V2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .base_evaluator import FitnessEvaluator
from objectives.base import ObjectiveFunction


def _evaluate_batch(args: tuple) -> list[float]:
    """
    Función auxiliar a nivel de módulo (necesario para pickle).

    Evalúa un bloque de partículas en un proceso hijo.
    Debe estar fuera de la clase porque multiprocessing serializa
    las tareas con pickle, y pickle no puede serializar métodos
    de instancia ni funciones lambda.

    Parameters
    ----------
    args : (positions_batch, objective_fn)
           positions_batch : array (batch_size, dim)
           objective_fn    : función objetivo serializable
    """
    positions_batch, objective_fn = args
    return [objective_fn(pos) for pos in positions_batch]


class ProcessEvaluator(FitnessEvaluator):
    """
    V2 — Evaluación paralela con procesos (ProcessPoolExecutor).

    A diferencia de los hilos, los procesos tienen su propio espacio
    de memoria y su propio GIL, por lo que pueden ejecutarse en
    paralelo real en múltiples cores.

    Coste de IPC (Inter-Process Communication):
    Cada tarea debe serializarse (pickle) para enviarse al proceso hijo
    y deserializarse al volver. Este overhead puede ser mayor que el
    beneficio para funciones objetivo baratas o enjambres pequeños.

    Optimización con batching:
    En vez de enviar una partícula por tarea (overhead máximo),
    se agrupan las partículas en bloques (batches) y se envía
    un bloque entero por tarea. Esto reduce el número de
    serializaciones y el overhead de IPC.

    Parameters
    ----------
    max_workers : número de procesos. None = usa os.cpu_count()
    batch_size  : partículas por tarea. None = un batch por worker
    """

    def __init__(self, max_workers: int = None, batch_size: int = None):
        self.max_workers = max_workers
        self.batch_size = batch_size

    def _make_batches(self, positions: np.ndarray) -> list[np.ndarray]:
        """Divide las posiciones en bloques de tamaño batch_size."""
        n = len(positions)
        if self.batch_size is None:
            workers = self.max_workers or 4
            size = max(1, n // workers)
        else:
            size = self.batch_size
        return [positions[i:i + size] for i in range(0, n, size)]

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        batches = self._make_batches(positions)
        args = [(batch, objective_fn) for batch in batches]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_evaluate_batch, args))

        fitnesses = [fit for batch_result in results for fit in batch_result]
        return np.array(fitnesses)