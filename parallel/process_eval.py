# parallel/process_eval.py — V2
import numpy as np
import multiprocessing
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

    Optimización clave: el pool se crea una sola vez en __init__
    y se reutiliza en todas las iteraciones del PSO. Esto elimina
    el overhead de crear y destruir procesos en cada evaluación,
    que era el principal cuello de botella de la versión anterior.

    Limitación conocida: hay que llamar a shutdown() al terminar,
    o usar el evaluador como context manager.

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

    def __init__(self, max_workers: int | None = None, batch_size: int | None = None) -> None:
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._ctx = multiprocessing.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Devuelve el executor existente o crea uno nuevo."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=self._ctx,
            )
        return self._executor

    def _make_batches(self, positions: np.ndarray) -> list[np.ndarray]:
        """Divide las posiciones en bloques de tamaño batch_size."""
        n = len(positions)
        workers = self.max_workers or self._ctx.cpu_count() or 4
        if self.batch_size is None:
            size = max(1, n // workers)
        else:
            size = self.batch_size
        return [positions[i:i + size] for i in range(0, n, size)]

    def evaluate(self, positions: np.ndarray, objective_fn: ObjectiveFunction) -> np.ndarray:
        batches = self._make_batches(positions)
        args = [(batch, objective_fn) for batch in batches]

        executor = self._get_executor()
        results = list(executor.map(_evaluate_batch, args))

        fitnesses = [fit for batch_result in results for fit in batch_result]
        return np.array(fitnesses)

    def shutdown(self) -> None:
        """Cierra el pool de procesos limpiamente."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"ProcessEvaluator(max_workers={self.max_workers}, batch_size={self.batch_size})"