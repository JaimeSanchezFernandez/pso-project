# parallel/async_eval.py
"""
V3 — Evaluador asíncrono con asyncio.

asyncio tiene sentido cuando la función objetivo requiere I/O externo:
consultar precios a una API REST, leer métricas de un servidor, etc.
asyncio.gather() envía todas las peticiones a la vez y recoge las
respuestas cuando llegan, en vez de esperar secuencialmente.

La latencia se simula con asyncio.sleep(). En producción se sustituiría
por llamadas reales con aiohttp.
"""

import asyncio
import numpy as np
from objectives.functions import ObjectiveFunction
from parallel.sequential import FitnessEvaluator


class AsyncEvaluator(FitnessEvaluator):
    """
    V3 — Evaluación concurrente mediante asyncio.

    Cada partícula se evalúa en una corrutina independiente que simula
    una llamada asíncrona a un servicio externo de precios.
    asyncio.gather() ejecuta todas de forma concurrente en un único hilo.

    Parameters
    ----------
    latency_mean : float
        Latencia media simulada por evaluación (segundos). Default: 0.01s
    latency_std : float
        Desviación estándar del jitter de red. Default: 0.003s
    seed : int | None
        Semilla para reproducibilidad de latencias. None = no reproducible.
    """

    def __init__(
        self,
        latency_mean: float    = 0.01,
        latency_std:  float    = 0.003,
        seed:         int|None = None,
    ) -> None:
        self.latency_mean = latency_mean
        self.latency_std  = latency_std
        self._rng         = np.random.default_rng(seed)

    async def _evaluar_particula(
        self,
        idx:          int,
        pos:          np.ndarray,
        objective_fn: ObjectiveFunction,
        latencias:    np.ndarray,
    ) -> tuple[int, float]:
        """
        Simula la llamada al servicio externo:
          1. await sleep  → latencia de red
          2. objective_fn → cómputo del fitness

        En producción:
            async with aiohttp.ClientSession() as s:
                async with s.post(API_URL, json={"w": pos.tolist()}) as r:
                    data = await r.json()
                    return idx, data["sharpe"]
        """
        await asyncio.sleep(latencias[idx])
        return idx, objective_fn(pos)

    async def _evaluar_todas(
        self,
        positions:    np.ndarray,
        objective_fn: ObjectiveFunction,
    ) -> np.ndarray:
        """
        Lanza todas las corrutinas con gather().
        Tiempo de pared ≈ max(latencias), no sum(latencias).
        """
        n         = len(positions)
        latencias = np.abs(self._rng.normal(self.latency_mean, self.latency_std, n))

        tareas     = [
            self._evaluar_particula(i, positions[i], objective_fn, latencias)
            for i in range(n)
        ]
        resultados = await asyncio.gather(*tareas)

        fitnesses = np.empty(n)
        for idx, fitness in resultados:
            fitnesses[idx] = fitness
        return fitnesses

    def evaluate(
        self,
        positions:    np.ndarray,
        objective_fn: ObjectiveFunction,
    ) -> np.ndarray:
        return asyncio.run(self._evaluar_todas(positions, objective_fn))

    def __repr__(self) -> str:
        return (
            f"AsyncEvaluator("
            f"latency_mean={self.latency_mean}s, "
            f"latency_std={self.latency_std}s)"
        )