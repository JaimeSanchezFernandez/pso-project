# parallel/numpy_eval.py
"""
V4 — Evaluador y motor PSO completamente vectorizados con NumPy.

V0-V3 evalúan fitness con bucles sobre partículas. V4 lleva TODO al nivel
de matriz: posiciones (n, dim), velocidades (n, dim), pBest (n, dim).

Una sola operación reemplaza el bucle "for p in particulas":
    v = w*V + c1*r1*(P - X) + c2*r2*(G - X)
    X = X + v

BLAS/LAPACK ejecutan estas operaciones en paralelo implícito usando
múltiples cores. Sin GIL, sin IPC, sin serialización.
"""

import numpy as np
from objectives.functions import ObjectiveFunction
from parallel.sequential import FitnessEvaluator


class NumpyEvaluator(FitnessEvaluator):
    """
    V4a — Solo la evaluación de fitness vectorizada.
    Compatible con el motor PSO estándar (core/swarm.py).
    """

    def evaluate(
        self,
        positions:    np.ndarray,
        objective_fn: ObjectiveFunction,
    ) -> np.ndarray:
        try:
            result = objective_fn(positions)
            if np.ndim(result) == 1 and len(result) == len(positions):
                return np.asarray(result, dtype=float)
            raise ValueError
        except Exception:
            # Fallback sin bucle Python explícito
            return np.apply_along_axis(objective_fn, 1, positions)

    def __repr__(self) -> str:
        return "NumpyEvaluator()"


class NumpySwarm:
    """
    V4b — Motor PSO completamente matricial.

    Reemplaza core/swarm.py. Cada paso del algoritmo opera sobre
    matrices completas, eliminando todos los bucles Python.

    Misma API de salida que Enjambre.ejecutar().
    """

    def __init__(
        self,
        funcion_objetivo: ObjectiveFunction,
        num_particulas:   int        = 30,
        w:                float      = 0.7,
        c1:               float      = 1.5,
        c2:               float      = 1.5,
        max_iter:         int        = 200,
        semilla:          int | None = None,
    ) -> None:
        self.fn       = funcion_objetivo
        self.n        = num_particulas
        self.w        = w
        self.c1       = c1
        self.c2       = c2
        self.max_iter = max_iter
        self.rng      = np.random.default_rng(semilla)

    def _eval_batch(self, X: np.ndarray) -> np.ndarray:
        try:
            result = self.fn(X)
            if np.ndim(result) == 1 and len(result) == self.n:
                return np.asarray(result, dtype=float)
            raise ValueError
        except Exception:
            return np.apply_along_axis(self.fn, 1, X)

    def ejecutar(self) -> dict:
        import time

        dim = self.fn.dim
        lb  = self.fn.lower_bounds
        ub  = self.fn.upper_bounds

        # Inicialización matricial
        X = self.rng.uniform(lb, ub, (self.n, dim))
        V = self.rng.uniform(-(ub - lb), (ub - lb), (self.n, dim))
        P = X.copy()

        t_eval   = 0.0
        t_update = 0.0

        t0    = time.perf_counter()
        F     = self._eval_batch(X)
        t_eval += time.perf_counter() - t0

        P_fit = F.copy()
        g_idx = int(np.argmin(P_fit))
        G     = P[g_idx].copy()
        g_fit = float(P_fit[g_idx])
        historial = [g_fit]

        for _ in range(self.max_iter):

            # Actualización vectorizada: velocidad y posición
            t0 = time.perf_counter()
            r1 = self.rng.random((self.n, dim))
            r2 = self.rng.random((self.n, dim))
            V  = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G - X)
            X  = X + V

            # Clamp + anular velocidad fuera de límites
            fuera       = (X < lb) | (X > ub)
            X           = np.clip(X, lb, ub)
            V[fuera]    = 0.0
            t_update   += time.perf_counter() - t0

            # Evaluación batch
            t0    = time.perf_counter()
            F     = self._eval_batch(X)
            t_eval += time.perf_counter() - t0

            # Actualizar pBest y gBest
            t0      = time.perf_counter()
            mejora  = F < P_fit
            P[mejora]     = X[mejora]
            P_fit[mejora] = F[mejora]

            g_idx_new = int(np.argmin(P_fit))
            if P_fit[g_idx_new] < g_fit:
                g_fit = float(P_fit[g_idx_new])
                G     = P[g_idx_new].copy()

            t_update   += time.perf_counter() - t0
            historial.append(g_fit)

        return {
            "fitness_global":       g_fit,
            "pos_global":           G,
            "historial_fitness":    historial,
            "num_iteraciones":      self.max_iter,
            "tiempo_evaluacion":    t_eval,
            "tiempo_actualizacion": t_update,
            "overhead":             0.0,
        }

    def __repr__(self) -> str:
        return f"NumpySwarm(n={self.n}, w={self.w}, c1={self.c1}, c2={self.c2})"