# core/swarm.py
import numpy as np
import logging
import time
from .particle import Particle
from .topology import Topologia, MejorGlobal
from .stopcriteria import CriterioParada, MaxIteraciones
from objectives.functions import ObjectiveFunction

logger = logging.getLogger(__name__)


class Enjambre:
    """
    Motor principal del PSO canónico.

    El evaluador de fitness se inyecta desde fuera (parallel/),
    lo que permite cambiar entre V0, V1 y V2 sin tocar este fichero.

    Estrategia de límites: clamp
    Las partículas que salen del espacio de búsqueda se recortan
    al límite más cercano y su velocidad se anula en esa dimensión.
    """

    def __init__(
        self,
        funcion_objetivo: ObjectiveFunction,
        evaluador,
        num_particulas: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        topologia: Topologia = None,
        criterio_parada: CriterioParada = None,
        semilla: int = None,
    ) -> None:
        self.funcion_objetivo = funcion_objetivo
        self.evaluador = evaluador
        self.num_particulas = num_particulas
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.topologia = topologia or MejorGlobal()
        self.criterio_parada = criterio_parada or MaxIteraciones(200)
        self.semilla = semilla

        self.rng = np.random.default_rng(semilla)
        self.particulas: list[Particle] = []
        self.pos_global: np.ndarray = None
        self.fitness_global: float = float("inf")
        self.historial_fitness: list[float] = []

        self.historial_posiciones: list[np.ndarray] = []
        self.historial_mejor_global: list[np.ndarray] = []

        self.tiempo_evaluacion: float = 0.0
        self.tiempo_actualizacion: float = 0.0
        self.tiempo_total: float = 0.0

    def _inicializar(self) -> None:
        """Inicializa posiciones y velocidades aleatorias dentro de los límites."""
        lb = self.funcion_objetivo.lower_bounds
        ub = self.funcion_objetivo.upper_bounds

        self.particulas = []
        for _ in range(self.num_particulas):
            pos = self.rng.uniform(lb, ub)
            vel = self.rng.uniform(-(ub - lb), (ub - lb))
            p = Particle(
                posicion=pos,
                velocidad=vel,
                mejor_pos=pos.copy(),
                mejor_fitness=float("inf"),
            )
            self.particulas.append(p)

    def _aplicar_limites(self, particula: Particle) -> None:
        """
        Estrategia de límites: clamp.
        Si una dimensión sale del rango se recorta y la velocidad
        en esa dimensión se pone a cero.
        """
        lb = self.funcion_objetivo.lower_bounds
        ub = self.funcion_objetivo.upper_bounds
        fuera = (particula.posicion < lb) | (particula.posicion > ub)
        particula.posicion = np.clip(particula.posicion, lb, ub)
        particula.velocidad[fuera] = 0.0

    def _actualizar_velocidad(self, particula: Particle, pos_global: np.ndarray) -> None:
        """Actualiza la velocidad según la ecuación PSO canónica."""
        dim = self.funcion_objetivo.dim
        r1 = self.rng.random(dim)
        r2 = self.rng.random(dim)

        inercia = self.w * particula.velocidad
        memoria_individual = self.c1 * r1 * (particula.mejor_pos - particula.posicion)
        atraccion_global = self.c2 * r2 * (pos_global - particula.posicion)
        particula.velocidad = inercia + memoria_individual + atraccion_global

    def _actualizar_posicion(self, particula: Particle) -> None:
        """Actualiza la posición y aplica clamp."""
        particula.posicion = particula.posicion + particula.velocidad
        self._aplicar_limites(particula)

    def ejecutar(self) -> dict:
        """
        Ejecuta el PSO hasta que el criterio de parada se cumple.

        Returns
        -------
        dict con fitness_global, pos_global, historial_fitness y tiempos
        """
        self.criterio_parada.reiniciar()
        self._inicializar()
        self.tiempo_evaluacion = 0.0
        self.tiempo_actualizacion = 0.0
        self.historial_posiciones = []
        self.historial_mejor_global = []

        t_inicio = time.perf_counter()

        posiciones = np.array([p.posicion for p in self.particulas])
        t0 = time.perf_counter()
        evaluaciones = self.evaluador.evaluate(posiciones, self.funcion_objetivo)
        self.tiempo_evaluacion += time.perf_counter() - t0

        for particula, fit in zip(self.particulas, evaluaciones):
            particula.actualizar_mejor(fit)

        self.pos_global = self.topologia.obtener_mejor_posicion(self.particulas)
        self.fitness_global = min(p.mejor_fitness for p in self.particulas)
        self.historial_fitness = [self.fitness_global]

        logger.info(
            f"PSO iniciado | particulas={self.num_particulas} | semilla={self.semilla}"
        )

        iteracion = 0
        while not self.criterio_parada.debe_parar(iteracion, self.fitness_global, self.historial_fitness):

            self.historial_posiciones.append(
                np.array([p.posicion.copy() for p in self.particulas])
            )
            self.historial_mejor_global.append(self.pos_global.copy())

            t0 = time.perf_counter()
            for particula in self.particulas:
                self._actualizar_velocidad(particula, self.pos_global)
                self._actualizar_posicion(particula)
            self.tiempo_actualizacion += time.perf_counter() - t0

            posiciones = np.array([p.posicion for p in self.particulas])
            t0 = time.perf_counter()
            evaluaciones = self.evaluador.evaluate(posiciones, self.funcion_objetivo)
            self.tiempo_evaluacion += time.perf_counter() - t0

            for particula, fit in zip(self.particulas, evaluaciones):
                particula.actualizar_mejor(fit)

            self.pos_global = self.topologia.obtener_mejor_posicion(self.particulas)
            self.fitness_global = min(p.mejor_fitness for p in self.particulas)
            self.historial_fitness.append(self.fitness_global)

            iteracion += 1
            logger.debug(
                f"iter={iteracion} | fitness_global={self.fitness_global:.6e} | "
                f"t_eval={self.tiempo_evaluacion:.4f}s | t_update={self.tiempo_actualizacion:.4f}s"
            )

        self.tiempo_total = time.perf_counter() - t_inicio
        overhead = self.tiempo_total - self.tiempo_evaluacion - self.tiempo_actualizacion

        logger.info(
            f"PSO finalizado | iter={iteracion} | fitness_global={self.fitness_global:.6e} | "
            f"t_total={self.tiempo_total:.4f}s | t_eval={self.tiempo_evaluacion:.4f}s | "
            f"t_update={self.tiempo_actualizacion:.4f}s | overhead={overhead:.4f}s"
        )

        return {
            "fitness_global":       self.fitness_global,
            "pos_global":           self.pos_global,
            "historial_fitness":    self.historial_fitness,
            "num_iteraciones":      iteracion,
            "semilla":              self.semilla,
            "tiempo_evaluacion":    self.tiempo_evaluacion,
            "tiempo_actualizacion": self.tiempo_actualizacion,
            "tiempo_total":         self.tiempo_total,
            "overhead":             overhead,
            "historial_posiciones": self.historial_posiciones,
            "historial_mejor_global": self.historial_mejor_global,
        }