# core/swarm.py
import numpy as np
import logging
import time
from .particle import Particle
from .topology import Topology, GlobalBest
from .stopcriteria import StopCriterion, MaxIterations
from objectives.base import ObjectiveFunction

logger = logging.getLogger(__name__)


class Swarm:
    """
    Motor principal del PSO canónico.

    El evaluador de fitness se inyecta desde fuera (parallel/),
    lo que permite cambiar entre V0, V1 y V2 sin tocar este fichero.

    Estrategia de límites: clamp
    Las partículas que salen del espacio de búsqueda se recortan
    al límite más cercano y su velocidad se anula en esa dimensión.
    Esta estrategia es simple, estable y fácil de razonar.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        evaluator,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        topology: Topology = None,
        stop_criterion: StopCriterion = None,
        seed: int = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.evaluator = evaluator
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.topology = topology or GlobalBest()
        self.stop_criterion = stop_criterion or MaxIterations(200)
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.particles: list[Particle] = []
        self.gbest_pos: np.ndarray = None
        self.gbest_fit: float = float("inf")
        self.fitness_history: list[float] = []

        # Historial de posiciones y gbest para visualización
        self.position_history: list[np.ndarray] = []
        self.gbest_history: list[np.ndarray] = []

        # Timings desglosados
        self.time_eval: float = 0.0
        self.time_update: float = 0.0
        self.time_total: float = 0.0

    def _initialize(self) -> None:
        """Inicializa posiciones y velocidades aleatorias dentro de los límites."""
        lb = self.objective_fn.lower_bounds
        ub = self.objective_fn.upper_bounds

        self.particles = []
        for _ in range(self.n_particles):
            pos = self.rng.uniform(lb, ub)
            vel = self.rng.uniform(-(ub - lb), (ub - lb))
            p = Particle(
                position=pos,
                velocity=vel,
                pbest_pos=pos.copy(),
                pbest_fit=float("inf"),
            )
            self.particles.append(p)

    def _clamp(self, particle: Particle) -> None:
        """
        Aplica estrategia de límites: clamp.
        Si una dimensión sale del rango, se recorta y la velocidad
        en esa dimensión se pone a cero para evitar rebotes.
        """
        lb = self.objective_fn.lower_bounds
        ub = self.objective_fn.upper_bounds
        out_of_bounds = (particle.position < lb) | (particle.position > ub)
        particle.position = np.clip(particle.position, lb, ub)
        particle.velocity[out_of_bounds] = 0.0

    def _update_velocity(self, particle: Particle, gbest_pos: np.ndarray) -> None:
        """Actualiza la velocidad según la ecuación PSO canónica."""
        dim = self.objective_fn.dim
        r1 = self.rng.random(dim)
        r2 = self.rng.random(dim)

        cognitive = self.c1 * r1 * (particle.pbest_pos - particle.position)
        social = self.c2 * r2 * (gbest_pos - particle.position)
        particle.velocity = self.w * particle.velocity + cognitive + social

    def _update_position(self, particle: Particle) -> None:
        """Actualiza la posición y aplica clamp."""
        particle.position = particle.position + particle.velocity
        self._clamp(particle)

    def run(self) -> dict:
        """
        Ejecuta el PSO hasta que el criterio de parada se cumple.

        Instrumentación de tiempos:
        - time_eval   : tiempo acumulado en evaluación de fitness
        - time_update : tiempo acumulado en actualización de partículas
        - time_total  : tiempo total de ejecución

        Returns
        -------
        dict con gbest_fit, gbest_pos, fitness_history, timings y n_iterations
        """
        self.stop_criterion.reset()
        self._initialize()
        self.time_eval = 0.0
        self.time_update = 0.0
        self.position_history = []
        self.gbest_history = []

        t_total_start = time.perf_counter()

        # Evaluación inicial
        positions = np.array([p.position for p in self.particles])

        t0 = time.perf_counter()
        fitnesses = self.evaluator.evaluate(positions, self.objective_fn)
        self.time_eval += time.perf_counter() - t0

        for particle, fit in zip(self.particles, fitnesses):
            particle.update_personal_best(fit)

        self.gbest_pos = self.topology.get_best_position(self.particles)
        self.gbest_fit = min(p.pbest_fit for p in self.particles)
        self.fitness_history = [self.gbest_fit]

        logger.info(
            f"PSO iniciado | particles={self.n_particles} | seed={self.seed}"
        )

        iteration = 0
        while not self.stop_criterion.should_stop(iteration, self.gbest_fit, self.fitness_history):

            # Registrar historial para visualización
            self.position_history.append(
                np.array([p.position.copy() for p in self.particles])
            )
            self.gbest_history.append(self.gbest_pos.copy())

            # Actualizar velocidades y posiciones
            t0 = time.perf_counter()
            for particle in self.particles:
                self._update_velocity(particle, self.gbest_pos)
                self._update_position(particle)
            self.time_update += time.perf_counter() - t0

            # Evaluar fitness
            positions = np.array([p.position for p in self.particles])
            t0 = time.perf_counter()
            fitnesses = self.evaluator.evaluate(positions, self.objective_fn)
            self.time_eval += time.perf_counter() - t0

            # Actualizar pbest y gbest
            for particle, fit in zip(self.particles, fitnesses):
                particle.update_personal_best(fit)

            self.gbest_pos = self.topology.get_best_position(self.particles)
            self.gbest_fit = min(p.pbest_fit for p in self.particles)
            self.fitness_history.append(self.gbest_fit)

            iteration += 1
            logger.debug(
                f"iter={iteration} | gbest_fit={self.gbest_fit:.6e} | "
                f"t_eval={self.time_eval:.4f}s | t_update={self.time_update:.4f}s"
            )

        self.time_total = time.perf_counter() - t_total_start
        overhead = self.time_total - self.time_eval - self.time_update

        logger.info(
            f"PSO finalizado | iter={iteration} | gbest_fit={self.gbest_fit:.6e} | "
            f"t_total={self.time_total:.4f}s | t_eval={self.time_eval:.4f}s | "
            f"t_update={self.time_update:.4f}s | overhead={overhead:.4f}s"
        )

        return {
            "gbest_fit":        self.gbest_fit,
            "gbest_pos":        self.gbest_pos,
            "fitness_history":  self.fitness_history,
            "n_iterations":     iteration,
            "seed":             self.seed,
            "time_eval":        self.time_eval,
            "time_update":      self.time_update,
            "time_total":       self.time_total,
            "overhead":         overhead,
            "position_history": self.position_history,
            "gbest_history":    self.gbest_history,
        }