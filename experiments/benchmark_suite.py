# experiments/benchmark_suite.py
from objectives.sphere import Sphere
from objectives.rosenbrock import Rosenbrock
from objectives.rastrigin import Rastrigin
from objectives.ackley import Ackley

# Instancias reproducibles: (nombre, clase, dimensión, seed)
# Estas combinaciones son el conjunto estándar de evaluación.
# Fijamos seeds para garantizar reproducibilidad entre estrategias.

BENCHMARK_INSTANCES = [
    # Sphere — fácil, para verificar correctitud
    {"name": "Sphere_d2",       "fn": Sphere(dim=2),       "seed": 42},
    {"name": "Sphere_d10",      "fn": Sphere(dim=10),      "seed": 42},
    {"name": "Sphere_d30",      "fn": Sphere(dim=30),      "seed": 42},
    # Rosenbrock — valle estrecho, difícil de seguir
    {"name": "Rosenbrock_d2",   "fn": Rosenbrock(dim=2),   "seed": 42},
    {"name": "Rosenbrock_d10",  "fn": Rosenbrock(dim=10),  "seed": 42},
    {"name": "Rosenbrock_d30",  "fn": Rosenbrock(dim=30),  "seed": 42},
    # Rastrigin — multimodal, muchos mínimos locales
    {"name": "Rastrigin_d2",    "fn": Rastrigin(dim=2),    "seed": 42},
    {"name": "Rastrigin_d10",   "fn": Rastrigin(dim=10),   "seed": 42},
    {"name": "Rastrigin_d30",   "fn": Rastrigin(dim=30),   "seed": 42},
    # Ackley — multimodal con mínimo global rodeado de locales
    {"name": "Ackley_d2",       "fn": Ackley(dim=2),       "seed": 42},
    {"name": "Ackley_d10",      "fn": Ackley(dim=10),      "seed": 42},
    {"name": "Ackley_d30",      "fn": Ackley(dim=30),      "seed": 42},
]