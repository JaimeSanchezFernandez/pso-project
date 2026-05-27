# run_portfolio.py
"""
Caso de uso real: Optimización de cartera de inversión con PSO.

Uso:
    python run_portfolio.py
    python run_portfolio.py --evaluador v4
    python run_portfolio.py --particulas 50 --iter 500 --semilla 42
    python run_portfolio.py --guardar
"""

import argparse
import logging
import time
import numpy as np

from objectives.portfolio    import PortfolioSharpe
from parallel.sequential     import SequentialEvaluator
from parallel.threading_eval import ThreadingEvaluator
from parallel.process_eval   import ProcessEvaluator
from parallel.async_eval     import AsyncEvaluator
from parallel.numpy_eval     import NumpyEvaluator, NumpySwarm
from core.swarm              import Enjambre
from core.stopcriteria       import MaxIteraciones
from storage.persistence     import guardar_resultado

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVALUADORES = {
    "v0": lambda: SequentialEvaluator(),
    "v1": lambda: ThreadingEvaluator(),
    "v2": lambda: ProcessEvaluator(),
    "v3": lambda: AsyncEvaluator(latency_mean=0.01, latency_std=0.003, seed=42),
    "v4": lambda: NumpyEvaluator(),
}


def _separador(char="─", ancho=58):
    print(char * ancho)


def _imprimir_cartera(titulo, metricas):
    _separador()
    print(f"  {titulo}")
    _separador()
    print(f"  {'Activo':<8}  {'Peso':>7}")
    print(f"  {'──────':<8}  {'────':>7}")
    for ticker, peso in metricas["pesos"].items():
        barra = "█" * int(peso * 30)
        print(f"  {ticker:<8}  {peso:>6.1%}  {barra}")
    _separador("─")
    print(f"  Retorno anual  : {metricas['retorno_anual']:>7.2%}")
    print(f"  Volatilidad    : {metricas['volatilidad']:>7.2%}")
    print(f"  Sharpe ratio   : {metricas['sharpe']:>7.4f}")
    _separador()


def _imprimir_comparacion(resultados):
    _separador("═")
    print("  Comparación de estrategias")
    _separador("═")
    print(f"  {'Evaluador':<35} {'Sharpe':>8} {'Tiempo':>8}")
    print(f"  {'─'*35} {'──────':>8} {'──────':>8}")
    mejor = max(resultados, key=lambda r: r["sharpe"])
    for r in resultados:
        marca = " ◄ mejor" if r["evaluador"] == mejor["evaluador"] else ""
        print(f"  {r['evaluador']:<35} {r['sharpe']:>8.4f} {r['tiempo']:>7.3f}s{marca}")
    _separador("═")


def ejecutar_con_evaluador(fn, nombre_eval, num_particulas, max_iter, semilla):
    logger.info(f"Ejecutando con {nombre_eval} ...")
    if nombre_eval == "v4":
        t0    = time.perf_counter()
        res   = NumpySwarm(fn, num_particulas=num_particulas,
                           max_iter=max_iter, semilla=semilla).ejecutar()
        tiempo = time.perf_counter() - t0
    else:
        evaluador = EVALUADORES[nombre_eval]()
        t0        = time.perf_counter()
        res       = Enjambre(fn, evaluador, num_particulas=num_particulas,
                             criterio_parada=MaxIteraciones(max_iter),
                             semilla=semilla).ejecutar()
        tiempo = time.perf_counter() - t0

    m = fn.metricas(res["pos_global"])
    return {
        "evaluador":  nombre_eval.upper(),
        "sharpe":     m["sharpe"],
        "retorno":    m["retorno_anual"],
        "volatilidad":m["volatilidad"],
        "pesos":      m["pesos"],
        "tiempo":     tiempo,
        "historial":  res["historial_fitness"],
        "pos_global": res["pos_global"],
        "fitness":    res["fitness_global"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="PSO — Optimización de cartera")
    parser.add_argument("--evaluador",  type=str, default="all",
                        choices=["all","v0","v1","v2","v3","v4"])
    parser.add_argument("--particulas", type=int,  default=50)
    parser.add_argument("--iter",       type=int,  default=500)
    parser.add_argument("--semilla",    type=int,  default=42)
    parser.add_argument("--guardar",    action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fn   = PortfolioSharpe()

    print()
    _separador("═")
    print("  PSO — Optimización de cartera de inversión")
    _separador("═")
    print(f"  Activos      : {', '.join(fn.asset_names)}")
    print(f"  Partículas   : {args.particulas}")
    print(f"  Iteraciones  : {args.iter}")
    print(f"  Semilla      : {args.semilla}")
    print(f"  Tasa libre   : {fn.risk_free:.1%}")
    _separador("═")
    print()

    ref = fn.metricas(np.ones(fn.dim) / fn.dim)
    _imprimir_cartera("Referencia — pesos iguales (1/N)", ref)
    print()

    nombres    = list(EVALUADORES.keys()) if args.evaluador == "all" else [args.evaluador]
    resultados = []

    for nombre in nombres:
        r = ejecutar_con_evaluador(fn, nombre, args.particulas, args.iter, args.semilla)
        resultados.append(r)

        if args.evaluador != "all":
            _imprimir_cartera(f"Cartera óptima — {nombre.upper()}", {
                "pesos": r["pesos"], "retorno_anual": r["retorno"],
                "volatilidad": r["volatilidad"], "sharpe": r["sharpe"],
            })

        if args.guardar:
            guardar_resultado({
                "funcion_objetivo": repr(fn), "evaluador": nombre.upper(),
                "num_particulas": args.particulas, "w": 0.7, "c1": 1.5, "c2": 1.5,
                "semilla": args.semilla, "fitness_global": r["fitness"],
                "pos_global": r["pos_global"].tolist(),
                "historial_fitness": r["historial"],
                "num_iteraciones": args.iter, "tiempo_total": r["tiempo"],
                "tiempo_evaluacion": 0.0, "tiempo_actualizacion": 0.0, "overhead": 0.0,
            })

    print()
    if len(resultados) > 1:
        _imprimir_comparacion(resultados)
        mejor = max(resultados, key=lambda r: r["sharpe"])
        print()
        _imprimir_cartera(f"Mejor cartera encontrada — {mejor['evaluador']}", {
            "pesos": mejor["pesos"], "retorno_anual": mejor["retorno"],
            "volatilidad": mejor["volatilidad"], "sharpe": mejor["sharpe"],
        })

    mejor_sharpe = max(r["sharpe"] for r in resultados)
    mejora = (mejor_sharpe - ref["sharpe"]) / ref["sharpe"] * 100
    print()
    print(f"  Sharpe referencia (1/N) : {ref['sharpe']:.4f}")
    print(f"  Mejor Sharpe PSO        : {mejor_sharpe:.4f}")
    print(f"  Mejora                  : +{mejora:.1f}%")
    print()

    if args.guardar:
        print("  Resultados guardados en results/")
    print()


if __name__ == "__main__":
    main()