# viz/convergence.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_convergence(
    fitness_histories: dict[str, list[float]],
    title: str = "Curva de convergencia",
    output_path: str = None,
    log_scale: bool = True,
) -> None:
    """
    Dibuja las curvas de convergencia de varias estrategias en la misma figura.

    Parameters
    ----------
    fitness_histories : dict {nombre_estrategia: lista de gbest_fit por iteración}
    title             : título del gráfico
    output_path       : si se indica, guarda la figura en disco
    log_scale         : usar escala logarítmica en el eje Y (recomendado)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, history in fitness_histories.items():
        ax.plot(history, label=label, linewidth=1.8)

    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mejor fitness (gbest)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()


def plot_speedup(
    elapsed_times: dict[str, float],
    title: str = "Speedup vs V0 (secuencial)",
    output_path: str = None,
) -> None:
    """
    Dibuja un gráfico de barras con el speedup de cada estrategia
    respecto al baseline secuencial (V0).

    Parameters
    ----------
    elapsed_times : dict {nombre_estrategia: tiempo_en_segundos}
    title         : título del gráfico
    output_path   : si se indica, guarda la figura en disco
    """
    baseline = elapsed_times.get("V0_Sequential")
    if baseline is None:
        baseline = list(elapsed_times.values())[0]

    labels = list(elapsed_times.keys())
    speedups = [baseline / t for t in elapsed_times.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, speedups, color=["steelblue", "darkorange", "seagreen"])
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="baseline (1x)")
    ax.set_ylabel("Speedup (x veces más rápido que V0)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{speedup:.2f}x",
            ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()