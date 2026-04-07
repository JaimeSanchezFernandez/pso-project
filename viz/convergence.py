# viz/convergence.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path


def plot_convergence(
    fitness_histories: dict[str, list[float]],
    title: str = "Curva de convergencia",
    output_path: str | None = None,
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
    output_path: str | None = None,
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


def plot_convergence_all_functions(
    results: dict[str, dict[str, list[float]]],
    title: str = "Convergencia por función",
    output_path: str | None = None,
    log_scale: bool = True,
) -> None:
    """
    Dibuja las curvas de convergencia para las 4 funciones benchmark
    en una figura con 4 subplots (2x2).

    Parameters
    ----------
    results : dict {nombre_función: {nombre_estrategia: fitness_history}}
    title   : título general de la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (fn_name, histories) in enumerate(results.items()):
        ax = axes[idx]
        for label, history in histories.items():
            ax.plot(history, label=label, linewidth=1.8)
        ax.set_title(fn_name)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("gbest fitness")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()


def plot_boxplot(
    results_by_strategy: dict[str, dict[str, list[float]]],
    title: str = "Distribución de fitness final",
    output_path: str | None = None,
) -> None:
    """
    Dibuja boxplots del fitness final para cada función y estrategia,
    ejecutando múltiples seeds para mostrar la distribución.

    Parameters
    ----------
    results_by_strategy : dict {nombre_función: {nombre_estrategia: [gbest_fit por seed]}}
    title               : título del gráfico
    output_path         : si se indica, guarda la figura en disco
    """
    fn_names = list(results_by_strategy.keys())
    strategy_names = list(next(iter(results_by_strategy.values())).keys())
    n_fns = len(fn_names)
    n_strategies = len(strategy_names)

    fig, axes = plt.subplots(1, n_fns, figsize=(5 * n_fns, 6))
    if n_fns == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "seagreen"]

    for idx, fn_name in enumerate(fn_names):
        ax = axes[idx]
        data = [results_by_strategy[fn_name][s] for s in strategy_names]
        bp = ax.boxplot(data, patch_artist=True, labels=strategy_names)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(fn_name)
        ax.set_ylabel("gbest fitness final")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()