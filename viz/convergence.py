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
    """
    baseline = elapsed_times.get("V0_Sequential")
    if baseline is None:
        baseline = list(elapsed_times.values())[0]

    labels = list(elapsed_times.keys())
    speedups = [baseline / t for t in elapsed_times.values()]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["steelblue", "darkorange", "seagreen", "purple", "crimson"]
    bars = ax.bar(labels, speedups, color=colors[:len(labels)])
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
    """
    fn_names = list(results_by_strategy.keys())
    strategy_names = list(next(iter(results_by_strategy.values())).keys())
    n_fns = len(fn_names)

    fig, axes = plt.subplots(1, n_fns, figsize=(5 * n_fns, 6))
    if n_fns == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "seagreen", "purple", "crimson"]

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


def plot_portfolio(
    sharpe_histories:   dict[str, list[float]],
    metricas_por_eval:  dict[str, dict],
    referencia_sharpe:  float,
    asset_names:        list[str],
    output_path:        str | None = None,
) -> None:
    """
    Figura de dos paneles para el caso de uso de cartera:
      - Izquierda: curvas de convergencia del Sharpe ratio por evaluador.
      - Derecha:   composición de la mejor cartera encontrada (barras).
    """
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])

    # --- Panel izquierdo: convergencia del Sharpe ---
    ax1 = fig.add_subplot(gs[0])
    colors = ["steelblue", "darkorange", "seagreen", "purple", "crimson"]
    for i, (nombre, historia) in enumerate(sharpe_histories.items()):
        ax1.plot(historia, label=nombre, linewidth=1.8, color=colors[i % len(colors)])
    ax1.axhline(referencia_sharpe, color="gray", linestyle="--",
                linewidth=1.2, label=f"Ref. 1/N ({referencia_sharpe:.3f})")
    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Convergencia del Sharpe ratio")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel derecho: composición de la mejor cartera ---
    ax2 = fig.add_subplot(gs[1])
    mejor_nombre = max(metricas_por_eval, key=lambda k: metricas_por_eval[k]["sharpe"])
    mejor        = metricas_por_eval[mejor_nombre]
    pesos        = [mejor["pesos"][a] for a in asset_names]

    bars = ax2.barh(asset_names, pesos, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Peso en la cartera")
    ax2.set_title(
        f"Cartera óptima ({mejor_nombre})\n"
        f"Sharpe={mejor['sharpe']:.4f}  "
        f"Ret={mejor['retorno_anual']:.1%}  "
        f"Vol={mejor['volatilidad']:.1%}"
    )
    ax2.set_xlim(0, max(pesos) * 1.25)
    for bar, peso in zip(bars, pesos):
        if peso > 0.01:
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{peso:.1%}", va="center", fontsize=9)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.suptitle("PSO — Optimización de cartera de inversión", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()