# viz/swarm_plot.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from objectives.functions import ObjectiveFunction


def animate_swarm_2d(
    objective_fn: ObjectiveFunction,
    position_history: list[np.ndarray],
    gbest_history: list[np.ndarray],
    output_path: str = None,
    interval: int = 100,
) -> None:
    """
    Genera una animación 2D de la evolución del enjambre.

    Parameters
    ----------
    objective_fn      : función objetivo (para dibujar el contorno)
    position_history  : lista de arrays (n_particles, 2), uno por iteración
    gbest_history     : lista de arrays (2,), posición gbest por iteración
    output_path       : si se indica, guarda como GIF o MP4
    interval          : milisegundos entre frames
    """
    lb = objective_fn.lower_bounds
    ub = objective_fn.upper_bounds

    # Malla para el contorno
    resolution = 100
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[objective_fn(np.array([xi, yi])) for xi in x] for yi in y])

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
    plt.colorbar(contour, ax=ax)

    scatter = ax.scatter([], [], c="white", s=20, zorder=3, label="partículas")
    gbest_dot = ax.scatter([], [], c="red", s=80, zorder=4, marker="*", label="gbest")
    title = ax.set_title("Iteración 0")
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.legend(loc="upper right")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    def update(frame):
        positions = position_history[frame]
        gbest = gbest_history[frame]
        scatter.set_offsets(positions)
        gbest_dot.set_offsets([gbest])
        title.set_text(f"Iteración {frame}")
        return scatter, gbest_dot, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(position_history),
        interval=interval, blit=True
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_path.endswith(".gif"):
            anim.save(output_path, writer="pillow", fps=10)
        else:
            anim.save(output_path, writer="ffmpeg", fps=10)

    plt.show()