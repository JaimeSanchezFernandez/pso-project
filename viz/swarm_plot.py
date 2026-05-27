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


def animate_swarm_3d(
    objective_fn: ObjectiveFunction,
    position_history: list[np.ndarray],
    gbest_history: list[np.ndarray],
    output_path: str = None,
    interval: int = 100,
    rotate: bool = True,
) -> None:
    """
    Genera una animación 3D de la evolución del enjambre sobre la
    superficie de la función objetivo (solo para dim=2).

    Las partículas se dibujan como puntos sobre la superficie 3D,
    a una altura igual a su fitness. El gbest se marca con una estrella roja.

    Parameters
    ----------
    objective_fn      : función objetivo (para dibujar la superficie)
    position_history  : lista de arrays (n_particles, 2), uno por iteración
    gbest_history     : lista de arrays (2,), posición gbest por iteración
    output_path       : si se indica, guarda como GIF o MP4
    interval          : milisegundos entre frames
    rotate            : si True, la cámara rota lentamente durante la animación
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra el proyector 3D)

    lb = objective_fn.lower_bounds
    ub = objective_fn.upper_bounds

    # Malla para la superficie
    resolution = 80
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[objective_fn(np.array([xi, yi])) for xi in x] for yi in y])

    # Offset para elevar las partículas por encima de la superficie
    z_offset = (Z.max() - Z.min()) * 0.04

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Superficie MUY transparente para ver las partículas a través de ella
    ax.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.25,
        linewidth=0, antialiased=True, zorder=1,
    )

    # Puntos del enjambre (cian, grandes) y gbest (estrella roja, enorme)
    scatter = ax.scatter([], [], [], c="cyan", s=45,
                         edgecolors="black", linewidths=0.6,
                         depthshade=False, zorder=10)
    gbest_dot = ax.scatter([], [], [], c="red", s=250,
                          marker="*", edgecolors="black", linewidths=0.8,
                          depthshade=False, zorder=20)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    title = ax.set_title("Iteración 0")

    def _fitness_de(posiciones):
        """Calcula el fitness de cada partícula para situarla en altura Z."""
        return np.array([objective_fn(p) for p in posiciones])

    def update(frame):
        posiciones = position_history[frame]
        gbest = gbest_history[frame]

        z_part = _fitness_de(posiciones) + z_offset
        scatter._offsets3d = (
            posiciones[:, 0], posiciones[:, 1], z_part,
        )

        z_gbest = objective_fn(gbest) + z_offset
        gbest_dot._offsets3d = (
            np.array([gbest[0]]),
            np.array([gbest[1]]),
            np.array([z_gbest]),
        )

        title.set_text(f"Iteración {frame}")

        if rotate:
            ax.view_init(elev=35, azim=frame * 1.5)

        return scatter, gbest_dot, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(position_history),
        interval=interval, blit=False,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_path.endswith(".gif"):
            anim.save(output_path, writer="pillow", fps=10)
        else:
            anim.save(output_path, writer="ffmpeg", fps=10)

    plt.show()