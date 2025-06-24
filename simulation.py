import dataclasses

import numpy as np
from matplotlib import pyplot as plt, animation

from projectile_tools import Projectile, Environment


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimulationResult:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    peak: tuple[float, float]


class ProjectileSimulation:
    def __init__(self, projectile: Projectile, environment: Environment):
        self.projectile = projectile
        self.environment = environment

    def simulate(self, duration: float, time_step: float) -> SimulationResult:
        position_data = []
        time_data = np.arange(0, duration + time_step, time_step)

        for nt in time_data:
            t = float(nt)
            x = self.projectile.x_0 + self.projectile.v_x0 * t + 0.5 * self.projectile.a_x * t ** 2
            y = self.projectile.y_0 + self.projectile.v_y0 * t + 0.5 * self.projectile.a_y * t ** 2
            position_data.append((x, max(y, self.environment.min_height)))

        # Find the peak coordinates
        _, peak = max(enumerate(position_data), key=lambda item: item[1][1])

        return SimulationResult(
            x=np.array([x for x, _ in position_data]),
            y=np.array([y for _, y in position_data]),
            t=time_data,
            peak=(peak[0], peak[1]),
        )


def setup_projectile_plot(ax, projectiles: list[Projectile], simulation_data: list[SimulationResult]):
    # Find and set min/max bounds for all plotted data (with padding)
    all_x = np.concatenate([data.x for data in simulation_data])
    all_y = np.concatenate([data.y for data in simulation_data])
    x_padding = 0.1 * (all_x.max() - all_x.min())
    y_padding = 0.1 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)

    # Labels, title, and grid
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Motion of {projectiles[0].name if len(projectiles) == 1 else 'Projectiles'}")
    ax.grid(True)


def plot_projectile_motion(projectiles: list[Projectile], simulation_data: list[SimulationResult], colors: list[str],
                           plot_steps: bool = False, ax=None):
    # Set up the plot (or use the provided one)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    setup_projectile_plot(ax, projectiles, simulation_data)
    # Trajectory
    for i, projectile in enumerate(projectiles):
        ax.plot(simulation_data[i].x, simulation_data[i].y, color='black')
        if plot_steps:
            ax.scatter(simulation_data[i].x, simulation_data[i].y, color='black', s=3)

    # Landmarks
    for i, projectile in enumerate(projectiles):
        ax.scatter(simulation_data[i].x[0], simulation_data[i].y[0], color=colors[i], marker='o',
                   label=f"{projectile.name} Initial Position", zorder=3)
        ax.scatter(simulation_data[i].x[-1], simulation_data[i].y[-1], color=colors[i], marker='x',
                   label=f"{projectile.name} Final Position", zorder=3)
        ax.scatter(simulation_data[i].peak[0], simulation_data[i].peak[1], color=colors[i], marker='^',
                   label=f"{projectile.name} Peak Height", zorder=3)

    ax.legend()


def animate_projectile_motion(projectiles: list[Projectile], simulation_data: list[SimulationResult],
                              colors: list[str], fps: int = 30, duration_s: float | None = None, ax=None):
    # Set up the plot
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

    setup_projectile_plot(ax, projectiles, simulation_data)

    points = []  # moving points
    traces = []  # path lines

    # Pre-create plot elements
    for i, data in enumerate(simulation_data):
        point = ax.scatter([], [], color=colors[i], s=30, label=projectiles[i].name)
        trace, = ax.plot([], [], linestyle='--', color=colors[i], alpha=0.5)
        points.append(point)
        traces.append(trace)

    ax.legend()

    # Get total frame count
    if duration_s is None:
        duration_s = max([float(md.t[-1]) for md in simulation_data])
    total_frames = int(duration_s * fps)

    def init():
        # This gets called when the animation starts
        empty_array = np.empty((0, 2))
        for point, trace in zip(points, traces):
            # Clear the plot for each projectile
            point.set_offsets(empty_array)
            trace.set_data([], [])
        return points + traces

    def update(frame):
        time = frame / fps
        for i, md in enumerate(simulation_data):
            # Get all steps up to the current frame time
            mask = md.t <= time
            if mask.any():
                # Update each point/line to the current position
                points[i].set_offsets([md.x[mask][-1], md.y[mask][-1]])
                traces[i].set_data(md.x[mask], md.y[mask])
        return points + traces

    # Create the animation (update gets called each frame, init gets called on start, interval makes it match fps)
    # (This is stored in a variable to keep it from being garbage collected)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames,
                                   interval=1000 / fps, blit=True)

    # Return the animation to keep it alive
    return anim
