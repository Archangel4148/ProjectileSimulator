import dataclasses

import numpy as np
from matplotlib import pyplot as plt, animation
from scipy.integrate import solve_ivp

from constants import Projectile, Environment


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

    def simulate(self, duration: float, dt: float) -> SimulationResult:
        p, env = self.projectile, self.environment

        t_eval = np.arange(0, duration + dt, dt)

        result = solve_ivp(
            fun=lambda t, y: motion_equations(t, y, p, env),
            t_span=(0, duration),
            y0=[p.x_0, p.y_0, p.v_x0, p.v_y0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        x = result.y[0]
        y = result.y[1]

        # Clamp to min height and stop where it hits the ground
        ground_idx = np.argmax(y < env.min_height)
        if ground_idx > 0:
            x = x[:ground_idx]
            y = y[:ground_idx]
            t_eval = t_eval[:ground_idx]

        peak_idx = np.argmax(y)

        return SimulationResult(
            x=x,
            y=y,
            t=t_eval,
            peak=(x[peak_idx], y[peak_idx])
        )


def motion_equations(t, state, projectile: Projectile, env: Environment):
    x, y, vx, vy = state

    # Gravity only for now
    ax = 0
    ay = -env.grav_acc

    return [vx, vy, ax, ay]


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
