import dataclasses

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from constants import Projectile, Environment


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimulationResult:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    theta: np.ndarray
    peak: tuple[float, float]


class ProjectileSimulation:
    def __init__(self, projectile: Projectile, environment: Environment):
        self.projectile = projectile
        self.environment = environment

    def simulate(self, duration: float, dt: float) -> SimulationResult:
        p, env = self.projectile, self.environment

        t_eval = np.arange(0, duration, dt)
        t_eval = np.append(t_eval, duration)

        result = solve_ivp(
            fun=lambda t, y: self.motion_equations(t, y),
            t_span=(0, duration),
            y0=[p.x_0, p.y_0, p.v_x0, p.v_y0, p.theta, 0.0],
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
        theta = result.y[4]

        return SimulationResult(
            x=x,
            y=y,
            t=t_eval,
            theta=theta,
            peak=(x[peak_idx], y[peak_idx])
        )

    def calculate_acceleration(self, t: float, x: float, y: float, vx: float, vy: float, theta: float) -> tuple[
        float, float]:
        p, env = self.projectile, self.environment

        # === Balancing Forces ===
        sum_fx, sum_fy = 0.0, 0.0

        # Gravity
        sum_fy -= p.mass * env.get_gravitational_acceleration(y)

        # Thrust (Only during thrust period)
        if p.thrust_start <= t <= p.thrust_end:
            sum_fx += p.thrust * np.cos(theta)
            sum_fy += p.thrust * np.sin(theta)

        # Solve for acceleration
        return sum_fx / p.mass, sum_fy / p.mass

    def motion_equations(self, t, state):
        x, y, vx, vy, theta, omega = state

        # Get acceleration (from forces)
        ax, ay = self.calculate_acceleration(t, x, y, vx, vy, theta)

        # Angular acceleration
        alpha = 0.0

        return [vx, vy, ax, ay, omega, alpha]


# ===================== PHYSICS =====================


# ===================== PLOTTING/VISUALIZATION =====================
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

    orientations = []  # Orientation lines

    # Pausing after each animation loop
    pause_duration_s = 1.0
    pause_frames = int(pause_duration_s * fps)

    # Pre-create plot elements
    for i, data in enumerate(simulation_data):
        point = ax.scatter([], [], color=colors[i], s=30, label=projectiles[i].name)
        trace, = ax.plot([], [], linestyle='--', color=colors[i], alpha=0.5)
        points.append(point)
        traces.append(trace)
        # Orientation lines
        line = Line2D([], [], color=colors[i], linestyle='-', linewidth=1)
        ax.add_line(line)
        orientations.append(line)

    ax.legend()

    # Get total frame count
    if duration_s is None:
        duration_s = max([float(md.t[-1]) for md in simulation_data])
    total_frames = int(duration_s * fps) + pause_frames

    def init():
        # This gets called when the animation starts
        empty_array = np.empty((0, 2))
        for point, trace in zip(points, traces):
            # Clear the plot for each projectile
            point.set_offsets(empty_array)
            trace.set_data([], [])
        return points + traces

    def update(frame):
        true_frame = min(frame, len(simulation_data[0].t) - 1)
        time = true_frame / fps
        for i, md in enumerate(simulation_data):
            # Get all steps up to the current frame time
            mask = md.t <= time
            if mask.any():
                # Update each point/line to the current position
                points[i].set_offsets([md.x[mask][-1], md.y[mask][-1]])
                traces[i].set_data(md.x[mask], md.y[mask])
                # Update orientation line
                theta = simulation_data[i].theta[mask][-1]
                length = 1.0
                x_pos, y_pos = md.x[mask][-1], md.y[mask][-1]
                x_end = x_pos + length * np.cos(theta)
                y_end = y_pos + length * np.sin(theta)
                orientations[i].set_data([x_pos, x_end], [y_pos, y_end])

        return points + traces + orientations

    # Create the animation (update gets called each frame, init gets called on start, interval makes it match fps)
    # (This is stored in a variable to keep it from being garbage collected)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames,
                                   interval=1000 / fps, blit=True)

    # Return the animation to keep it alive
    return anim
