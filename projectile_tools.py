import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


@dataclasses.dataclass(kw_only=True, frozen=True)
class MotionData:
    x: list[float]
    y: list[float]
    t: list[float]
    t_f: float
    peak: tuple[float, float]


@dataclasses.dataclass(kw_only=True, frozen=True)
class Projectile:
    name: str
    x_0: float
    y_0: float
    v_x0: float
    v_y0: float
    a_x: float
    a_y: float

    def position_at_time(self, t: float) -> tuple[float, float]:
        x_f = self.x_0 + self.v_x0 * t + 0.5 * self.a_x * t ** 2
        y_f = self.y_0 + self.v_y0 * t + 0.5 * self.a_y * t ** 2
        return x_f, y_f

    def get_projectile_motion_steps(self, t_0: float, t_f: float, t_step: float, min_height: float) -> MotionData:
        position_data = []
        time_data = np.arange(t_0, t_f + t_step, t_step)
        for t in time_data:
            x, y = self.position_at_time(float(t))
            position_data.append((x, max(y, min_height)))

        # Find the peak coordinates
        _, peak = max(enumerate(position_data), key=lambda item: item[1][1])

        return MotionData(
            x=[x for x, _ in position_data],
            y=[y for _, y in position_data],
            t=time_data.tolist(),
            t_f=float(t_f),
            peak=(peak[0], peak[1]),
        )


def setup_projectile_plot(ax, projectiles: list[Projectile], motion_data: list[MotionData]):
    # Find min/max bounds for plot
    max_x = max(max(data.x) for data in motion_data)
    min_x = min(min(data.x) for data in motion_data)
    max_y = max(max(data.y) for data in motion_data)
    min_y = min(min(data.y) for data in motion_data)

    ax.set_xlim(min_x - max_x * 0.1, max_x * 1.1)
    ax.set_ylim(min_y - max_y * 0.1, max_y * 1.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Motion of {projectiles[0].name if len(projectiles) == 1 else 'Projectiles'}")
    ax.grid(True)


def plot_projectile_motion(projectiles: list[Projectile], motion_data: list[MotionData], colors: list[str],
                           plot_steps: bool = False, ax=None):
    # Set up the plot (or use the provided one)
    if ax is None:
        fig, ax = plt.subplots()

    setup_projectile_plot(ax, projectiles, motion_data)
    # Trajectory
    for i, projectile in enumerate(projectiles):
        ax.plot(motion_data[i].x, motion_data[i].y, color='black')
        if plot_steps:
            plt.scatter(motion_data[i].x, motion_data[i].y, color='black', s=3)

    # Landmarks
    for i, projectile in enumerate(projectiles):
        ax.scatter(motion_data[i].x[0], motion_data[i].y[0], color=colors[i], marker='o',
                   label=f"{projectile.name} Initial Position")
        ax.scatter(motion_data[i].x[-1], motion_data[i].y[-1], color=colors[i], marker='x',
                   label=f"{projectile.name} Final Position")
        ax.scatter(motion_data[i].peak[0], motion_data[i].peak[1], color=colors[i], marker='^',
                   label=f"{projectile.name} Peak Height")

    ax.legend()


def animate_projectile_motion(projectiles: list[Projectile], motion_data: list[MotionData],
                              colors: list[str], fps: int = 30, duration_s: float | None = None, ax=None):
    # Set up the plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    setup_projectile_plot(ax, projectiles, motion_data)

    points = []  # moving points
    traces = []  # path lines

    # Pre-create plot elements
    for i, data in enumerate(motion_data):
        point = ax.scatter([], [], color=colors[i], s=30, label=projectiles[i].name)
        trace, = ax.plot([], [], linestyle='--', color=colors[i], alpha=0.5)
        points.append(point)
        traces.append(trace)

    ax.legend()

    # Get total frame count
    if duration_s is None:
        duration_s = max(md.t[-1] for md in motion_data)
    total_frames = int(duration_s * fps)

    def init():
        # This gets called when the animation starts
        for point, trace in zip(points, traces):
            # Clear the plot for each projectile
            point.set_offsets(np.empty((0, 2)))
            trace.set_data([], [])
        return points + traces

    def update(frame):
        time = frame / fps
        for i, md in enumerate(motion_data):
            # Get the projectile motion data
            t_arr = np.array(md.t)
            x_arr = np.array(md.x)
            y_arr = np.array(md.y)

            # Get all steps up to the current frame time
            mask = t_arr <= time
            if mask.any():
                # Update each point/line to the current position
                points[i].set_offsets([x_arr[mask][-1], y_arr[mask][-1]])
                traces[i].set_data(x_arr[mask], y_arr[mask])
        return points + traces

    # Create the animation (update gets called each frame, init gets called on start, interval makes it match fps)
    # (This is stored in a variable to keep it from being garbage collected)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames,
                                interval=1000 / fps, blit=False)

    # Return the animation to keep it alive
    return anim
