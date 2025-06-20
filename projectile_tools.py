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


def setup_projectile_plot(ax, projectile: Projectile, motion_data: MotionData):
    max_x, max_y = max(motion_data.x), max(motion_data.y)
    min_x, min_y = min(motion_data.x), min(motion_data.y)
    ax.set_xlim(min_x - max_x * 0.1, max_x * 1.1)
    ax.set_ylim(min_y - max_y * 0.1, max_y * 1.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Motion of {projectile.name}")
    ax.grid(True)

    # Static elements
    ax.scatter(projectile.x_0, projectile.y_0, color='red', label="Initial Position")
    ax.scatter(motion_data.x[-1], motion_data.y[-1], color='blue', label="Final Position")
    ax.scatter(*motion_data.peak, color='green', marker='x', label="Peak Height")
    ax.legend()


def plot_projectile_motion(projectile: Projectile, motion_data: MotionData):
    fig, ax = plt.subplots()
    setup_projectile_plot(ax, projectile, motion_data)
    ax.plot(motion_data.x, motion_data.y, color='black')
    plt.show()


def animate_projectile_motion(projectile: Projectile, motion_data: MotionData, fps: int = 30):
    fig, ax = plt.subplots()
    setup_projectile_plot(ax, projectile, motion_data)

    # Moving point
    point = ax.scatter(motion_data.x[0], motion_data.y[0], c='k', s=5, label='Current Position')
    # Trajectory line
    trace, = ax.plot([], [], 'k--', linewidth=1, alpha=0.5)

    # Total animation frames based on desired fps and simulated time
    total_time = motion_data.t[-1] - motion_data.t[0]  # in seconds
    total_frames = int(total_time * fps)
    interval_ms = int(1000 / fps)

    # Evenly spaced indices for animation frames
    indices = np.linspace(0, len(motion_data.t) - 1, total_frames, dtype=int)

    def update(i: int):
        idx = indices[i]
        x = motion_data.x[:idx + 1]
        y = motion_data.y[:idx + 1]

        point.set_offsets(np.column_stack((x[-1:], y[-1:])))
        trace.set_data(x, y)
        return point, trace

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(indices),
        interval=interval_ms,
        blit=True,
        repeat=False
    )

    plt.show()
