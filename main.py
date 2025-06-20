from projectile_tools import Projectile, animate_projectile_motion

projectile = Projectile(
    name="Car",
    x_0=0,
    y_0=0,
    v_x0=20,
    v_y0=5,
    a_x=0,
    a_y=-9.8,
)

# Time constants
simulation_fps = 30
simulation_time = 5

position_data = projectile.get_projectile_motion_steps(0, simulation_time, 1 / simulation_fps, 0)

animate_projectile_motion(projectile, position_data, fps=simulation_fps)
