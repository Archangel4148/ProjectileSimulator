from matplotlib import pyplot as plt

from projectile_tools import Projectile, plot_projectile_motion, animate_projectile_motion

projectiles = [
    Projectile(
        name="Ball",
        x_0=0,
        y_0=0,
        v_x0=20,
        v_y0=5,
        a_x=0,
        a_y=-9.8,
    ),
    Projectile(
        name="Wrench",
        x_0=0,
        y_0=0,
        v_x0=20,
        v_y0=3,
        a_x=0,
        a_y=-9.8,
    ),
    Projectile(
        name="Josh",
        x_0=20,
        y_0=1,
        v_x0=-40,
        v_y0=1,
        a_x=0,
        a_y=-9.8,
    ),
]
colors = ["red", "blue", "green"]

# Time constants
simulation_steps = 200
simulation_time = 2

# Simulate projectile motion and store data
all_position_data = [
    projectile.get_projectile_motion_steps(0, simulation_time, simulation_time / simulation_steps, 0) for projectile in
    projectiles
]

# Display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_projectile_motion(projectiles, all_position_data, colors, plot_steps=False, ax=ax1)
anim = animate_projectile_motion(projectiles, all_position_data, colors, 30, ax=ax2)
plt.show()
