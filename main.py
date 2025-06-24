from matplotlib import pyplot as plt

from projectile_tools import Projectile, Environment
from simulation import ProjectileSimulation, plot_projectile_motion, animate_projectile_motion

projectiles = [
    # Projectile(
    #     name="Ball",
    #     x_0=0,
    #     y_0=0,
    #     v_x0=20,
    #     v_y0=5,
    #     a_x=0,
    #     a_y=-9.8,
    # ),
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
environment = Environment(
    grav_acc=9.8,
    min_height=0,
)

colors = ["red", "blue", "green"]

# Time constants
simulation_steps = 200
simulation_time = 2
animation_fps = 30
plot_steps = False
show_animation = True

# Simulate projectile motion and store data
simulation_data = []
for projectile in projectiles:
    simulation = ProjectileSimulation(projectile, environment)
    result = simulation.simulate(simulation_time, simulation_time / simulation_steps)
    simulation_data.append(result)

# Display the results
if show_animation:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
else:
    fig, ax1 = plt.subplots()
plot_projectile_motion(projectiles, simulation_data, colors, plot_steps=plot_steps, ax=ax1)
if show_animation:
    anim = animate_projectile_motion(projectiles, simulation_data, colors, animation_fps, ax=ax2)
plt.show()
