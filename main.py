import math

from matplotlib import pyplot as plt

from constants import Projectile, Environment
from simulation import ProjectileSimulation, plot_projectile_motion, animate_projectile_motion

projectiles = [
    Projectile(
        name="Rocket",
        mass=10,
        x_0=15,
        y_0=10,
        v_x0=0,
        v_y0=0,
        thrust=300,
        theta=math.pi / 4,
        thrust_start=1,
        thrust_end=3,
    )
]
earth = Environment(
    planetary_mass=5.97219e24,
    planetary_radius=6.3781e6,
    min_height=0,
)

colors = ["red", "blue", "green"]

# Time constants
simulation_steps = 200
simulation_time = 6
animation_fps = 30
plot_steps = False
show_animation = True

# Simulate projectile motion and store data
simulation_data = []
for projectile in projectiles:
    simulation = ProjectileSimulation(projectile, earth)
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
