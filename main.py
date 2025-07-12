import math

from matplotlib import pyplot as plt

from constants import Projectile, Environment, get_earth_air_density_nasa
from simulation import ProjectileSimulation, plot_projectile_motion, animate_projectile_motion

projectiles = [
    Projectile(
        name="Rocket",
        mass=5,
        x_0=0,
        y_0=10,
        v_x0=0,
        v_y0=0,
        thrust=200,
        theta=math.radians(45),
        thrust_start=0.5,
        thrust_end=2.5,
        moment_of_inertia=0.25,
        r_drag=0.75,
        drag_coefficient=0.75,  # Between a cone and a cylinder
        cross_sectional_area=0.008,  # 10cm diameter
    )
]
earth = Environment(
    planetary_mass=5.97219e24,
    planetary_radius=6.3781e6,
    min_height=0,
    air_density_model=get_earth_air_density_nasa
)

colors = ["blue"]

# Plot bounds
OVERRIDE_PLOT_DIMENSIONS = False
PLOT_DIMENSIONS = {
    "x": (-10, 250),
    "y": (0, 20),
}

# Time constants
simulation_steps = 200
simulation_time = 10
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
plot_projectile_motion(projectiles, simulation_data, colors, plot_steps=plot_steps,
                       bounds_override=PLOT_DIMENSIONS if OVERRIDE_PLOT_DIMENSIONS else None, ax=ax1)
if show_animation:
    anim = animate_projectile_motion(projectiles, simulation_data, colors, animation_fps,
                                     bounds_override=PLOT_DIMENSIONS if OVERRIDE_PLOT_DIMENSIONS else None, ax=ax2)
plt.show()
