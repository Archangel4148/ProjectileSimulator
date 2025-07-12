import dataclasses
import math
from typing import Callable

UNIVERSAL_GRAVITY = 6.6743e-11


@dataclasses.dataclass(kw_only=True, frozen=True)
class Projectile:
    name: str = "Default Projectile"
    mass: float = 0.0
    moment_of_inertia: float = math.inf  # Don't allow rotational acceleration by default
    x_0: float = 0.0
    y_0: float = 0.0
    v_x0: float = 0.0
    v_y0: float = 0.0
    theta: float = 0.0  # rad
    omega: float = 0.0  # rad/s

    thrust: float = 0.0
    thrust_start: float = 0.0  # sec
    thrust_end: float = float("inf")  # sec

    # Aerodynamics
    drag_coefficient: float = 0.0
    r_drag: float = 0.0  # radius of torque from drag (m from center of mass)
    cross_sectional_area: float = 0.0  # Cross-section from the nose down (m^2)


@dataclasses.dataclass(kw_only=True, frozen=True)
class Environment:
    planetary_mass: float = 0.0
    planetary_radius: float = 0.0
    planetary_rotation_rate: float = 0.0  # rad/s
    min_height: float = 0.0
    # Model to calculate air density given altitude
    air_density_model: Callable[[float], float] = lambda h: 0

    def get_gravitational_acceleration(self, height: float) -> float:
        r = self.planetary_radius + height
        return (UNIVERSAL_GRAVITY * self.planetary_mass) / (r ** 2)


def get_earth_air_density_nasa(altitude: float) -> float:
    # Source: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    if altitude < 11000:
        # Troposphere
        air_temp = 15.04 - 0.00649 * altitude
        air_pressure = 101.29 * ((air_temp + 273.1) / 288.08) ** 5.256
    elif 11000 <= altitude < 25000:
        # Lower Stratosphere
        air_temp = -56.46
        air_pressure = 22.65 * math.exp(1.73 - 0.000157 * altitude)
    else:
        # Upper Stratosphere
        air_temp = -131.21 + 0.00299 * altitude
        air_pressure = 2.488 * ((air_temp + 273.1) / 216.6) ** -11.388

    return air_pressure / (0.2869 * (air_temp + 273.1))
