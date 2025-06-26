import dataclasses

UNIVERSAL_GRAVITY = 6.6743e-11


@dataclasses.dataclass(kw_only=True, frozen=True)
class Projectile:
    name: str = "Default Projectile"
    mass: float = 0.0
    x_0: float = 0.0
    y_0: float = 0.0
    v_x0: float = 0.0
    v_y0: float = 0.0
    theta: float = 0.0  # rad

    thrust: float = 0.0
    thrust_start: float = 0.0  # sec
    thrust_end: float = float("inf")  # sec


@dataclasses.dataclass(kw_only=True, frozen=True)
class Environment:
    planetary_mass: float = 0.0
    planetary_radius: float = 0.0
    planetary_rotation_rate: float = 0.0  # rad/s
    min_height: float = 0.0

    def get_gravitational_acceleration(self, height: float) -> float:
        r = self.planetary_radius + height
        return (UNIVERSAL_GRAVITY * self.planetary_mass) / (r ** 2)
