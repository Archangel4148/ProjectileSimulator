import dataclasses


@dataclasses.dataclass(kw_only=True, frozen=True)
class Projectile:
    name: str
    x_0: float
    y_0: float
    v_x0: float
    v_y0: float


@dataclasses.dataclass(kw_only=True, frozen=True)
class Environment:
    grav_acc: float
    min_height: float
