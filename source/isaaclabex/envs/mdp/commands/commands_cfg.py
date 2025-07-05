
from dataclasses import MISSING
from collections.abc import Sequence
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .cartpole_command import CartpoleCommand

@configclass
class CartpoleCommandCfg(CommandTermCfg):

    class_type: type = CartpoleCommand

    asset_name: str = MISSING
    statistics_name: str = MISSING

    car_pos: float = 0.8
    car_swing: float = 0.5

    pole_swing: float = 0.1

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        car_pos: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        car_swing: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        pole_swing: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
