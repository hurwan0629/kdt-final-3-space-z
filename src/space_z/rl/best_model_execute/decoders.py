"""배포 모델의 이산 행동과 연속 회전을 게임 제어 입력으로 바꾼다."""
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from space_z.config.progression_definitions import STAT_UPGRADES
from space_z.game.actions import MOVE_VECTORS
from .action_decoder import ActionDecoder
from .player_control import PlayerControl


class FactorizedActionDecoder(ActionDecoder):
    """[이동, 회전, 발사, 강화, 진화]의 독립 이산 선택을 해석한다."""

    rotations = (-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1)

    def __init__(self):
        self._action_space = MultiDiscrete([9, 9, 2, 6, 5])
        self._stat_keys = tuple(stat.key for stat in STAT_UPGRADES)
        if len(self._stat_keys) != 5:
            raise ValueError('This model requires five stat upgrades')

    @property
    def action_space(self):
        return self._action_space

    def decode(self, action):
        values = np.asarray(action)
        if not self._action_space.contains(values):
            raise ValueError(f'Invalid factorized action: {values}')
        movement, rotation, fire, stat, evolution = map(int, values)
        move_x, move_y = MOVE_VECTORS[movement]
        return PlayerControl(
            move_x=move_x, move_y=move_y, rotation=self.rotations[rotation], want_fire=bool(fire),
            stat_upgrade_key=self._stat_keys[stat - 1] if stat else None,
            evolution_choice=evolution - 1 if evolution else None,
        )


class RotationContinuousActionDecoder(ActionDecoder):
    """회전은 -1~1, 나머지는 범주 선택으로 해석하며 진화 번호를 int로 전달한다."""

    def __init__(self):
        self._action_space = Box(
            low=np.array([0, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([8, 1, 1, 5, 4], dtype=np.float32),
        )
        self._stat_keys = tuple(stat.key for stat in STAT_UPGRADES)
        if len(self._stat_keys) != 5:
            raise ValueError('This model requires five stat upgrades')

    @property
    def action_space(self):
        return self._action_space

    def decode(self, action):
        values = np.asarray(action)
        if values.shape != (5,) or values.dtype.kind not in 'iuf':
            raise ValueError('Action must contain five real numbers')
        if not np.isfinite(values).all():
            raise ValueError('Action must be finite')
        if (values < self._action_space.low).any() or (values > self._action_space.high).any():
            raise ValueError('Action is outside the declared bounds')
        choices = values[[0, 2, 3, 4]]
        if not np.equal(choices, np.floor(choices)).all():
            raise ValueError('Movement, fire and growth choices must be integers')
        movement, fire, stat, evolution = map(int, choices)
        move_x, move_y = MOVE_VECTORS[movement]
        return PlayerControl(
            move_x=move_x, move_y=move_y, rotation=float(values[1]), want_fire=bool(fire),
            stat_upgrade_key=self._stat_keys[stat - 1] if stat else None,
            evolution_choice=evolution - 1 if evolution else None,
        )
