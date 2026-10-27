"""정규화한 실수 배열에서 고정 세기의 8방향 이동과 연속 회전을 해석한다."""

from __future__ import annotations

from collections import Counter

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from space_z.config.progression_definitions import FAMILIES, SHIPS, STAT_UPGRADES
from space_z.game.actions import MOVE_VECTORS

from space_z.rl.actions.action_decoder import ActionDecoder
from space_z.rl.actions.player_control import PlayerControl


class ContinuousActionDecoder(ActionDecoder):
    """[이동 선택, 회전, 발사, (스탯 선택), (전직 선택)]을 해석한다.

    각 값은 [-1, 1]이다. 이동은 같은 폭의 9구간으로 나눠 정지·8방향을 고른다.
    회전은 실수 세기를 유지하고 발사는 0 초과일 때 요청한다.
    선택적인 스탯·전직도 같은 폭으로 나누며 첫 구간은 선택 안 함이다.
    """

    def __init__(self, *, enable_stat_upgrades: bool = False, enable_evolution: bool = False) -> None:
        """성장 선택 옵션에 맞춰 3~5차원의 Box 행동 공간을 만든다."""
        if type(enable_stat_upgrades) is not bool or type(enable_evolution) is not bool:
            raise TypeError("성장 선택 옵션은 bool이어야 합니다.")
        self._stat_keys = tuple(card.key for card in STAT_UPGRADES) if enable_stat_upgrades else ()
        branches = Counter((ship.parent, ship.level) for ship in SHIPS.values() if ship.level in (20, 30))
        self._evolution_slots = max(len(FAMILIES), max(branches.values(), default=0), 1) if enable_evolution else 0
        self._action_space = Box(-1.0, 1.0, (3 + int(enable_stat_upgrades) + int(enable_evolution),), np.float32)

    @property
    def action_space(self) -> Box:
        """입력 배열의 크기와 [-1, 1] 범위를 반환한다."""
        return self._action_space

    @staticmethod
    def _choice(value: float, count: int) -> int:
        """[-1, 1] 값을 같은 폭의 count개 구간 번호로 바꾼다. 양 끝도 포함한다."""
        return min(count - 1, int((value + 1) * count / 2))

    def decode(self, action: int | NDArray[np.float32]) -> PlayerControl:
        """배열 크기·실수 자료형·유한성을 검사하고 clipping 후 공통 제어값을 반환한다."""
        if not isinstance(action, np.ndarray) or not np.issubdtype(action.dtype, np.floating):
            raise TypeError("연속 행동은 실수형 NumPy 배열이어야 합니다.")
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"연속 행동은 유한한 {self.action_space.shape} 배열이어야 합니다.")
        values = np.clip(action, -1, 1)
        move_x, move_y = MOVE_VECTORS[self._choice(float(values[0]), 9)]
        offset, stat_key, evolution = 3, None, None
        if self._stat_keys:
            choice = self._choice(float(values[offset]), len(self._stat_keys) + 1)
            stat_key = self._stat_keys[choice - 1] if choice else None
            offset += 1
        if self._evolution_slots:
            choice = self._choice(float(values[offset]), self._evolution_slots + 1)
            evolution = choice - 1 if choice else None
        return PlayerControl(move_x, move_y, float(values[1]), bool(values[2] > 0), stat_key, evolution)
