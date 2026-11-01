"""벡터·픽셀 관찰에서 공유하는 성장 상태 인코딩."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from space_z.config import ObsConfig
from space_z.config.progression_definitions import SHIPS, STAT_UPGRADES, STAT_MAX_LEVEL

if TYPE_CHECKING:
    from space_z.game.game_core import GameCore

SHIP_KEYS = tuple(SHIPS)  # 기체 원핫 인코딩의 고정 순서


def ship_one_hot(key: str) -> list[float]:
    """기체 키를 SHIPS 순서의 원핫 벡터로 반환한다."""
    return [float(key == ship_key) for ship_key in SHIP_KEYS]


def build_progression(game: GameCore) -> NDArray[np.float32]:
    """성장 상태를 읽어 [0, 1] 벡터로 반환한다. 게임과 선택지는 변경하지 않는다.

    순서: 기체 원핫 → 레벨·경험치·강화포인트·특수보상 수 → 스탯 단계 → 강화 가능 여부
    → 전직 슬롯별 존재 표시·기체 원핫. 무제한 증가 수치는 x/(x+1)로 정규화한다.
    전직 슬롯은 현재 열린 선택지이며 비어 있는 슬롯은 0이다.
    """
    progression = game.progression
    values = ship_one_hot(game.player.job)
    values.extend((
        progression.level / (progression.level + 1),
        progression.xp / max(1, progression.xp_to_next_level),
        progression.upgrade_points / (progression.upgrade_points + 1),
        len(progression.special_reward_levels) / (len(progression.special_reward_levels) + 1),
    ))
    values.extend(progression.stat_levels[card.key] / STAT_MAX_LEVEL for card in STAT_UPGRADES)
    values.extend(float(progression.can_upgrade_stat(card.key)) for card in STAT_UPGRADES)
    for index in range(ObsConfig.EVOLUTION_SLOTS):
        if index < len(progression.reward_choices) and not game.game_over and game.player.alive:
            values.extend((1.0, *ship_one_hot(progression.reward_choices[index].key)))
        else:
            values.extend([0.0] * (1 + len(SHIP_KEYS)))
    return np.clip(np.asarray(values, dtype=np.float32), 0, 1)
