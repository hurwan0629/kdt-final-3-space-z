"""이산 행동 번호를 이동·회전·발사 및 선택적인 성장 요청으로 해석한다."""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import TYPE_CHECKING

from gymnasium.spaces import Discrete

from space_z.config.progression_definitions import FAMILIES, SHIPS, STAT_UPGRADES
from space_z.game.actions import MOVE_VECTORS, N_MOVES, N_ROTS, ROT_DIR

from space_z.rl.actions.action_decoder import ActionDecoder
from space_z.rl.actions.player_control import PlayerControl

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DiscreteActionDecoder(ActionDecoder):
    """하나의 정수에 결합한 행동을 공통 게임 제어 입력으로 바꾼다.

    번호 조합 순서:
        이동 → 회전 → 발사 → 스탯 강화 → 전직.
        action = ((((move * 3 + rotation) * 2 + fire) * stat_count + stat)
                  * evolution_count + evolution)

    이동 번호는 기존 게임의 정지·8방향 순서, 회전 번호는 유지·반시계·시계 순서다.
    fire는 0=발사 안 함, 1=발사 요청이다.
    스탯·전직 번호는 0=선택 안 함이고, 1부터 각 선택지의 순서에 대응한다.
    action=0은 이동·회전·발사·강화·전직을 모두 요청하지 않는다.
    """

    def __init__(
        self,
        *,
        enable_stat_upgrades: bool = False,
        enable_evolution: bool = False,
    ) -> None:
        """성장 선택의 포함 여부를 정하고 고정된 이산 행동 공간을 만든다.

        인자:
            enable_stat_upgrades: True이면 선택 안 함과 STAT_UPGRADES의 스탯 선택을 포함한다.
            enable_evolution: True이면 선택 안 함과 전직·숙련 선택지 번호를 포함한다.
                선택지 수는 콘텐츠의 계열·전직 후보 중 최대 개수로 고정한다.

        반환:
            None. 기본 54개 행동이며 현재 콘텐츠에서 스탯만 포함하면 324개,
            전직만 포함하면 270개, 둘 다 포함하면 1,620개다.

        예외:
            TypeError: 포함 여부 인자가 bool이 아닌 경우.
        """
        # 1) 옵션을 검증하고 스탯 선택 번호와 키의 대응 관계를 준비한다.
        for name, enabled in (
            ("enable_stat_upgrades", enable_stat_upgrades),
            ("enable_evolution", enable_evolution),
        ):
            if not isinstance(enabled, bool):
                raise TypeError(f"{name}는 bool이어야 합니다.")
        
        self._stat_upgrade_keys = tuple(card.key for card in STAT_UPGRADES) if enable_stat_upgrades else ()
        self._stat_choice_count = len(self._stat_upgrade_keys) + 1  # 선택 안 함을 포함한 스탯 경우의 수

        # 2) 현재 기체에 따라 공간 크기가 바뀌지 않도록 콘텐츠의 최대 전직 선택지 수를 구한다.
        self._evolution_choice_count = 1  # 전직을 끄면 선택 안 함만 존재
        if enable_evolution:
            branch_counts = Counter(
                (ship.parent, ship.level) for ship in SHIPS.values() if ship.level in (20, 30)
            )
            max_choices = max(len(FAMILIES), max(branch_counts.values(), default=0), 1)
            self._evolution_choice_count = max_choices + 1  # 전직·숙련 선택과 선택 안 함

        # 3) 모든 조합을 하나의 정수 번호로 표현하는 Gymnasium 공간을 만든다.
        self._action_space = Discrete(
            N_MOVES * N_ROTS * 2 * self._stat_choice_count * self._evolution_choice_count
        )

    @property
    def action_space(self) -> Discrete:
        """현재 옵션으로 허용하는 행동 번호 범위 0~n-1을 반환한다."""
        return self._action_space

    def decode(self, action: int | NDArray[np.float32]) -> PlayerControl:
        """정수 행동 번호를 검증하고 이동·회전·발사·성장 선택으로 분리한다.

        인자:
            action: action_space 범위의 정수. NumPy 정수도 허용하고 bool·실수·배열은 거부한다.

        반환:
            정지 또는 정규화된 8방향 이동, -1·0·1 회전 입력, 발사 여부와 성장 선택을 담은 PlayerControl.
            스탯 번호는 STAT_UPGRADES의 키로, 전직 번호는 현재 선택지의 0부터 시작하는 인덱스로 바꾼다.
            비활성화된 성장 선택은 항상 None이다.
            현재 포인트·스탯 한도·전직 가능 여부는 조회하지 않으며 적용 시 코어가 검사한다.

        예외:
            TypeError: 정수 이외의 입력 또는 bool을 받은 경우.
            ValueError: 행동 번호가 action_space 범위를 벗어난 경우.
        """
        # 1) 잘못된 입력을 정수로 강제 변환해 다른 행동으로 해석하지 않도록 먼저 검증한다.
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("이산 행동은 bool을 제외한 정수여야 합니다.")
        if not 0 <= action < self._action_space.n:
            raise ValueError(f"행동 번호는 0~{self._action_space.n - 1} 범위여야 합니다.")

        # 2) 조합할 때의 역순으로 나누어 각 선택 번호를 꺼낸다.
        remaining_action, evolution = divmod(int(action), self._evolution_choice_count)
        remaining_action, stat = divmod(remaining_action, self._stat_choice_count)
        remaining_action, fire = divmod(remaining_action, 2)
        move, rotation = divmod(remaining_action, N_ROTS)

        # 3) 게임 상태를 바꾸지 않고 번호를 공통 제어값으로 변환한다.
        move_x, move_y = MOVE_VECTORS[move]
        return PlayerControl(
            move_x=move_x,
            move_y=move_y,
            rotation=ROT_DIR[rotation],
            want_fire=bool(fire),
            stat_upgrade_key=self._stat_upgrade_keys[stat - 1] if stat else None,
            evolution_choice=evolution - 1 if evolution else None,
        )
