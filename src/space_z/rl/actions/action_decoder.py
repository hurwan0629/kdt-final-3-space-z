"""이산·연속 행동 해석기가 따라야 할 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from space_z.rl.actions.player_control import PlayerControl

if TYPE_CHECKING:
    import numpy as np
    from gymnasium.spaces import Space
    from numpy.typing import NDArray


class ActionDecoder(ABC):
    """모델의 행동을 공통 PlayerControl로 변환하는 행동 해석기의 계약.

    이산·연속 구현체가 서로 다른 모델 출력을 받아도 같은 형태의 제어값을 반환한다.
    게임 진행·충돌·발사 대기·탄환 생성은 담당하지 않는다.
    """

    @property
    @abstractmethod
    def action_space(self) -> Space[Any]:
        """모델이 출력할 수 있는 행동의 종류·크기·값 범위를 정의한 Gymnasium 공간.

        반환:
            이산 구현체는 정수 행동 번호를 나타내는 Discrete 공간,
            연속 구현체는 실수 행동 배열을 나타내는 Box 공간.
            공간의 정의는 decode()에서 해석하는 행동 형식과 일치해야 한다.
        """
        # 구현체가 준비한 공간을 제공한다. 속성을 조회할 때 행동을 추첨하거나 게임을 진행하지 않는다.
        pass

    @abstractmethod
    def decode(self, action: int | NDArray[np.float32]) -> PlayerControl:
        """모델 출력을 검증·해석해 이동·회전·발사 요청으로 변환한다.

        인자:
            action: 이산 모드의 정수 행동 번호 또는 연속 모드의 실수 배열.

        반환:
            move_x·move_y·rotation·want_fire와 선택적인 stat_upgrade_key·evolution_choice를 갖는 PlayerControl.
            이동과 회전 입력은 -1~1 범위이고 이동 벡터 길이는 최대 1이다.
            이동은 정지 또는 고정 세기의 8방향 가속 입력, 회전은 최대 회전 속도에 곱할 비율이다.

        예외:
            TypeError: 구현체가 받을 수 없는 자료형인 경우.
            ValueError: 행동 번호·배열 크기가 잘못되거나 NaN·무한대가 포함된 경우.
        """
        # 이산 구현체는 번호를 이동·회전·발사 선택으로 분리한다.
        # 연속 구현체는 배열 크기·유한성을 확인하고 입력 범위를 제한해 제어값으로 바꾼다.
        # 대각선 입력이 더 강해지지 않도록 이동 벡터 길이를 최대 1로 제한한다.
        # 발사는 요청 여부만 반환한다. 실제 발사 가능 여부나 발사 비용은 여기서 처리하지 않는다.
        # 스탯 강화·전직은 선택값만 반환하고 포인트·레벨·현재 선택지의 유효성은 코어에서 검사한다.
        pass
