"""벡터·픽셀 관찰 생성기가 따라야 할 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

Observation = NDArray[np.float32] | NDArray[np.uint8] | dict[str, NDArray[np.float32] | NDArray[np.uint8]]

if TYPE_CHECKING:
    from gymnasium.spaces import Space

    from space_z.game.game_core import GameCore


class ObservationBuilder(ABC):
    """게임 상태를 읽어 모델 입력을 만드는 관찰 생성기의 공통 계약.

    TankEnv는 이 인터페이스만 사용하고 벡터·픽셀 종류를 직접 구분하지 않는다.
    구현체는 관찰 공간·관찰 생성·새 판 초기화·자원 정리를 모두 정의해야 한다.
    """

    @property
    @abstractmethod
    def observation_space(self) -> Space[Any]:
        """반환할 관찰 배열의 크기·자료형·값 범위를 정의한 Gymnasium 공간.

        반환:
            벡터 구현체는 Box, 픽셀 구현체는 이미지·성장 벡터의 Dict 공간.
            build() 결과는 이 공간에 속해야 하며 한 환경을 사용하는 동안 형태가 일정해야 한다.
        """
        # 구현체가 준비한 공간을 제공한다. 속성을 조회할 때 게임을 진행하지 않는다.
        pass

    @abstractmethod
    def build(self, game: GameCore) -> Observation:
        """현재 게임 상태로부터 모델에 전달할 관찰 배열을 만든다.

        인자:
            game: 관찰할 현재 게임 코어. 위치·속도·체력·지형 등을 읽는 원본이다.

        반환:
            observation_space를 따르는 벡터 또는 이미지·성장 벡터 딕셔너리.
            게임의 변경 가능한 내부 배열을 외부에서 직접 수정할 수 없도록 반환한다.
        """
        # 벡터 구현체는 필요한 상태를 추출·정규화하고, 픽셀 구현체는 게임 화면 이미지를 만든다.
        # 내부 캐시·이미지 누적 상태는 갱신할 수 있지만 게임 진행·전투 이벤트 소비는 하지 않는다.
        # 픽셀 관찰을 만들 때 사람용 창 표시나 FPS 제한을 요구하지 않는다.
        pass

    @abstractmethod
    def reset(self) -> None:
        """새 에피소드의 첫 관찰을 만들기 전에 관찰 생성기의 이전 판 상태를 비운다.

        인자:
            없음. 구현체가 보관한 관찰용 상태만 초기화한다.

        반환:
            None. 게임 자체의 초기화는 TankEnv와 GameCore가 담당한다.
        """
        # 이전 지형 캐시·누적 이미지 등을 비워 새 판의 첫 build()에 섞이지 않도록 한다.
        # 보관하는 상태가 없는 구현체도 이 메서드를 정의하되 별도 작업 없이 끝낼 수 있다.
        pass

    @abstractmethod
    def close(self) -> None:
        """관찰 생성기가 소유한 이미지 생성 자원 등을 정리한다.

        인자:
            없음. 구현체 자신이 생성·소유한 자원만 정리한다.

        반환:
            None. 정리할 자원이 없거나 반복 호출한 경우에도 안전해야 한다.
        """
        # 픽셀 생성용 자원을 해제한다. 다른 객체가 소유한 게임 창이나 자원은 종료하지 않는다.
        # 별도 자원을 사용하지 않는 구현체는 아무 작업 없이 끝낼 수 있다.
        pass
