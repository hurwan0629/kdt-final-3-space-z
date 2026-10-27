"""강화학습의 관찰·보상·환경 설정."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from .progression_definitions import FAMILIES, SHIPS, STAT_UPGRADES


@dataclass
class ObsConfig:
    n_enemies: int = 9  # 관찰 범위 안에서 가까운 적의 최대 수
    n_shapes: int = 30  # 관찰 범위 안에서 가까운 천체의 최대 수
    n_bullets: int = 64  # 관찰 범위 안에서 가까운 적 탄환의 최대 수
    grid_width: int = 45  # 관찰 범위의 가로 타일 수(홀수)
    grid_height: int = 29  # 관찰 범위의 세로 타일 수(홀수)
    include_mud: bool = True  # 중력장 격자 채널 포함 여부
    include_breakable: bool = True  # 파괴 가능한 벽의 남은 체력 비율 채널 포함 여부
    image_width: int = 180  # 픽셀 관찰의 가로 픽셀 수
    image_height: int = 116  # 픽셀 관찰의 세로 픽셀 수
    frame_stack: int = 4  # 최근 흑백 이미지 누적 장수(CHW)
    n_rays: int = 0  # 레이캐스트 센서 개수 (0이면 끔)
    ray_max_dist: float = 480.0  # 레이캐스트 센서 최대 탐지 거리 (px)
    SELF_DIM: ClassVar[int] = 10  # 속도2·체력·방향2·발사대기·반지름·탄약활성·탄약비율·재장전
    ENEMY_DIM: ClassVar[int] = 10 + len(SHIPS)  # 존재·위치2·속도2·체력·방향2·반지름·기체·엘리트
    SHAPE_DIM: ClassVar[int] = 10  # 존재·위치2·속도2·체력·반지름·종류3
    BULLET_DIM: ClassVar[int] = 8  # 존재·위치2·속도2·반지름·종류2
    EVOLUTION_SLOTS: ClassVar[int] = max(len(FAMILIES), 1)  # 최대 전직 선택지 수
    PROGRESSION_DIM: ClassVar[int] = (
        len(SHIPS) + 4 + 2 * len(STAT_UPGRADES) + EVOLUTION_SLOTS * (1 + len(SHIPS))
    )  # 기체·성장수치·스탯단계/가능여부·전직슬롯

    def __post_init__(self) -> None:
        """관찰 크기와 센서 범위가 유효한지 검사한다."""
        for name in ("n_enemies", "n_shapes", "n_bullets", "n_rays"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name}는 0 이상의 정수여야 합니다.")
        for name in ("grid_width", "grid_height", "image_width", "image_height", "frame_stack"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}는 양의 정수여야 합니다.")
        if self.grid_width % 2 == 0 or self.grid_height % 2 == 0:
            raise ValueError("관찰 타일 수는 홀수여야 합니다.")
        if type(self.include_mud) is not bool or type(self.include_breakable) is not bool:
            raise TypeError("추가 지형 채널의 포함 여부는 bool이어야 합니다.")
        if not math.isfinite(self.ray_max_dist) or self.ray_max_dist <= 0:
            raise ValueError("ray_max_dist는 유한한 양수여야 합니다.")

    @property
    def vector_size(self) -> int:  # 지형 격자를 제외한 전체 관찰 벡터 길이
        """지형 격자를 제외한 벡터 부분 (레이 포함)."""
        return (
            self.SELF_DIM
            + self.PROGRESSION_DIM
            + self.n_enemies * self.ENEMY_DIM
            + self.n_shapes * self.SHAPE_DIM
            + self.n_bullets * self.BULLET_DIM
            + self.n_rays
        )

    def grid_shape(self, n_channels: int | None = None) -> tuple[int, int, int]:
        """지형 격자 크기(채널, 세로, 가로)를 반환한다."""
        if n_channels is None:
            n_channels = 1 + int(self.include_mud) + int(self.include_breakable)
        return (n_channels, self.grid_height, self.grid_width)

    def flat_size(
        self, n_channels: int | None = None
    ) -> int:  # 벡터와 지형 격자를 펼쳐 합친 관찰 길이
        channels, height, width = self.grid_shape(n_channels)
        return self.vector_size + channels * height * width


@dataclass
class RewardConfig:
    # 천체 피해 보상 계수: 피해량 × (점수 / 최대 체력)에 곱함
    shape_damage: float = 0.002
    shape_destroy: float = 0.01  # 천체 파괴 보상 계수: 파괴한 천체 점수에 곱함
    enemy_damage: float = 0.01  # 적 피해 보상 계수: 적에게 준 피해량에 곱함
    enemy_kill: float = 2.0  # 적 한 대 처치 시 보상
    damage_taken: float = -0.02  # 피격 보상 계수: 받은 피해량에 곱함 (음수면 페널티)
    death: float = -20.0  # 플레이어 사망 시 보상 (음수면 페널티)
    time_step: float = -0.0001  # step마다 (생존 보너스/시간 페널티용)
    shot_cost: float = 0.01  # 실제 발사 탄환 1발당 차감할 보상(0 이상의 유한한 값, 0이면 비용 없음)


@dataclass
class EnvConfig:
    frame_skip: int = 4  # 1 step = 몇 게임 프레임
    max_steps: int = 3000  # 에피소드 최대 step 수 (도달 시 truncated=True)
