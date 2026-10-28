"""플레이어 중심의 일정 범위에서 개체와 지형을 숫자로 관찰한다."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence, TypeVar

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from space_z.config import ObsConfig
from space_z.game.entities import Bullet, Shape, Tank
from space_z.game.terrain import BREAKABLE, MUD
from space_z.rl.observations.observation_builder import ObservationBuilder
from space_z.rl.observations.progression_observation import build_progression, ship_one_hot

if TYPE_CHECKING:
    from space_z.game.game_core import GameCore

Entity = TypeVar("Entity", Tank, Shape, Bullet)


class VectorObservation(ObservationBuilder):
    """플레이어 주변 상태를 고정 길이의 1차원 float32 벡터로 생성한다.

    생성 순서:
        자기 상태 → 성장 → 적 → 천체 → 적 탄환 → 레이 → 지형.
        기본 설정에서는 (5139,) 배열이며 모든 원소는 [-1, 1] 범위다.
        슬롯 수·관찰 범위·지형 채널·레이 수에 따라 전체 길이는 달라진다.

    사용:
        builder.build(game)으로 실제 관찰을 만들고,
        observation[builder.slices["enemies"]]처럼 항목별 구간을 조회한다.
        상세 원소 순서와 정규화 기준은 build()의 docstring을 참고한다.
    """

    def __init__(self, observation_config: ObsConfig | None = None) -> None:
        """관찰 설정으로 배열 규격과 항목별 슬라이스를 준비한다.

        인자:
            observation_config: 개체 슬롯 수·관찰 타일 범위·지형 채널·레이 설정.
                None이면 기본 ObsConfig를 사용하고, 전달한 설정은 복사해 보관한다.

        생성하는 값:
            config: 관찰 생성에 사용하는 설정.
            grid_shape: 지형 격자 크기 (채널 수, 세로 타일 수, 가로 타일 수).
                기본값은 (3, 29, 45).
            _observation_space: 전체 배열 규격인 Box(-1, 1), 기본 shape=(5139,).
            slices: self·progression·enemies·shapes·bullets·rays·grid별 슬라이스.

        반환:
            None. 실제 게임 상태를 담은 배열은 build(game)에서 생성한다.
        """
        self.config = replace(observation_config) if observation_config is not None else ObsConfig()
        self.grid_shape = self.config.grid_shape()
        self._observation_space = Box(-1.0, 1.0, (self.config.flat_size(),), np.float32)
        offset = 0
        self.slices: dict[str, slice] = {}  # 항목별 벡터 구간(디버그·검증용)
        for name, size in (
            ("self", self.config.SELF_DIM),
            ("progression", self.config.PROGRESSION_DIM),
            ("enemies", self.config.n_enemies * self.config.ENEMY_DIM),
            ("shapes", self.config.n_shapes * self.config.SHAPE_DIM),
            ("bullets", self.config.n_bullets * self.config.BULLET_DIM),
            ("rays", self.config.n_rays),
            ("grid", math.prod(self.grid_shape)),
        ):
            self.slices[name] = slice(offset, offset + size)
            offset += size

    @property
    def observation_space(self) -> Box:
        """고정 길이의 [-1, 1] float32 공간을 반환한다."""
        return self._observation_space

    def _nearby(self, entities: Sequence[Entity], game: GameCore, limit: int) -> list[Entity]:
        """관찰 사각형과 충돌 원이 겹치는 개체를 가까운 순서로 최대 limit개 반환한다."""
        player = game.player
        half_width = self.config.grid_width * game.terrain.tile / 2
        half_height = self.config.grid_height * game.terrain.tile / 2
        visible = [entity for entity in entities if
                   max(0, abs(entity.x - player.x) - half_width)**2 +
                   max(0, abs(entity.y - player.y) - half_height)**2 <= entity.radius**2]
        return sorted(visible, key=lambda entity: (entity.x-player.x)**2 + (entity.y-player.y)**2)[:limit]

    def build(self, game: GameCore) -> NDArray[np.float32]:
        """현재 상태를 읽어 아래 순서로 연결한 관찰 벡터를 반환한다.

        인자:
            game: 플레이어·개체·성장·지형을 조회할 GameCore.
                게임을 진행하거나 전투 이벤트를 소비하지 않는다.

        반환:
            shape=(config.flat_size(),), dtype=float32인 새 배열.
            기본 설정의 구간은 아래와 같다. 슬라이스의 끝 인덱스는 포함하지 않는다.
                [0:10]      자기 상태 10개
                [10:133]    성장 상태 123개
                [133:412]   적 9기 x 31개
                [412:712]   천체 30개 x 10개
                [712:1224]  적 탄환 64발 x 8개
                [1224:1224] 레이 0개 (기본 비활성)
                [1224:5139] 지형 3채널 x 29행 x 45열
            설정이 다르면 실제 구간은 self.slices와 self.grid_shape로 확인한다.

        자기 상태 (10개, 아래 순서):
            [0:2] 월드 속도 (velocity_x, velocity_y) / 기본 최대 속도 기준.
            [2]   현재 체력 / max(1, 최대 체력).
            [3:5] 포신 방향의 (sin(angle), cos(angle)).
            [5]   남은 발사 대기 프레임 / max(1, 현재 무기의 발사 대기 간격).
            [6]   충돌 반지름 / 타일 길이.
            [7]   탄약 제한 여부 (켜짐=1, 꺼짐=0).
            [8]   남은 탄약 / max(1, 탄창 용량). 탄약 무제한이면 0.
            [9]   남은 재장전 프레임 / max(1, 전체 재장전 프레임). 무제한이면 0.

        성장 상태 (현재 콘텐츠에서 123개):
            현재 기체 원핫 → 레벨·경험치·강화 포인트·대기 특수 보상 수
            → 스탯별 강화 단계 → 스탯별 강화 가능 여부 → 전직 선택지 슬롯 순서.
            레벨·강화 포인트·대기 특수 보상 수는 x/(x+1), 경험치는 현재 값/다음 레벨
            필요 경험치, 강화 단계는 현재 단계/STAT_MAX_LEVEL로 정규화한다.
            스탯은 STAT_UPGRADES 순서 (현재 health, damage, range, rapid, speed).
            기체 원핫은 SHIPS 순서이며 해당 기체만 1이다.
            전직 슬롯은 존재 여부 1개와 후보 기체 원핫 21개씩 총 4칸이며,
            없는 선택지는 0으로 채운다. 실제 생성은 build_progression(game)이 맡는다.

        적 한 기 (현재 콘텐츠에서 31개):
            [0] 존재 여부 (실제 개체=1).
            [1:3] 상대 위치 (dx, dy), [3:5] 월드 속도 (velocity_x, velocity_y).
            [5] 체력 비율, [6:8] 포신 방향 (sin, cos), [8] 반지름/타일 길이.
            [9:30] 기체 원핫 21개 (SHIPS 순서), [30] 엘리트 여부 (맞으면 1).

        천체 한 개 (10개):
            [0] 존재 여부, [1:3] 상대 위치, [3:5] 월드 속도, [5] 체력 비율,
            [6] 반지름/타일 길이, [7:10] 종류 원핫 (square, triangle, pentagon).

        적 탄환 한 발 (8개):
            [0] 존재 여부, [1:3] 상대 위치, [3:5] 월드 속도,
            [5] 반지름/타일 길이, [6:8] 종류 표시 (plasma, shell).
            두 종류 이외의 탄환은 종류 표시가 모두 0이다.

        개체 선택·공통 정규화:
            플레이어 중심 사각형과 충돌 원이 겹치는 개체를 종류별로 가까운 순서로 넣는다.
            죽은 적과 플레이어 진영 탄환은 제외하고, 부족한 슬롯은 전부 0으로 채운다.
            벽에 가려진 개체도 범위 안이면 포함한다.
            dx=(개체 x-플레이어 x)/관찰 반폭, dy=(개체 y-플레이어 y)/관찰 반높이.
            오른쪽·아래쪽이 양수다. 선체·천체 속도는 max(1e-6, abs(game.config.max_speed)),
            탄환 속도는 max(1e-6, game.config.bullet_speed)로 나눈다.

        레이·지형:
            레이는 포신 방향부터 시계방향으로 같은 각도 간격의 벽까지 거리/ray_max_dist.
            지형은 플레이어가 속한 타일 중심의 격자이며 채널 → 행 → 열 순서로 펼친다.
            첫 채널은 이동을 막는 타일=1, 바닥=0이며 맵 밖도 1이다.
            include_mud가 켜지면 중력장 여부 채널, include_breakable이 켜지면
            파괴 가능한 벽의 남은 체력 비율 채널이 뒤에 붙는다. 해당 지형이 없으면 0이다.

        범위 처리:
            최종 모든 값은 [-1, 1]로 clipping한다. 존재 여부·원핫·비율은 [0, 1]이며,
            상대 위치·속도·방향의 sin/cos에는 음수가 들어갈 수 있다.
        """
        config, player = self.config, game.player
        tile = game.terrain.tile
        half_width, half_height = config.grid_width * tile / 2, config.grid_height * tile / 2
        tank_speed = max(1e-6, abs(game.config.max_speed))
        bullet_speed = max(1e-6, game.config.bullet_speed)
        result = np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1) 자기 상태와 두 관찰 방식에서 공유하는 성장 상태를 기록한다.
        result[self.slices["self"]] = (
            player.velocity_x/tank_speed, player.velocity_y/tank_speed,
            player.hp/max(1, player.max_hp), math.sin(player.angle), math.cos(player.angle),
            player.cooldown/max(1, game.combat.weapon_values(player)[2]), player.radius/tile,
            float(game.config.limited_ammo),
            player.ammo/max(1, player.max_ammo) if game.config.limited_ammo else 0,
            game.combat.reload_frames_left/max(1, game.combat.reload_frames_total) if game.config.limited_ammo else 0,
        )
        result[self.slices["progression"]] = build_progression(game)

        # 2) 범위 안의 개체를 거리순으로 고정 슬롯에 넣는다.
        enemies = self._nearby([enemy for enemy in game.enemies if enemy.alive], game, config.n_enemies)
        shapes = self._nearby(game.shapes, game, config.n_shapes)
        bullets = self._nearby([bullet for bullet in game.combat.bullets if bullet.team != player.team], game, config.n_bullets)
        for name, entities, dimension in (
            ("enemies", enemies, config.ENEMY_DIM),
            ("shapes", shapes, config.SHAPE_DIM),
            ("bullets", bullets, config.BULLET_DIM),
        ):
            for index, entity in enumerate(entities):
                speed = bullet_speed if isinstance(entity, Bullet) else tank_speed
                values = [1.0, (entity.x-player.x)/half_width, (entity.y-player.y)/half_height,
                          entity.velocity_x/speed, entity.velocity_y/speed]
                if isinstance(entity, Tank):
                    values.extend((entity.hp/max(1, entity.max_hp), math.sin(entity.angle),
                                   math.cos(entity.angle), entity.radius/tile,
                                   *ship_one_hot(entity.job), float(entity.elite)))
                elif isinstance(entity, Shape):
                    values.extend((entity.hp/max(1, entity.max_hp), entity.radius/tile,
                                   *(float(entity.kind == kind) for kind in ("square", "triangle", "pentagon"))))
                else:
                    values.extend((entity.radius/tile, float(entity.kind == "plasma"), float(entity.kind == "shell")))
                start = self.slices[name].start + index * dimension
                result[start:start+dimension] = values

        # 3) 선택적인 레이와 플레이어 주변 지형을 기록한다.
        for index in range(config.n_rays):
            angle = player.angle + index * math.tau/config.n_rays
            result[self.slices["rays"].start + index] = game.terrain.raycast(
                player.x, player.y, math.cos(angle), math.sin(angle), config.ray_max_dist
            )/config.ray_max_dist
        grid = result[self.slices["grid"]].reshape(self.grid_shape)
        center_col, center_row = game.terrain.tile_of(player.x, player.y)
        for row in range(config.grid_height):
            for col in range(config.grid_width):
                world_col = center_col + col - config.grid_width//2
                world_row = center_row + row - config.grid_height//2
                grid[0, row, col] = game.terrain.is_solid_tile(world_col, world_row)
                channel = 1
                inside = 0 <= world_col < game.terrain.cols and 0 <= world_row < game.terrain.rows
                kind = game.terrain.tiles[world_row][world_col] if inside else None
                if config.include_mud:
                    grid[channel, row, col] = kind == MUD
                    channel += 1
                if config.include_breakable and kind == BREAKABLE:
                    grid[channel, row, col] = game.terrain.hp[(world_col, world_row)]/max(1e-6, game.terrain.breakable_hp_max)
        return np.clip(result, -1, 1)

    def reset(self) -> None:
        """이전 프레임을 보관하지 않으므로 초기화할 상태가 없다."""

    def close(self) -> None:
        """외부 자원을 소유하지 않으므로 해제할 자원이 없다."""
