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
    """자기 상태 → 성장 → 적 → 천체 → 적 탄환 → 레이 → 지형 순서의 float32 벡터."""

    def __init__(self, observation_config: ObsConfig | None = None) -> None:
        """슬롯 수와 타일 범위를 받아 고정된 관찰 공간과 구간을 준비한다."""
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
        """현재 상태만 읽어 정규화한 관찰을 반환한다. 없는 슬롯은 0으로 채운다.

        위치는 관찰 반폭·반높이, 속도는 기본 속도 기준, 반지름은 타일 길이로 나눈다.
        범위를 넘는 값은 [-1, 1]로 제한한다. 지형은 플레이어가 속한 타일 중심의 격자이며
        맵 밖은 벽이다. 선택적인 부서지는 벽 채널은 남은 체력 비율을 표현한다.
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
