"""사람 화면과 독립된 지역 이미지와 성장 벡터를 생성한다."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.spaces import Box, Dict
from numpy.typing import NDArray

from space_z.config import ObsConfig

from space_z.rl.observations.observation_builder import ObservationBuilder
from space_z.rl.observations.progression_observation import build_progression

if TYPE_CHECKING:
    from space_z.game.game_core import GameCore
    from space_z.rendering.renderer import Renderer


class PixelObservation(ObservationBuilder):
    """CHW 흑백 이미지 스택(uint8)과 성장 벡터(float32)를 딕셔너리로 반환한다."""

    def __init__(self, observation_config: ObsConfig | None = None) -> None:
        """관찰 범위·이미지 해상도·누적 장수를 받아 고정 공간을 준비한다."""
        self.config = replace(observation_config) if observation_config is not None else ObsConfig()
        self._observation_space = Dict({
            "image": Box(0, 255, (self.config.frame_stack, self.config.image_height, self.config.image_width), np.uint8),
            "progression": Box(0.0, 1.0, (self.config.PROGRESSION_DIM,), np.float32),
        })
        self._frames: deque[NDArray[np.uint8]] = deque(maxlen=self.config.frame_stack)
        self._renderer: Renderer | None = None
        self._last_frame: int | None = None
        self._terrain: object | None = None

    @property
    def observation_space(self) -> Dict:
        """image와 progression 공간을 반환한다."""
        return self._observation_space

    def build(self, game: GameCore) -> dict[str, NDArray[np.uint8] | NDArray[np.float32]]:
        """지역 화면을 흑백으로 축소해 최근 이미지와 성장 상태를 반환한다.

        첫 관찰은 같은 이미지로 스택을 채운다. 같은 프레임을 다시 읽으면 스택을 밀지 않는다.
        pygame 창·메뉴·미니맵을 사용하지 않으며 게임 상태와 이벤트를 바꾸지 않는다.
        """
        import pygame
        from space_z.rendering.renderer import Renderer

        # 1) 타일 관찰 범위와 정확히 같은 크기의 창 없는 렌더러를 준비한다.
        if self._terrain is not game.terrain:
            self.reset()
            self._terrain = game.terrain
        if self._renderer is None:
            pygame.font.init()
            config = replace(game.config, window_width=self.config.grid_width * game.terrain.tile,
                             window_height=self.config.grid_height * game.terrain.tile, camera_zoom=1.0)
            self._renderer = Renderer(config, visible=False)

        # 2) 새 프레임만 이미지로 만들어 스택에 넣는다.
        if self._last_frame != game.frame:
            self._renderer.draw(game, show_controls=False, include_ui=False)
            surface = pygame.transform.smoothscale(self._renderer.screen, (self.config.image_width, self.config.image_height))
            rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
            grayscale = np.rint(rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
            if not self._frames:
                self._frames.extend(grayscale.copy() for _ in range(self.config.frame_stack))
            else:
                self._frames.append(grayscale)
            self._last_frame = game.frame
        return {"image": np.stack(self._frames), "progression": build_progression(game)}

    def reset(self) -> None:
        """이전 판 이미지·지형·연출 캐시를 비운다. 다른 객체의 창은 종료하지 않는다."""
        self._frames.clear()
        self._renderer = None
        self._last_frame = None
        self._terrain = None

    def close(self) -> None:
        """관찰 생성기가 소유한 이미지와 렌더러 참조를 해제한다."""
        self.reset()
