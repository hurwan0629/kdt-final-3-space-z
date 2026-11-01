"""벡터 관찰의 지형 격자를 CNN으로 처리하고 나머지 수치 정보와 결합한다."""
from __future__ import annotations

from math import prod

from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn


class TerrainGridFeaturesExtractor(BaseFeaturesExtractor):
    """지형 격자만 학습 가능한 CNN으로 압축하는 벡터 특징 추출기다."""

    def __init__(
        self,
        observation_space: Box,
        terrain_grid_slice: slice,
        terrain_grid_shape: tuple[int, int, int],
        terrain_feature_count: int,
    ) -> None:
        # 관찰과 격자 구간의 일치 확인
        if not isinstance(observation_space, Box) or len(observation_space.shape) != 1:
            raise ValueError("지형 CNN에는 1차원 Box 관찰이 필요합니다.")
        if len(terrain_grid_shape) != 3 or any(type(size) is not int or size <= 0 for size in terrain_grid_shape):
            raise ValueError("지형 격자는 양의 정수인 (채널, 높이, 너비)여야 합니다.")
        if type(terrain_feature_count) is not int or terrain_feature_count <= 0:
            raise ValueError("지형 특징 개수는 양의 정수여야 합니다.")
        if not isinstance(terrain_grid_slice, slice):
            raise ValueError("지형 격자 범위는 slice여야 합니다.")
        grid_start = terrain_grid_slice.start
        grid_stop = terrain_grid_slice.stop
        observation_size = observation_space.shape[0]
        if (
            type(grid_start) is not int
            or type(grid_stop) is not int
            or terrain_grid_slice.step not in (None, 1)
            or not 0 <= grid_start < grid_stop <= observation_size
            or grid_stop - grid_start != prod(terrain_grid_shape)
        ):
            raise ValueError("지형 격자의 slice와 shape가 관찰 벡터에 맞지 않습니다.")

        numeric_feature_count = observation_size - prod(terrain_grid_shape)
        combined_feature_count = numeric_feature_count + terrain_feature_count
        super().__init__(observation_space, features_dim=combined_feature_count)
        self.terrain_grid_slice = terrain_grid_slice
        self.terrain_grid_shape = terrain_grid_shape
        self.terrain_feature_count = terrain_feature_count

        # 작은 격자도 처리할 수 있도록 패딩을 유지한다.
        grid_channels, grid_height, grid_width = terrain_grid_shape
        convolved_height = (grid_height + 3) // 4
        convolved_width = (grid_width + 3) // 4
        convolved_feature_count = 32 * convolved_height * convolved_width
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(convolved_feature_count, terrain_feature_count),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        terrain_values = observations[:, self.terrain_grid_slice]
        terrain_grid = terrain_values.reshape(-1, *self.terrain_grid_shape)
        terrain_features = self.terrain_cnn(terrain_grid)
        return torch.cat(
            (
                observations[:, :self.terrain_grid_slice.start],
                observations[:, self.terrain_grid_slice.stop:],
                terrain_features,
            ),
            dim=1,
        )
