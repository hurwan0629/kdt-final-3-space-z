from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import partial
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from space_z.rl import ContinuousActionDecoder, DiscreteActionDecoder, PixelObservation
from space_z.rl import RewardCalculator, TankEnv, VectorObservation
from space_z.rl.actions.action_decoder import ActionDecoder
from space_z.rl.observations.observation_builder import ObservationBuilder

from .configuration import ExperimentConfig


class PlayerSnapshot(TypedDict):
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    angle: float
    hp: float
    max_hp: float


class ControlRecord(TypedDict):
    move_x: float
    move_y: float
    rotation: float
    want_fire: bool
    stat_upgrade_key: str | None
    evolution_choice: int | None


class EpisodeStepRecord(TypedDict):
    episode_id: int
    episode_seed: int
    policy_action: int | list[float]
    player_control: ControlRecord
    player_snapshot: PlayerSnapshot
    terminated: bool
    truncated: bool


class RecordedState(gym.Wrapper):
    """환경을 변경하지 않고 실제 에피소드 시드·행동 후 상태를 info에 추가한다."""

    def __init__(
        self,
        environment: TankEnv,
        initial_episode_seed: int,
        episode_seed_increment: int,
    ) -> None:
        super().__init__(environment)
        self.next_episode_seed = initial_episode_seed
        self.episode_seed_increment = episode_seed_increment
        self.episode_id = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_seed = self.next_episode_seed if seed is None else int(seed)
        self.next_episode_seed = (self.episode_seed + self.episode_seed_increment) % 2**32
        self.episode_id += 1
        return self.env.reset(seed=self.episode_seed, options=options)

    def step(self, action):
        observation, step_reward, terminated, truncated, environment_info = self.env.step(action)

        # 행동 후 플레이어 상태
        player_state = self.unwrapped.game.player
        player_snapshot: PlayerSnapshot = {
            "x": player_state.x,
            "y": player_state.y,
            "velocity_x": player_state.velocity_x,
            "velocity_y": player_state.velocity_y,
            "angle": player_state.angle,
            "hp": player_state.hp,
            "max_hp": player_state.max_hp,
        }
        # 조작 입력·step 기록
        player_control_record = asdict(self.unwrapped.action_decoder.decode(action))
        episode_step_record: EpisodeStepRecord = {
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "policy_action": np.asarray(action).tolist(),
            "player_control": player_control_record,
            "player_snapshot": player_snapshot,
            "terminated": terminated, 
            "truncated": truncated,
        }
        environment_info["episode_step_record"] = episode_step_record
        return observation, step_reward, terminated, truncated, environment_info


def make_environment(
    experiment_config: ExperimentConfig,
    environment_index: int = 0,
    record_episode_steps: bool = True,
) -> VecEnv:
    """관찰·행동·보상 전략을 조립한다. 병렬 환경도 이 함수를 프로세스 안에서 호출한다."""
    # 관찰 생성기
    observation_builder_classes: dict[str, type[ObservationBuilder]] = {
        "vector": VectorObservation,
        "pixel": PixelObservation,
    }
    observation_builder_class = observation_builder_classes[experiment_config.observation.observation_type]
    observation_config = deepcopy(experiment_config.observation.sensor_config)
    observation_builder: ObservationBuilder = observation_builder_class(observation_config)

    # 행동 해석기
    action_decoder_classes: dict[str, type[ActionDecoder]] = {
        "continuous": ContinuousActionDecoder,
        "discrete": DiscreteActionDecoder,
    }
    action_decoder_class = action_decoder_classes[experiment_config.action.action_type]
    action_decoder = action_decoder_class(
        enable_stat_upgrades=experiment_config.action.enable_stat_upgrades,
        enable_evolution=experiment_config.action.enable_evolution,
    )

    # 게임·보상·환경 설정
    game_config = deepcopy(experiment_config.game)
    reward_config = deepcopy(experiment_config.reward)
    reward_calculator = RewardCalculator(reward_config)
    environment_config = deepcopy(experiment_config.environment)
    # 공통 게임 환경
    game_environment = TankEnv(
        game_config,
        observation_builder,
        action_decoder,
        reward_calculator,
        environment_config,
    )
    # 선택적 step·에피소드 기록
    if not record_episode_steps:
        return game_environment
    training_config = experiment_config.training
    recorded_game_environment = RecordedState(
        environment=game_environment,
        initial_episode_seed=training_config.seed + environment_index,
        episode_seed_increment=training_config.num_envs,
    )
    return Monitor(recorded_game_environment)


def make_training_environments(experiment_config: ExperimentConfig):
    """개별 Monitor를 가진 단일 또는 다중 프로세스 환경을 준비한다."""
    environment_factories = [
        partial(make_environment, experiment_config, environment_index)
        for environment_index in range(experiment_config.training.num_envs)
    ]
    if experiment_config.training.parallel:
        return SubprocVecEnv(environment_factories, start_method="spawn")
    return DummyVecEnv(environment_factories)
