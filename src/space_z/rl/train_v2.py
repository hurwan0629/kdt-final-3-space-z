"""설정 JSON으로 실행하는 PPO 기록 예제. 기존 환경·train.py는 수정하지 않는다.

실행: python -m space_z.rl.train_v2 --config src/space_z/rl/runs/model_001/fighting.json
설정 최소 예: {"game": {"num_enemies": 0, "terrain": {"enabled": false}},
              "training": {"additional_steps": 10000}}
필요 패키지: python -m pip install stable-baselines3 tensorboard

JSON 옆 model.json에서 자동 재개하고 001_fighting/ 등에 실제 설정·소스 사본,
TensorBoard·step/에피소드 JSONL·체크포인트·고정 시나리오 평가를 저장한다.
추가 학습은 최근 저장 완료 지점부터 시작한다. 게임 중간 상태·rollout 버퍼·영상은 저장하지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Literal, TextIO, TypedDict
from zipfile import ZipFile, ZIP_DEFLATED

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from space_z.rl.observations.observation_builder import ObservationBuilder
from space_z.rl.actions.action_decoder import ActionDecoder

from space_z.config import EnemyAIConfig, EnvConfig, GameConfig, ObsConfig, RewardConfig, TerrainConfig
from space_z.rl import ContinuousActionDecoder, DiscreteActionDecoder, PixelObservation
from space_z.rl import PolicyEvaluator, RewardCalculator, TankEnv, VectorObservation
from space_z.game.types import CombatEvents


class PPOOverrides(TypedDict, total=False):
    """생략한 항목은 재개할 모델에서 상속한다. 명시적인 None과 구분한다."""

    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float | None
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    normalize_advantage: bool
    target_kl: float | None


PPO_DEFAULTS: PPOOverrides = {
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "target_kl": None,
}


@dataclass(frozen=True)
class TrainingConfig:
    """추가 학습량·시드·연산 장치·병렬 환경 수·저장 주기를 설정한다."""

    additional_steps: int = 10000
    seed: int = 42
    device: str = "cpu"
    parallel: bool = False
    num_envs: int = 4
    checkpoint_interval_steps: int = 10000


@dataclass(frozen=True)
class ObservationSettings:
    """벡터·픽셀 관찰 방식과 관찰 범위·크기를 설정한다."""

    observation_type: Literal["vector", "pixel"]
    sensor_config: ObsConfig


@dataclass(frozen=True)
class ActionSettings:
    """연속·이산 행동 방식과 강화·진화 선택 지원 여부를 설정한다."""

    action_type: Literal["continuous", "discrete"]
    enable_stat_upgrades: bool
    enable_evolution: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """모델 비교에 사용할 고정 시드·시나리오·보상·환경 조건이다."""

    seeds: list[int]
    environment: EnvConfig
    reward: RewardConfig
    scenarios: dict[str, GameConfig]


@dataclass
class ExperimentConfig:
    """실험 전체 설정과 모델 생성·복원 후 확정된 PPO 값을 보관한다."""

    description: str
    game: GameConfig
    reward: RewardConfig
    environment: EnvConfig
    observation: ObservationSettings
    action: ActionSettings
    training: TrainingConfig
    ppo_overrides: PPOOverrides
    evaluation: EvaluationConfig
    resolved_ppo_parameters: PPOOverrides | None = None


class TrainingRunRecordUpdates(TypedDict, total=False):
    latest_checkpoint: str
    finished_at: str
    error: str


class TrainingRunRecord(TrainingRunRecordUpdates):
    id: int
    folder: str
    source_config: str
    parent_checkpoint: str | None
    start_step: int | None
    end_step: int | None
    status: Literal["running", "completed", "interrupted", "failed"]
    started_at: str


class ModelRegistry(TypedDict):
    model_id: str
    description: str
    algorithm: str
    latest_checkpoint: str | None
    evaluation: dict[str, Any]
    training_sessions: list[TrainingRunRecord]


@dataclass
class RunSession:
    """이번 실행의 저장 경로·부모 체크포인트·공유 모델 이력을 연결한다."""

    model_directory: Path
    run_directory: Path
    parent_checkpoint_path: Path | None
    model_registry: ModelRegistry
    training_run_record: TrainingRunRecord


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


@dataclass(frozen=True)
class RewardBreakdown:
    """환경 한 step의 보상을 원인별 기여값으로 나눈 결과다."""

    shape_damage: float
    shape_destroy: float
    enemy_damage: float
    enemy_kill: float
    damage_taken: float
    death: float
    time_step: float
    shot_cost: float


class StepLogRecord(TypedDict):
    """기존 JSONL 필드명을 유지하는 저장용 step 기록이다."""

    global_step: int
    env_id: int
    episode_id: int
    seed: int
    action: int | list[float]
    control: ControlRecord
    player: PlayerSnapshot
    terminated: bool
    truncated: bool
    reward: float
    reward_components: dict[str, float]
    info: dict[str, Any]


def json_default(value: Any) -> Any:
    """NumPy 수치·배열과 경로만 JSON의 기본 자료형으로 변환한다."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON으로 저장할 수 없는 값: {type(value).__name__}")


def write_json(json_file_path: Path, json_document: dict[str, Any]) -> None:
    """완성된 임시 파일을 교체해 대표 JSON의 부분 쓰기를 막는다."""
    temporary_json_path = json_file_path.with_suffix(".tmp")
    temporary_json_path.write_text(json.dumps(json_document, ensure_ascii=False, indent=2, default=json_default,
                                    allow_nan=False), encoding="utf-8")
    temporary_json_path.replace(json_file_path)


def append_record(log_file: TextIO, log_record: dict[str, Any]) -> None:
    """메인 프로세스가 JSONL에 원본 기록 한 줄을 추가한다."""
    serialized_log_record = json.dumps(
        log_record, ensure_ascii=False, default=json_default, allow_nan=False
    )
    log_file.write(serialized_log_record + "\n")


def merge_settings(default_settings: dict[str, Any], user_overrides: dict[str, Any]) -> dict[str, Any]:
    """기본값을 채우되 알 수 없는 키는 오타로 처리한다."""
    if not isinstance(user_overrides, dict):
        raise ValueError("설정 항목은 JSON 객체여야 합니다.")
    merged_settings = deepcopy(default_settings)
    for setting_name, override_value in user_overrides.items():
        if setting_name not in merged_settings:
            raise ValueError(f"알 수 없는 설정: {setting_name}")
        merged_settings[setting_name] = merge_settings(merged_settings[setting_name], override_value) if isinstance(merged_settings[setting_name], dict) else override_value
    return merged_settings


def read_configuration(configuration_file_path: Path) -> ExperimentConfig:
    """사용자 설정을 검증하고 기본값을 채운다. 생략한 PPO 값은 재개 시 부모에서 상속한다."""
    if configuration_file_path.name == "model.json":
        raise ValueError("model.json 대신 학습 설정 JSON을 지정하세요.")
    # 설정 파일·기본 평가 조건
    user_settings = json.loads(configuration_file_path.read_text(encoding="utf-8-sig"))
    asteroid_terrain_config = TerrainConfig(enabled=False)
    asteroid_game_config = GameConfig(
        num_enemies=0,
        terrain=asteroid_terrain_config,
    )
    combat_game_config = GameConfig(num_enemies=3)
    default_settings = {
        # 환경
        "description": "", 
        "game": asdict(GameConfig()), 
        "reward": asdict(RewardConfig()),
        "environment": asdict(EnvConfig()), 
        "observation": {"type": "vector", **asdict(ObsConfig())},
        "action": {
            "type": "continuous",
            "enable_stat_upgrades": True,
            "enable_evolution": True,
        },
        # 모델
        "training": asdict(TrainingConfig()),
        "ppo": PPO_DEFAULTS,
        # 평가
        "evaluation": {
            "seeds": [42, 43, 44],
            "environment": asdict(EnvConfig()),
            "reward": asdict(RewardConfig()),
            "scenarios": {
                "asteroid_only": asdict(asteroid_game_config),
                "combat": asdict(combat_game_config),
            },
        },
    }

    # 기본값 병합
    merged_settings = json.loads(json.dumps(merge_settings(default_settings, user_settings), allow_nan=False))
    
    training_values = merged_settings["training"]

    # 입력 검증
    for training_setting_name in ("additional_steps", "num_envs", "checkpoint_interval_steps"):
        if type(training_values[training_setting_name]) is not int or training_values[training_setting_name] <= 0:
            raise ValueError(f"training.{training_setting_name}는 양의 정수여야 합니다.")
    if type(training_values["parallel"]) is not bool:
        raise ValueError("training.parallel은 bool이어야 합니다.")
    if type(training_values["seed"]) is not int or not 0 <= training_values["seed"] < 2**32 - training_values["num_envs"]:
        raise ValueError("training.seed는 환경 수를 더해도 32비트 범위 안인 음이 아닌 정수여야 합니다.")
    if merged_settings["observation"]["type"] not in ("vector", "pixel"):
        raise ValueError("observation.type은 vector 또는 pixel이어야 합니다.")
    if merged_settings["action"]["type"] not in ("continuous", "discrete"):
        raise ValueError("action.type은 continuous 또는 discrete여야 합니다.")
    evaluation_seeds = merged_settings["evaluation"]["seeds"]
    if not isinstance(evaluation_seeds, list) or not evaluation_seeds or any(type(evaluation_seed) is not int or evaluation_seed < 0 for evaluation_seed in evaluation_seeds):
        raise ValueError("evaluation.seeds에는 음이 아닌 정수가 하나 이상 필요합니다.")

    
    # 단일 환경 설정
    if not training_values["parallel"]:
        training_values["num_envs"] = 1

    # 관찰·행동 설정
    observation_values = dict(merged_settings["observation"])
    observation_type = observation_values.pop("type")
    observation_config = ObsConfig(**observation_values)
    observation_settings = ObservationSettings(
        observation_type=observation_type,
        sensor_config=observation_config,
    )
    action_settings = ActionSettings(
        action_type=merged_settings["action"]["type"],
        enable_stat_upgrades=merged_settings["action"]["enable_stat_upgrades"],
        enable_evolution=merged_settings["action"]["enable_evolution"],
    )
    # 평가 설정
    evaluation_values = merged_settings["evaluation"]
    evaluation_environment_config = EnvConfig(**evaluation_values["environment"])
    evaluation_reward_config = RewardConfig(**evaluation_values["reward"])
    evaluation_scenarios = {
        scenario_name: game_configuration(game_values)
        for scenario_name, game_values in evaluation_values["scenarios"].items()
    }
    evaluation_config = EvaluationConfig(
        seeds=evaluation_values["seeds"],
        environment=evaluation_environment_config,
        reward=evaluation_reward_config,
        scenarios=evaluation_scenarios,
    )
    # 실험 설정 조립
    game_config = game_configuration(merged_settings["game"])
    reward_config = RewardConfig(**merged_settings["reward"])
    environment_config = EnvConfig(**merged_settings["environment"])
    training_config = TrainingConfig(**training_values)
    return ExperimentConfig(
        description=merged_settings["description"],
        game=game_config,
        reward=reward_config,
        environment=environment_config,
        observation=observation_settings,
        action=action_settings,
        training=training_config,
        ppo_overrides=user_settings.get("ppo", {}),
        evaluation=evaluation_config,
    )


def evaluation_to_dict(evaluation_config: EvaluationConfig) -> dict[str, Any]:
    """평가 설정을 기존 model.json 형식으로 변환한다."""
    return {
        "seeds": list(evaluation_config.seeds),
        "environment": asdict(evaluation_config.environment),
        "reward": asdict(evaluation_config.reward),
        "scenarios": {
            scenario_name: asdict(game_config)
            for scenario_name, game_config in evaluation_config.scenarios.items()
        },
    }


def configuration_to_dict(experiment_config: ExperimentConfig) -> dict[str, Any]:
    """내부 데이터클래스를 기존 설정 JSON 구조로 변환한다."""
    # 저장할 관찰·PPO 값
    observation_settings_for_json = asdict(experiment_config.observation.sensor_config)
    observation_settings_for_json["type"] = experiment_config.observation.observation_type
    ppo_parameters_for_json = experiment_config.resolved_ppo_parameters
    if ppo_parameters_for_json is None:
        ppo_parameters_for_json = experiment_config.ppo_overrides

    # 기존 JSON 구조
    return {
        "description": experiment_config.description,
        "game": asdict(experiment_config.game),
        "reward": asdict(experiment_config.reward),
        "environment": asdict(experiment_config.environment),
        "observation": observation_settings_for_json,
        "action": {
            "type": experiment_config.action.action_type,
            "enable_stat_upgrades": experiment_config.action.enable_stat_upgrades,
            "enable_evolution": experiment_config.action.enable_evolution,
        },
        "training": asdict(experiment_config.training),
        "ppo": dict(ppo_parameters_for_json),
        "evaluation": evaluation_to_dict(experiment_config.evaluation),
    }


def game_configuration(game_values: dict[str, Any]) -> GameConfig:
    """중첩 JSON을 실제 게임·지형·적 AI 설정 객체로 만든다."""
    game_values = deepcopy(game_values)
    game_values["terrain"] = TerrainConfig(**game_values["terrain"])
    game_values["enemy_ai"] = EnemyAIConfig(**game_values["enemy_ai"])
    return GameConfig(**game_values)


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
):
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


def reward_components(combat_events: CombatEvents, reward_config: RewardConfig) -> RewardBreakdown:
    """현재 RewardCalculator 공식에 따라 최종 보상을 항목별로 분해한다."""
    return RewardBreakdown(
        shape_damage=float(reward_config.shape_damage * combat_events["shape_damage_weighted"]),
        shape_destroy=float(reward_config.shape_destroy * combat_events["shape_destroyed_score"]),
        enemy_damage=float(reward_config.enemy_damage * combat_events["enemy_damage"]),
        enemy_kill=float(reward_config.enemy_kill * combat_events["enemies_killed"]),
        damage_taken=float(reward_config.damage_taken * combat_events["damage_taken"]),
        death=float(reward_config.death * combat_events["died"]),
        time_step=reward_config.time_step,
        shot_cost=-reward_config.shot_cost * combat_events["player_shots_fired"],
    )


class LoggedPPO(PPO):
    """PPO 계산은 유지하고 갱신 직후 실제 step으로 학습 지표를 내보낸다."""

    def train(self) -> None:
        super().train()
        self.logger.dump(self.num_timesteps)


class TrainingLog(BaseCallback):
    """메인 프로세스에서 모든 환경의 step·에피소드 원본과 TensorBoard 지표를 기록한다."""

    def __init__(
        self,
        run_directory: Path,
        reward_config: RewardConfig,
        num_envs: int,
    ) -> None:
        super().__init__()
        self.reward_config = reward_config
        self.step_log_file = (run_directory / "steps.jsonl").open("a", encoding="utf-8")
        self.episode_log_file = (run_directory / "episodes.jsonl").open("a", encoding="utf-8")
        self.episode_totals_by_env: list[Counter[str]] = [
            Counter() for _ in range(num_envs)
        ]

    def _on_step(self) -> bool:
        # 환경별 실행 결과
        environment_results = zip(
            self.locals["infos"],
            self.locals["rewards"],
            self.locals["dones"],
        )
        for environment_index, environment_result in enumerate(environment_results):
            environment_info, step_reward, episode_finished = environment_result

            episode_step_record: EpisodeStepRecord = environment_info["episode_step_record"]
            combat_events: CombatEvents = environment_info["events"]

            # 보상 분해·검증
            reward_breakdown = reward_components(combat_events, self.reward_config)
            reward_component_values = asdict(reward_breakdown)
            assert math.isclose(
                sum(reward_component_values.values()), float(step_reward), rel_tol=1e-5, abs_tol=1e-5
            )
            # step 로그 저장
            logged_environment_info = {
                field_name: field_value
                for field_name, field_value in environment_info.items()
                if field_name not in ("episode_step_record", "terminal_observation")
            }
            step_log_record: StepLogRecord = {
                "global_step": self.num_timesteps,
                "env_id": environment_index,
                "episode_id": episode_step_record["episode_id"],
                "seed": episode_step_record["episode_seed"],
                "action": episode_step_record["policy_action"],
                "control": episode_step_record["player_control"],
                "player": episode_step_record["player_snapshot"],
                "terminated": episode_step_record["terminated"],
                "truncated": episode_step_record["truncated"],
                "reward": float(step_reward),
                "reward_components": reward_component_values,
                "info": logged_environment_info,
            }
            append_record(self.step_log_file, step_log_record)

            # 에피소드 누적값
            episode_totals = self.episode_totals_by_env[environment_index]
            step_totals = {"total_reward": float(step_reward)}
            for reward_name, reward_value in reward_component_values.items():
                step_totals[f"reward/{reward_name}"] = reward_value
            for event_name, event_value in combat_events.items():
                step_totals[f"event/{event_name}"] = event_value
            episode_totals.update(step_totals)

            # step 지표
            self.logger.record_mean("state/hp", episode_step_record["player_snapshot"]["hp"])
            self.logger.record_mean(
                "action/fire_requested", float(episode_step_record["player_control"]["want_fire"])
            )
            for reward_name, reward_value in reward_component_values.items():
                self.logger.record_mean(f"reward_step/{reward_name}", reward_value)
            # 에피소드 종료 기록
            if episode_finished:
                episode_log_record = dict(step_log_record)
                episode_log_record["episode_totals"] = dict(episode_totals)
                append_record(self.episode_log_file, episode_log_record)

                episode_metric_values = dict(episode_totals)
                episode_metric_values["score"] = environment_info["score"]
                episode_metric_values["kills"] = environment_info["kills"]
                episode_metric_values["level"] = environment_info["level"]
                for metric_name, metric_value in episode_metric_values.items():
                    self.logger.record_mean(f"episode/{metric_name}", metric_value)
                episode_totals.clear()
        return True

    def _on_rollout_end(self) -> None:
        self.step_log_file.flush()
        self.episode_log_file.flush()

    def close(self) -> None:
        self.step_log_file.close()
        self.episode_log_file.close()


def prepare_session(configuration_file_path: Path, experiment_config: ExperimentConfig) -> RunSession:
    """대표 JSON에서 부모를 찾고 다음 번호의 폴더와 실행 이력을 생성한다."""
    # 모델 이력 경로
    model_directory = configuration_file_path.parent
    model_registry_path = model_directory / "model.json"
    evaluation_settings_for_registry = evaluation_to_dict(experiment_config.evaluation)

    # 모델 이력 로드·검증
    if model_registry_path.exists():
        model_registry: ModelRegistry = json.loads(model_registry_path.read_text(encoding="utf-8"))
    else:
        model_registry = {
            "model_id": model_directory.name,
            "description": experiment_config.description,
            "algorithm": "ppo",
            "latest_checkpoint": None,
            "evaluation": evaluation_settings_for_registry,
            "training_sessions": [],
        }
    model_registry.setdefault("evaluation", evaluation_settings_for_registry)
    if model_registry["algorithm"] != "ppo" or model_registry["evaluation"] != evaluation_settings_for_registry:
        raise ValueError("같은 모델의 알고리즘·평가 조건은 고정합니다. 다른 조건은 새 모델 폴더를 사용하세요.")
    # 부모 체크포인트 확인
    parent_checkpoint_path = None
    if model_registry["latest_checkpoint"]:
        parent_checkpoint_path = (
            model_directory / model_registry["latest_checkpoint"]
        ).resolve()
    if parent_checkpoint_path is not None and (
        not parent_checkpoint_path.is_relative_to(model_directory)
        or not parent_checkpoint_path.is_file()
    ):
        raise ValueError("model.json의 최근 체크포인트 경로가 올바르지 않습니다.")

    # 다음 실행 번호
    existing_run_numbers = [
        int(existing_directory.name.split("_", 1)[0])
        for existing_directory in model_directory.iterdir()
        if existing_directory.is_dir() and existing_directory.name.split("_", 1)[0].isdigit()
    ]
    registered_run_numbers = [
        int(training_run_record["id"]) for training_run_record in model_registry["training_sessions"]
    ]
    next_run_number = max(registered_run_numbers + existing_run_numbers + [0]) + 1

    # 실행 폴더 생성
    run_directory = model_directory / f"{next_run_number:03d}_{configuration_file_path.stem}"
    run_directory.mkdir()

    for subdirectory_name in ("checkpoints", "evaluations"):
        (run_directory / subdirectory_name).mkdir()

    # 실행 이력 등록
    training_run_record: TrainingRunRecord = {
        "id": next_run_number,
        "folder": run_directory.name,
        "source_config": configuration_file_path.name,
        "parent_checkpoint": model_registry["latest_checkpoint"],
        "start_step": None,
        "end_step": None,
        "status": "running", 
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    model_registry["training_sessions"].append(training_run_record)
    write_json(model_registry_path, model_registry)

    return RunSession(
        model_directory=model_directory,
        run_directory=run_directory,
        parent_checkpoint_path=parent_checkpoint_path,
        model_registry=model_registry,
        training_run_record=training_run_record,
    )


@contextmanager
def session_status(run_session: RunSession):
    """완료·중단·오류를 대표 JSON에 남긴다. 이미 저장된 부모 체크포인트는 보존한다."""
    training_run_record = run_session.training_run_record
    try:
        yield
    except KeyboardInterrupt:
        training_run_record["status"] = "interrupted"
        print("중단했습니다. 다음 실행은 최근 저장 완료 체크포인트에서 시작합니다.")
    except Exception as error:
        training_run_record["status"] = "failed"
        training_run_record["error"] = str(error)
        raise
    else:
        training_run_record["status"] = "completed"
    finally:
        training_run_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_session.model_directory / "model.json", run_session.model_registry)


def create_model(experiment_config: ExperimentConfig, training_environments, run_session: RunSession) -> PPO:
    """부모의 가중치·Adam 상태를 복원하거나 새 PPO를 만든다. 명시한 PPO 항목만 덮어쓴다."""
    # 공통 실행 설정
    training_config = experiment_config.training
    model_seed = training_config.seed
    training_device = training_config.device
    tensorboard_path = str(run_session.run_directory / "tensorboard")
    # 새 모델 생성 또는 재개
    if run_session.parent_checkpoint_path:
        ppo_model = LoggedPPO.load(
            run_session.parent_checkpoint_path,
            env=training_environments,
            seed=model_seed,
            device=training_device,
            tensorboard_log=tensorboard_path,
            **experiment_config.ppo_overrides,
        )
    else:
        policy_class_name = (
            "MultiInputPolicy"
            if experiment_config.observation.observation_type == "pixel"
            else "MlpPolicy"
        )
        ppo_hyperparameters = PPO_DEFAULTS | experiment_config.ppo_overrides
        ppo_model = LoggedPPO(
            policy_class_name,
            env=training_environments,
            seed=model_seed,
            device=training_device,
            tensorboard_log=tensorboard_path,
            verbose=1,
            **ppo_hyperparameters,
        )
    # 실제 적용값 기록
    run_session.training_run_record["start_step"] = ppo_model.num_timesteps
    run_session.training_run_record["end_step"] = ppo_model.num_timesteps
    resolved_ppo_parameters: PPOOverrides = {}
    for parameter_name in PPO_DEFAULTS:
        parameter_value = getattr(ppo_model, parameter_name)
        if callable(parameter_value):
            parameter_value = parameter_value(1.0)
        resolved_ppo_parameters[parameter_name] = parameter_value
    experiment_config.resolved_ppo_parameters = resolved_ppo_parameters
    return ppo_model


def save_configuration(experiment_config: ExperimentConfig, run_session: RunSession) -> None:
    """기본값·부모 설정까지 반영한 설정, 라이브러리 버전, 실행 소스 사본을 보존한다."""
    # 소스 사본 저장
    project_root = Path(__file__).resolve().parents[3]
    source_archive_path = run_session.run_directory / "source_code.zip"
    source_files = sorted((project_root / "src/space_z").rglob("*.py"))
    map_files = sorted((project_root / "maps").glob("*.txt"))
    with ZipFile(source_archive_path, "w", ZIP_DEFLATED) as source_archive:
        for source_path in source_files + map_files:
            source_archive.write(source_path, source_path.relative_to(project_root))

    # 실행 설정·이력
    training_run_record = run_session.training_run_record
    saved_configuration = configuration_to_dict(experiment_config)
    saved_configuration["run"] = {
        "id": training_run_record["id"],
        "folder": training_run_record["folder"],
        "source_config": training_run_record["source_config"],
        "parent_checkpoint": training_run_record["parent_checkpoint"],
        "start_step": training_run_record["start_step"],
        "started_at": training_run_record["started_at"],
    }
    # 소스 해시·라이브러리 버전
    saved_configuration["source_sha256"] = hashlib.sha256(
        source_archive_path.read_bytes()
    ).hexdigest()
    saved_configuration["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
    }
    write_json(run_session.run_directory / "config.json", saved_configuration)


def save_checkpoint(ppo_model: PPO, run_session: RunSession) -> Path:
    """저장 완료한 ZIP만 최근 체크포인트로 등록한다. 이름의 step은 실제 누적값이다."""
    # 모델 파일 저장·확정
    checkpoint_path = run_session.run_directory / "checkpoints" / f"step_{ppo_model.num_timesteps:08d}.zip"
    temporary_checkpoint_path = checkpoint_path.with_suffix(".tmp.zip")
    ppo_model.save(temporary_checkpoint_path)
    temporary_checkpoint_path.replace(checkpoint_path)
    # 최근 체크포인트 등록
    checkpoint_relative_path = checkpoint_path.relative_to(run_session.model_directory).as_posix()
    run_session.training_run_record["end_step"] = ppo_model.num_timesteps
    run_session.training_run_record["latest_checkpoint"] = checkpoint_relative_path
    run_session.model_registry["latest_checkpoint"] = checkpoint_relative_path
    write_json(run_session.model_directory / "model.json", run_session.model_registry)
    return checkpoint_path


def evaluate_checkpoint(
    ppo_model: PPO,
    experiment_config: ExperimentConfig,
    run_session: RunSession,
    checkpoint_path: Path,
) -> None:
    """학습 보상과 독립된 고정 시나리오·보상·시드로 평가하고 결과를 누적한다."""
    def select_evaluation_action(observation: Any) -> Any:
        policy_prediction = ppo_model.predict(
            observation,
            deterministic=True,
        )
        return policy_prediction[0]

    # 시나리오별 평가
    evaluation_config = experiment_config.evaluation
    evaluation_log_path = run_session.run_directory / "evaluations/results.jsonl"
    with evaluation_log_path.open("a", encoding="utf-8") as evaluation_log_file:
        for scenario_name, scenario_game_config in evaluation_config.scenarios.items():
            evaluation_settings = replace(
                experiment_config,
                game=scenario_game_config,
                reward=evaluation_config.reward,
                environment=evaluation_config.environment,
            )
            with closing(make_environment(evaluation_settings, record_episode_steps=False)) as evaluation_environment:
                policy_evaluator = PolicyEvaluator(
                    environment=evaluation_environment,
                    action_selector=select_evaluation_action,
                )
                evaluation_report = policy_evaluator.evaluate(evaluation_config.seeds)
            # 평가 결과 저장
            evaluation_record = {
                "global_step": ppo_model.num_timesteps,
                "checkpoint": checkpoint_path.name,
                "scenario": scenario_name,
                "deterministic": True,
                "episodes": evaluation_report["episodes"],
                "summary": evaluation_report["summary"],
                "resources": evaluation_report["resources"],
            }
            append_record(evaluation_log_file, evaluation_record)
            # TensorBoard 지표
            for metric_name, metric_statistics in evaluation_report["summary"].items():
                ppo_model.logger.record(
                    f"evaluation/{scenario_name}/{metric_name}", metric_statistics["mean"]
                )
    ppo_model.logger.dump(ppo_model.num_timesteps)


def train_with_checkpoints(ppo_model: PPO, experiment_config: ExperimentConfig, run_session: RunSession) -> None:
    """학습 → 정책 갱신 완료 시점의 저장 → 평가를 목표 step까지 반복한다."""
    # 학습 범위·로그 준비
    training_config = experiment_config.training
    training_end_step = ppo_model.num_timesteps + training_config.additional_steps
    training_log_callback = TrainingLog(
        run_directory=run_session.run_directory,
        reward_config=experiment_config.reward,
        num_envs=ppo_model.n_envs,
    )
    with closing(training_log_callback):
        try:
            while ppo_model.num_timesteps < training_end_step:
                # 체크포인트 단위 학습
                remaining_training_steps = training_end_step - ppo_model.num_timesteps
                steps_until_checkpoint = min(
                    training_config.checkpoint_interval_steps, remaining_training_steps
                )
                ppo_model.learn(
                    steps_until_checkpoint,
                    callback=training_log_callback,
                    tb_log_name="training",
                    reset_num_timesteps=False,
                )
                # 저장·평가
                checkpoint_path = save_checkpoint(ppo_model, run_session)
                evaluate_checkpoint(ppo_model, experiment_config, run_session, checkpoint_path)
                ppo_model.logger.close()
                print(f"저장·평가 완료: {checkpoint_path}", flush=True)
        finally:
            if hasattr(ppo_model, "_logger"):
                ppo_model.logger.close()


def configuration_path() -> Path:
    """직접 작성한 설정 JSON 경로를 실행 인자로 받는다."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--config", type=Path, required=True, help="모델 폴더 안의 학습 설정 JSON")
    return argument_parser.parse_args().config.resolve()


def main() -> None:
    """설정 읽기 → 실행 등록 → 환경·모델 준비 → 설정 보존 → 기록하며 학습·저장·평가."""
    # 설정 로드
    configuration_file_path = configuration_path()
    experiment_config = read_configuration(configuration_file_path)

    # 실행 등록
    run_session = prepare_session(configuration_file_path, experiment_config)
    # shortcut: 같은 모델 폴더의 동시 학습 실행은 지원하지 않는다. 필요하면 대표 JSON 잠금을 추가한다.
    with session_status(run_session), closing(make_training_environments(experiment_config)) as training_environments:
        # 모델 준비·설정 보존
        ppo_model = create_model(experiment_config, training_environments, run_session)
        save_configuration(experiment_config, run_session)

        # 학습 실행
        train_with_checkpoints(ppo_model, experiment_config, run_session)


if __name__ == "__main__":
    main()
