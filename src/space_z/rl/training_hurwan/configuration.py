from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from space_z.config import EnemyAIConfig, EnvConfig, GameConfig, ObsConfig, RewardConfig, TerrainConfig


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


@dataclass(frozen=True)
class FeatureExtractorSettings:
    """정책의 특징 추출 방식과 지형 CNN의 출력 크기를 설정한다."""

    extractor_type: Literal["default", "terrain_cnn"] = "default"
    terrain_feature_count: int = 64


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
    feature_extractor: FeatureExtractorSettings = field(default_factory=FeatureExtractorSettings)


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
    default_feature_extractor = FeatureExtractorSettings()
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
        "policy": {
            "feature_extractor": {
                "type": default_feature_extractor.extractor_type,
                "terrain_feature_count": default_feature_extractor.terrain_feature_count,
            },
        },
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
    feature_extractor_values = merged_settings["policy"]["feature_extractor"]
    extractor_type = feature_extractor_values["type"]
    if extractor_type not in ("default", "terrain_cnn"):
        raise ValueError("policy.feature_extractor.type은 default 또는 terrain_cnn이어야 합니다.")
    terrain_feature_count = feature_extractor_values["terrain_feature_count"]
    if type(terrain_feature_count) is not int or terrain_feature_count <= 0:
        raise ValueError("policy.feature_extractor.terrain_feature_count는 양의 정수여야 합니다.")
    if extractor_type == "terrain_cnn" and merged_settings["observation"]["type"] != "vector":
        raise ValueError("terrain_cnn은 vector 관찰에서만 사용할 수 있습니다.")
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
    feature_extractor_settings = FeatureExtractorSettings(
        extractor_type=extractor_type,
        terrain_feature_count=terrain_feature_count,
    )
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
        feature_extractor=feature_extractor_settings,
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
        "policy": {
            "feature_extractor": {
                "type": experiment_config.feature_extractor.extractor_type,
                "terrain_feature_count": experiment_config.feature_extractor.terrain_feature_count,
            },
        },
        "evaluation": evaluation_to_dict(experiment_config.evaluation),
    }


def game_configuration(game_values: dict[str, Any]) -> GameConfig:
    """중첩 JSON을 실제 게임·지형·적 AI 설정 객체로 만든다."""
    game_values = deepcopy(game_values)
    game_values["terrain"] = TerrainConfig(**game_values["terrain"])
    game_values["enemy_ai"] = EnemyAIConfig(**game_values["enemy_ai"])
    return GameConfig(**game_values)
