from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor

from space_z.rl import PolicyEvaluator, VectorObservation

from .configuration import ExperimentConfig, PPOOverrides, PPO_DEFAULTS
from .environments import make_environment
from .feature_extractors import TerrainGridFeaturesExtractor
from .run_session import RunSession, save_checkpoint
from .training_log import TrainingLog, append_record


class LoggedPPO(PPO):
    """PPO 계산은 유지하고 갱신 직후 실제 step으로 학습 지표를 내보낸다."""

    def train(self) -> None:
        super().train()
        self.logger.dump(self.num_timesteps)


def build_policy_kwargs(experiment_config: ExperimentConfig) -> dict[str, Any]:
    """검증된 설정과 관찰 구조를 SB3 정책 생성 인자로 변환한다."""
    feature_extractor_settings = experiment_config.feature_extractor
    if feature_extractor_settings.extractor_type == "default":
        return {}

    observation_builder = VectorObservation(experiment_config.observation.sensor_config)
    features_extractor_kwargs = {
        "terrain_grid_slice": observation_builder.slices["grid"],
        "terrain_grid_shape": observation_builder.grid_shape,
        "terrain_feature_count": feature_extractor_settings.terrain_feature_count,
    }
    return {
        "features_extractor_class": TerrainGridFeaturesExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
    }


def validate_policy_configuration(ppo_model: PPO, experiment_config: ExperimentConfig) -> None:
    """복원된 모델의 특징 추출기와 이번 실행의 요청이 일치하는지 확인한다."""
    policy_kwargs = build_policy_kwargs(experiment_config)
    default_extractor_class = (
        CombinedExtractor
        if experiment_config.observation.observation_type == "pixel"
        else FlattenExtractor
    )
    expected_extractor_class = policy_kwargs.get("features_extractor_class", default_extractor_class)
    restored_extractor = ppo_model.policy.features_extractor
    if type(restored_extractor) is not expected_extractor_class:
        raise ValueError("체크포인트의 특징 추출기가 policy 설정과 다릅니다. 구조 변경은 새 모델 폴더에서 시작하세요.")

    for parameter_name, expected_value in policy_kwargs.get("features_extractor_kwargs", {}).items():
        if getattr(restored_extractor, parameter_name) != expected_value:
            raise ValueError(f"체크포인트의 {parameter_name}이 현재 설정과 다릅니다. 새 모델 폴더를 사용하세요.")


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
        validate_policy_configuration(ppo_model, experiment_config)
    else:
        policy_class_name = (
            "MultiInputPolicy"
            if experiment_config.observation.observation_type == "pixel"
            else "MlpPolicy"
        )
        ppo_hyperparameters = PPO_DEFAULTS | experiment_config.ppo_overrides
        policy_kwargs = build_policy_kwargs(experiment_config)
        ppo_model = LoggedPPO(
            policy_class_name,
            env=training_environments,
            seed=model_seed,
            device=training_device,
            tensorboard_log=tensorboard_path,
            verbose=1,
            policy_kwargs=policy_kwargs,
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


def evaluate_checkpoint(
    ppo_model: PPO,
    experiment_config: ExperimentConfig,
    run_session: RunSession,
    checkpoint_path: Path,
) -> None:
    """학습 보상과 독립된 고정 시나리오·보상·시드로 평가하고 결과를 누적한다."""
    def select_evaluation_action(observation: Any) -> Any:
        action, _ = ppo_model.predict(
            observation,
            deterministic=True,
        )
        if isinstance(ppo_model.action_space, gym.spaces.Discrete):
          return action.item()

        return action

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
