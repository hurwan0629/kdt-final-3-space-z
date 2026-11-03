from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, TextIO, TypedDict

from stable_baselines3.common.callbacks import BaseCallback

from space_z.config import RewardConfig
from space_z.game.types import CombatEvents

from .environments import ControlRecord, EpisodeStepRecord, PlayerSnapshot
from .run_session import json_default


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


def append_record(log_file: TextIO, log_record: dict[str, Any]) -> None:
    """메인 프로세스가 JSONL에 원본 기록 한 줄을 추가한다."""
    serialized_log_record = json.dumps(
        log_record, ensure_ascii=False, default=json_default, allow_nan=False
    )
    log_file.write(serialized_log_record + "\n")


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
