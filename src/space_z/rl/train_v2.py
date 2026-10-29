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
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from space_z.config import EnemyAIConfig, EnvConfig, GameConfig, ObsConfig, RewardConfig, TerrainConfig
from space_z.rl import ContinuousActionDecoder, DiscreteActionDecoder, PixelObservation
from space_z.rl import PolicyEvaluator, RewardCalculator, TankEnv, VectorObservation


PPO_DEFAULTS = {"n_steps": 256, "batch_size": 64, "n_epochs": 10, "learning_rate": 0.0003,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "clip_range_vf": None,
                "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
                "normalize_advantage": True, "target_kl": None}
REWARD_EVENTS = {"shape_damage": "shape_damage_weighted", "shape_destroy": "shape_destroyed_score",
                 "enemy_damage": "enemy_damage", "enemy_kill": "enemies_killed",
                 "damage_taken": "damage_taken", "death": "died"}


def json_default(value: Any) -> Any:
    """NumPy 수치·배열과 경로만 JSON의 기본 자료형으로 변환한다."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON으로 저장할 수 없는 값: {type(value).__name__}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """완성된 임시 파일을 교체해 대표 JSON의 부분 쓰기를 막는다."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_record(stream: Any, data: dict[str, Any]) -> None:
    """메인 프로세스가 JSONL에 원본 기록 한 줄을 추가한다."""
    stream.write(json.dumps(data, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")


def merge_settings(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """기본값을 채우되 알 수 없는 키는 오타로 처리한다."""
    if not isinstance(overrides, dict):
        raise ValueError("설정 항목은 JSON 객체여야 합니다.")
    result = deepcopy(defaults)
    for key, value in overrides.items():
        if key not in result:
            raise ValueError(f"알 수 없는 설정: {key}")
        result[key] = merge_settings(result[key], value) if isinstance(result[key], dict) else value
    return result


def read_configuration(path: Path) -> dict[str, Any]:
    """사용자 설정을 검증하고 기본값을 채운다. 생략한 PPO 값은 재개 시 부모에서 상속한다."""
    if path.name == "model.json":
        raise ValueError("model.json 대신 학습 설정 JSON을 지정하세요.")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    defaults = {
        # 환경
        "description": "", 
        "game": asdict(GameConfig()), 
        "reward": asdict(RewardConfig()),
        "environment": asdict(EnvConfig()), 
        "observation": {"type": "vector", **asdict(ObsConfig())},
        "action": {"type": "continuous", "enable_stat_upgrades": True, "enable_evolution": True},
        # 모델
        "training": {"additional_steps": 10000, "seed": 42, "device": "cpu", "parallel": False,
                     "num_envs": 4, "checkpoint_interval_steps": 10000},
        "ppo": PPO_DEFAULTS,
        # 평가
        "evaluation": {
            "seeds": [42, 43, 44], "environment": asdict(EnvConfig()), "reward": asdict(RewardConfig()),
            "scenarios": {
                "asteroid_only": asdict(GameConfig(num_enemies=0, terrain=TerrainConfig(enabled=False))),
                "combat": asdict(GameConfig(num_enemies=3)),
            },
        },
    }

    # 설정 덮어씌우기 (일부만 채워도 되지만 defualts에 없는 키는 안됨)
    config = json.loads(json.dumps(merge_settings(defaults, raw), allow_nan=False))
    
    training = config["training"]

    # 유효성 검사
    for key in ("additional_steps", "num_envs", "checkpoint_interval_steps"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key}는 양의 정수여야 합니다.")
    if type(training["parallel"]) is not bool:
        raise ValueError("training.parallel은 bool이어야 합니다.")
    if type(training["seed"]) is not int or not 0 <= training["seed"] < 2**32 - training["num_envs"]:
        raise ValueError("training.seed는 환경 수를 더해도 32비트 범위 안인 음이 아닌 정수여야 합니다.")
    if config["observation"]["type"] not in ("vector", "pixel"):
        raise ValueError("observation.type은 vector 또는 pixel이어야 합니다.")
    if config["action"]["type"] not in ("continuous", "discrete"):
        raise ValueError("action.type은 continuous 또는 discrete여야 합니다.")
    seeds = config["evaluation"]["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("evaluation.seeds에는 음이 아닌 정수가 하나 이상 필요합니다.")

    
    config["ppo"] = raw.get("ppo", {})
    if not training["parallel"]:
        training["num_envs"] = 1
    return config


def game_configuration(values: dict[str, Any]) -> GameConfig:
    """중첩 JSON을 실제 게임·지형·적 AI 설정 객체로 만든다."""
    values = deepcopy(values)
    values["terrain"] = TerrainConfig(**values["terrain"])
    values["enemy_ai"] = EnemyAIConfig(**values["enemy_ai"])
    return GameConfig(**values)


class RecordedState(gym.Wrapper):
    """환경을 변경하지 않고 실제 에피소드 시드·행동 후 상태를 info에 추가한다."""

    def __init__(self, environment: TankEnv, seed: int, stride: int) -> None:
        super().__init__(environment)
        self.next_seed, self.stride, self.episode_id = seed, stride, 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_seed = self.next_seed if seed is None else int(seed)
        self.next_seed = (self.episode_seed + self.stride) % 2**32
        self.episode_id += 1
        return self.env.reset(seed=self.episode_seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        player = self.unwrapped.game.player
        info["record"] = {
            "episode_id": self.episode_id,
            "seed": self.episode_seed,
            "action": np.asarray(action).tolist(), 
            "control": asdict(self.unwrapped.action_decoder.decode(action)),
            "player": {
                key: getattr(player, key) 
                for key in 
                ("x", "y", "velocity_x", "velocity_y", "angle", "hp", "max_hp")},
            "terminated": terminated, 
            "truncated": truncated,
        }
        return observation, reward, terminated, truncated, info


def make_environment(config: dict[str, Any], rank: int = 0, record: bool = True):
    """관찰·행동·보상 전략을 조립한다. 병렬 환경도 이 함수를 프로세스 안에서 호출한다."""
    observation = dict(config["observation"])
    observer = {"vector": VectorObservation, "pixel": PixelObservation}[observation.pop("type")](ObsConfig(**observation))
    action = dict(config["action"])
    decoder = {"continuous": ContinuousActionDecoder, "discrete": DiscreteActionDecoder}[action.pop("type")](**action)
    environment = TankEnv(game_configuration(config["game"]), observer, decoder,
                          RewardCalculator(RewardConfig(**config["reward"])), EnvConfig(**config["environment"]))
    if not record:
        return environment
    training = config["training"]
    return Monitor(RecordedState(environment, training["seed"] + rank, training["num_envs"]))


def make_training_environments(config: dict[str, Any]):
    """개별 Monitor를 가진 단일 또는 다중 프로세스 환경을 준비한다."""
    factories = [partial(make_environment, config, rank) for rank in range(config["training"]["num_envs"])]
    if config["training"]["parallel"]:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def reward_components(events: dict[str, Any], weights: dict[str, Any]) -> dict[str, float]:
    """현재 RewardCalculator 공식에 따라 최종 보상을 항목별로 분해한다."""
    parts = {
      key: float(weights[key] * events[event]) 
      for key, event
      in REWARD_EVENTS.items()
    }
    return {
      **parts, 
      "time_step": weights["time_step"], 
      "shot_cost": -weights["shot_cost"] * events["player_shots_fired"]
    }


class LoggedPPO(PPO):
    """PPO 계산은 유지하고 갱신 직후 실제 step으로 학습 지표를 내보낸다."""

    def train(self) -> None:
        super().train()
        self.logger.dump(self.num_timesteps)


class TrainingLog(BaseCallback):
    """메인 프로세스에서 모든 환경의 step·에피소드 원본과 TensorBoard 지표를 기록한다."""

    def __init__(self, folder: Path, weights: dict[str, Any], count: int) -> None:
        super().__init__()
        self.weights = weights
        self.steps = (folder / "steps.jsonl").open("a", encoding="utf-8")
        self.episodes = (folder / "episodes.jsonl").open("a", encoding="utf-8")
        self.totals = [Counter() for _ in range(count)]

    def _on_step(self) -> bool:
        for rank, (info, reward, done) in enumerate(zip(self.locals["infos"], self.locals["rewards"], self.locals["dones"])):
            state = info["record"]
            parts = reward_components(info["events"], self.weights)
            assert math.isclose(sum(parts.values()), float(reward), rel_tol=1e-5, abs_tol=1e-5)
            clean_info = {k: v for k, v in info.items() if k not in ("record", "terminal_observation")}
            row = {"global_step": self.num_timesteps, "env_id": rank, **state,
                   "reward": float(reward), "reward_components": parts, "info": clean_info}
            append_record(self.steps, row)
            totals = self.totals[rank]
            totals.update({"total_reward": float(reward), **{f"reward/{k}": v for k, v in parts.items()},
                           **{f"event/{k}": v for k, v in info["events"].items()}})
            self.logger.record_mean("state/hp", state["player"]["hp"])
            self.logger.record_mean("action/fire_requested", float(state["control"]["want_fire"]))
            for key, value in parts.items():
                self.logger.record_mean(f"reward_step/{key}", value)
            if done:
                append_record(self.episodes, {**row, "episode_totals": dict(totals)})
                for key, value in {**totals, "score": info["score"], "kills": info["kills"], "level": info["level"]}.items():
                    self.logger.record_mean(f"episode/{key}", value)
                totals.clear()
        return True

    def _on_rollout_end(self) -> None:
        self.steps.flush()
        self.episodes.flush()

    def close(self) -> None:
        self.steps.close()
        self.episodes.close()


def prepare_session(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """대표 JSON에서 부모를 찾고 다음 번호의 폴더와 실행 이력을 생성한다."""
    # 이전에 작업하던 모델이 있는지 확인하기
    model_dir = path.parent
    registry_path = model_dir / "model.json"

    # 파일 읽기 또는 생성하기
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {
        "model_id": model_dir.name, "description": config["description"], "algorithm": "ppo",
        "latest_checkpoint": None, "evaluation": config["evaluation"], "training_sessions": [],
    }
    registry.setdefault("evaluation", config["evaluation"])
    if registry["algorithm"] != "ppo" or registry["evaluation"] != config["evaluation"]:
        raise ValueError("같은 모델의 알고리즘·평가 조건은 고정합니다. 다른 조건은 새 모델 폴더를 사용하세요.")
    parent = (model_dir / registry["latest_checkpoint"]).resolve() if registry["latest_checkpoint"] else None
    if parent is not None and (not parent.is_relative_to(model_dir) or not parent.is_file()):
        raise ValueError("model.json의 최근 체크포인트 경로가 올바르지 않습니다.")

    # runs/[model]/00n_... 형식의 문자열 읽어서 다음 폴더 생성할 준비
    identifiers = [int(p.name.split("_", 1)[0]) for p in model_dir.iterdir()
                   if p.is_dir() and p.name.split("_", 1)[0].isdigit()]
    number = max([int(s["id"]) for s in registry["training_sessions"]] + identifiers + [0]) + 1

    # 이번에 실행에 만들 폴더
    folder = model_dir / f"{number:03d}_{path.stem}"
    folder.mkdir()

    # 체크포인트와 평가 폴더 만들어주기
    for name in ("checkpoints", "evaluations"):
        (folder / name).mkdir()
    entry = {
        "id": number, 
        "folder": folder.name, 
        "source_config": path.name,
        "parent_checkpoint": registry["latest_checkpoint"], 
        "start_step": None, 
        "end_step": None,
        "status": "running", 
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    registry["training_sessions"].append(entry)
    write_json(registry_path, registry)

    return {"model_dir": model_dir, "folder": folder, "registry": registry, "entry": entry, "parent": parent}


@contextmanager
def session_status(session: dict[str, Any]):
    """완료·중단·오류를 대표 JSON에 남긴다. 이미 저장된 부모 체크포인트는 보존한다."""
    entry = session["entry"]
    try:
        yield
    except KeyboardInterrupt:
        entry["status"] = "interrupted"
        print("중단했습니다. 다음 실행은 최근 저장 완료 체크포인트에서 시작합니다.")
    except Exception as error:
        entry.update(status="failed", error=str(error))
        raise
    else:
        entry["status"] = "completed"
    finally:
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(session["model_dir"] / "model.json", session["registry"])


def create_model(config: dict[str, Any], environment, session: dict[str, Any]) -> PPO:
    """부모의 가중치·Adam 상태를 복원하거나 새 PPO를 만든다. 명시한 PPO 항목만 덮어쓴다."""
    training = config["training"]
    options = dict(env=environment, seed=training["seed"], device=training["device"],
                   tensorboard_log=str(session["folder"] / "tensorboard"))
    if session["parent"]:
        model = LoggedPPO.load(session["parent"], **options, **config["ppo"])
    else:
        policy = "MultiInputPolicy" if config["observation"]["type"] == "pixel" else "MlpPolicy"
        model = LoggedPPO(policy, **options, **(PPO_DEFAULTS | config["ppo"]), verbose=1)
    session["entry"].update(start_step=model.num_timesteps, end_step=model.num_timesteps)
    config["ppo"] = {key: getattr(model, key)(1.0) if callable(getattr(model, key)) else getattr(model, key)
                     for key in PPO_DEFAULTS}
    return model


def save_configuration(config: dict[str, Any], session: dict[str, Any]) -> None:
    """기본값·부모 설정까지 반영한 설정, 라이브러리 버전, 실행 소스 사본을 보존한다."""
    root = Path(__file__).resolve().parents[3]
    archive = session["folder"] / "source_code.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for path in sorted((root / "src/space_z").rglob("*.py")) + sorted((root / "maps").glob("*.txt")):
            zipped.write(path, path.relative_to(root))
    write_json(session["folder"] / "config.json", {
        **config, "run": {key: session["entry"][key] for key in
                          ("id", "folder", "source_config", "parent_checkpoint", "start_step", "started_at")},
        "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "gymnasium": gym.__version__, "numpy": np.__version__},
    })


def save_checkpoint(model: PPO, session: dict[str, Any]) -> Path:
    """저장 완료한 ZIP만 최근 체크포인트로 등록한다. 이름의 step은 실제 누적값이다."""
    path = session["folder"] / "checkpoints" / f"step_{model.num_timesteps:08d}.zip"
    temporary = path.with_suffix(".tmp.zip")
    model.save(temporary)
    temporary.replace(path)
    relative = path.relative_to(session["model_dir"]).as_posix()
    session["entry"].update(end_step=model.num_timesteps, latest_checkpoint=relative)
    session["registry"]["latest_checkpoint"] = relative
    write_json(session["model_dir"] / "model.json", session["registry"])
    return path


def evaluate_checkpoint(model: PPO, config: dict[str, Any], session: dict[str, Any], checkpoint: Path) -> None:
    """학습 보상과 독립된 고정 시나리오·보상·시드로 평가하고 결과를 누적한다."""
    evaluation = session["registry"]["evaluation"]
    with (session["folder"] / "evaluations/results.jsonl").open("a", encoding="utf-8") as stream:
        for name, game in evaluation["scenarios"].items():
            settings = {**config, "game": game, "reward": evaluation["reward"], "environment": evaluation["environment"]}
            with closing(make_environment(settings, record=False)) as environment:
                evaluator = PolicyEvaluator(environment, lambda obs: model.predict(obs, deterministic=True)[0])
                report = evaluator.evaluate(evaluation["seeds"])
            append_record(stream, {"global_step": model.num_timesteps, "checkpoint": checkpoint.name,
                                   "scenario": name, "deterministic": True, **report})
            for key, values in report["summary"].items():
                model.logger.record(f"evaluation/{name}/{key}", values["mean"])
    model.logger.dump(model.num_timesteps)


def train_with_checkpoints(model: PPO, config: dict[str, Any], session: dict[str, Any]) -> None:
    """학습 → 정책 갱신 완료 시점의 저장 → 평가를 목표 step까지 반복한다."""
    training = config["training"]
    target = model.num_timesteps + training["additional_steps"]
    with closing(TrainingLog(session["folder"], config["reward"], model.n_envs)) as callback:
        try:
            while model.num_timesteps < target:
                steps = min(training["checkpoint_interval_steps"], target - model.num_timesteps)
                model.learn(steps, callback=callback, tb_log_name="training", reset_num_timesteps=False)
                checkpoint = save_checkpoint(model, session)
                evaluate_checkpoint(model, config, session, checkpoint)
                model.logger.close()
                print(f"저장·평가 완료: {checkpoint}", flush=True)
        finally:
            if hasattr(model, "_logger"):
                model.logger.close()


def configuration_path() -> Path:
    """직접 작성한 설정 JSON 경로를 실행 인자로 받는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="모델 폴더 안의 학습 설정 JSON")
    return parser.parse_args().config.resolve()


def main() -> None:
    """설정 읽기 → 실행 등록 → 환경·모델 준비 → 설정 보존 → 기록하며 학습·저장·평가."""
    # 실험 실행 설정파일 읽기
    # --config path 만 넣으면 그 경로에 알아서 model.json 만들지고 path.stem/ 폴더안에 결과가 생성됨
    path = configuration_path()
    # 설정파일에서 config 덮어씌워서 가져오기
    config = read_configuration(path)
    # model.json 있으면 이전 센션 등 확인해서 진행
    session = prepare_session(path, config)
    # shortcut: 같은 모델 폴더의 동시 학습 실행은 지원하지 않는다. 필요하면 대표 JSON 잠금을 추가한다.
    with session_status(session), closing(make_training_environments(config)) as environment:
        model = create_model(config, environment, session)
        save_configuration(config, session)
        train_with_checkpoints(model, config, session)


if __name__ == "__main__":
    main()
