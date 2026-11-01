from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Literal, TypedDict
from zipfile import ZipFile, ZIP_DEFLATED

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
import torch

from .configuration import ExperimentConfig, configuration_to_dict, evaluation_to_dict


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
    user_configuration_path = run_directory / "user_config.json"
    user_configuration_path.write_bytes(configuration_file_path.read_bytes())

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


def save_configuration(experiment_config: ExperimentConfig, run_session: RunSession) -> None:
    """기본값·부모 설정까지 반영한 설정, 라이브러리 버전, 실행 소스 사본을 보존한다."""
    # 소스 사본 저장
    project_root = Path(__file__).resolve().parents[4]
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
