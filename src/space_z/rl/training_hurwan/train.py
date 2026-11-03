"""설정 JSON으로 실행하는 PPO 기록 예제. 기존 환경·train.py는 수정하지 않는다.

실행: python -m space_z.rl.training_hurwan.train --config src/space_z/rl/runs/model_001/fighting.json
설정 최소 예: {"game": {"num_enemies": 0, "terrain": {"enabled": false}},
              "training": {"additional_steps": 10000}}
필요 패키지: python -m pip install stable-baselines3 tensorboard

JSON 옆 model.json에서 자동 재개하고 001_fighting/ 등에 실제 설정·소스 사본,
TensorBoard·step/에피소드 JSONL·체크포인트·고정 시나리오 평가를 저장한다.
추가 학습은 최근 저장 완료 지점부터 시작한다. 게임 중간 상태·rollout 버퍼·영상은 저장하지 않는다."""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from .configuration import read_configuration
from .environments import make_training_environments
from .run_session import prepare_session, save_configuration, session_status
from .trainer import create_model, train_with_checkpoints


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
