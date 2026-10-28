"""PPO 학습 실행: python -m space_z.rl.train --steps 100000 [--pixels].

추가 의존성 설치: python -m pip install stable-baselines3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gymnasium.spaces import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from space_z.config import GameConfig, RewardConfig
from space_z.rl.actions.continuous_action_decoder import ContinuousActionDecoder
from space_z.rl.environment import TankEnv
from space_z.rl.observations.pixel_observation import PixelObservation
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.rewards import RewardCalculator


class ModelTrainer:
    """전달받은 환경에서 PPO를 학습하고 모델을 저장한다. 환경 해제는 호출자가 맡는다."""

    def __init__(self, environment: TankEnv, seed: int = 42) -> None:
        """관찰 형식에 맞는 PPO 모델을 생성한다.

        인자:
            environment: 관찰·행동·보상 구성 요소를 연결한 학습 환경.
            seed: 모델 초기화·행동 추첨·첫 환경 초기화의 난수 시드.

        반환:
            None. 벡터는 MlpPolicy, 이미지·성장 딕셔너리는 MultiInputPolicy를 사용한다.
        """
        policy = "MultiInputPolicy" if isinstance(environment.observation_space, Dict) else "MlpPolicy"
        self.model = PPO(
            policy,
            Monitor(environment),  # 에피소드 보상과 길이를 학습 로그에 기록한다.
            n_steps=256,  # 256 환경 step을 모은 뒤 정책을 갱신한다.
            batch_size=64,  # 한 번의 경사 계산에 사용하는 경험 수
            seed=seed,
            device="cpu",
            verbose=1,
        )

    def train(self, total_timesteps: int, model_path: str | Path) -> PPO:
        """지정한 환경 step 수만큼 학습한 뒤 ZIP 모델을 저장하고 반환한다.

        인자:
            total_timesteps: 양의 정수인 학습 step 목표. 256 step 묶음 단위로 올림된다.
            model_path: 모델 저장 경로. 부모 폴더를 만들고 .zip 확장자는 자동으로 붙인다.

        반환:
            학습한 PPO 모델. model.predict(observation, deterministic=True)로 행동을 고른다.
        """
        if type(total_timesteps) is not int or total_timesteps <= 0:
            raise ValueError("total_timesteps는 양의 정수여야 합니다.")
        # 1) 환경의 관찰·보상을 사용해 모델을 학습한다.
        self.model.learn(total_timesteps=total_timesteps)
        # 2) 학습한 모델과 정책 설정을 저장한다.
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))
        return self.model


def main() -> None:
    """실행 옵션을 읽어 환경을 생성하고 학습·저장 후 환경을 해제한다."""
    parser = argparse.ArgumentParser(description="Space Z PPO 학습")
    parser.add_argument("--steps", type=int, default=100_000, help="학습할 환경 step 수")
    parser.add_argument("--seed", type=int, default=42, help="학습 난수 시드")
    parser.add_argument("--pixels", action="store_true", help="이미지·성장 벡터 관찰 사용")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps는 양의 정수여야 합니다.")
    if args.seed < 0:
        parser.error("--seed는 0 이상의 정수여야 합니다.")

    # 1) 관찰·행동·보상 구성 요소로 화면 없는 학습 환경을 만든다.
    environment = TankEnv(
        GameConfig(),
        PixelObservation() if args.pixels else VectorObservation(),
        ContinuousActionDecoder(enable_stat_upgrades=True, enable_evolution=True),
        RewardCalculator(RewardConfig(
            shape_damage = 0.02,
            shape_destroy = 0.04,
            enemy_damage = 0.04,
            enemy_kill = 2.0,
            damage_taken = -0.02,
            death = -20.0,
            time_step = -0.0001,
            shot_cost = -0.0001,
        )),
        # render_mode="human"
    )
    # 2) 학습·저장을 수행하고, 중간에 오류가 나도 환경 자원을 정리한다.
    try:
        trainer = ModelTrainer(environment, seed=args.seed)
        mode = "pixel" if args.pixels else "vector"
        from datetime import datetime
        path = Path("trained_models") / f"{datetime.now().strftime("%Y%m%d%H%M%S")}ppo_{mode}"
        trainer.train(args.steps, path)
        print(f"모델 저장 완료: {path.with_suffix('.zip').resolve()}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
