"""PPO 학습 실행: python -m space_z.rl.train --steps 100000 [--pixels] [--async].

추가 의존성 설치: python -m pip install stable-baselines3
--async는 환경 수집을 별도 프로세스에서 병렬 실행한다. 정책 갱신은 수집 후 함께 수행한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gymnasium.spaces import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from space_z.config import GameConfig, RewardConfig
from space_z.rl.actions.continuous_action_decoder import ContinuousActionDecoder
from space_z.rl.environment import TankEnv
from space_z.rl.observations.pixel_observation import PixelObservation
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.rewards import RewardCalculator


class ModelTrainer:
    """전달받은 환경에서 PPO를 학습하고 모델을 저장한다. 환경 해제는 호출자가 맡는다."""

    def __init__(self, environment: TankEnv | VecEnv, seed: int = 42, post_train = None) -> None:
        """관찰 형식에 맞는 PPO 모델을 생성한다.

        인자:
            environment: 구성 요소를 연결한 단일 환경 또는 Monitor가 적용된 병렬 환경.
            seed: 모델 초기화·행동 추첨·첫 환경 초기화의 난수 시드.

        반환:
            None. 벡터는 MlpPolicy, 이미지·성장 딕셔너리는 MultiInputPolicy를 사용한다.
        """
        policy = "MultiInputPolicy" if isinstance(environment.observation_space, Dict) else "MlpPolicy"
        training_environment = environment if isinstance(environment, VecEnv) else Monitor(environment)
        
        self.is_post_train = True if post_train is not None else False

        if post_train is not None:
            self.model = PPO.load(post_train, env=training_environment, device="cpu")
        else:
            self.model = PPO(
                policy,
                training_environment,  # 각 환경의 Monitor가 에피소드 보상과 길이를 기록한다.
                n_steps=256,  # 환경마다 256 step을 모은 뒤 정책을 갱신한다.
                batch_size=64,  # 한 번의 경사 계산에 사용하는 경험 수
                seed=seed,
                device="cpu",
                verbose=1,
            )

    def train(self, total_timesteps: int, model_path: str | Path) -> PPO:
        """지정한 환경 step 수만큼 학습한 뒤 ZIP 모델을 저장하고 반환한다.

        인자:
            total_timesteps: 모든 환경의 step 합계 목표. 모델의 n_steps × 환경 수 단위로 올림된다.
            model_path: 모델 저장 경로. 부모 폴더를 만들고 .zip 확장자는 자동으로 붙인다.

        반환:
            학습한 PPO 모델. model.predict(observation, deterministic=True)로 행동을 고른다.
        """
        if type(total_timesteps) is not int or total_timesteps <= 0:
            raise ValueError("total_timesteps는 양의 정수여야 합니다.")
        # 1) 환경의 관찰·보상을 사용해 모델을 학습한다.
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=(not self.is_post_train))
        # 2) 학습한 모델과 정책 설정을 저장한다.
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))
        return self.model


def make_environment(pixels: bool = False) -> TankEnv:
    """기존 게임·보상 설정으로 독립된 환경을 생성한다. 각 프로세스 안에서 호출한다."""
    return TankEnv(
        GameConfig(),
        PixelObservation() if pixels else VectorObservation(),
        ContinuousActionDecoder(enable_stat_upgrades=True, enable_evolution=True),
        RewardCalculator(RewardConfig(
            shape_damage = 5.0,
            shape_destroy = 0.5,
            enemy_damage = 0.5,
            enemy_kill = 5.0,
            damage_taken = -0.02,
            death = -5.0,
            time_step = -0.0001,
            shot_cost = 0.0001,
        )),
        # render_mode="human"
    )


def main() -> None:
    """실행 옵션에 따라 단일·병렬 환경을 생성하고 학습·저장 후 환경을 해제한다."""
    parser = argparse.ArgumentParser(description="Space Z PPO 학습")
    parser.add_argument("--steps", type=int, default=100_000, help="학습할 전체 환경 step 수")
    parser.add_argument("--seed", type=int, default=42, help="학습 난수 시드")
    parser.add_argument("--pixels", action="store_true", help="이미지·성장 벡터 관찰 사용")
    parser.add_argument("--post-train", type=Path, default=None, help="학습되있는 모델 추가 학습 (zip 경로)")
    parser.add_argument("--async", dest="async_environments", action="store_true", help="별도 프로세스에서 환경을 병렬 실행")
    parser.add_argument("--num-envs", type=int, default=4, help="--async 사용 시 환경 수(기본 4)")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps는 양의 정수여야 합니다.")
    if args.seed < 0:
        parser.error("--seed는 0 이상의 정수여야 합니다.")
    if args.num_envs <= 0:
        parser.error("--num-envs는 양의 정수여야 합니다.")

    # 1) 각 병렬 프로세스가 자체 환경을 만들며 make_vec_env가 개별 Monitor를 적용한다.
    environment: TankEnv | VecEnv
    if args.async_environments:
        environment = make_vec_env(
            make_environment, n_envs=args.num_envs, seed=args.seed,
            env_kwargs={"pixels": args.pixels}, vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"},
        )
    else:
        environment = make_environment(args.pixels)
    # 2) 학습·저장을 수행하고, 중간에 오류가 나도 환경 자원을 정리한다.
    try:
        trainer = ModelTrainer(environment, seed=args.seed, post_train=args.post_train)
        mode = "pixel" if args.pixels else "vector"
        if args.post_train is None:
            from datetime import datetime
            path = Path("trained_models") / f"{datetime.now().strftime("%Y%m%d%H%M%S")}_entropy_ppo_{mode}_steps_{args.steps}"
        else:
            path = Path("trained_models") / f"{args.post_train.stem}_plus_{args.steps}"
        trainer.train(args.steps, path)
        print(f"모델 저장 완료: {path.with_suffix('.zip').resolve()}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
