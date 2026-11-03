"""배포 폴더의 두 PPO 모델을 기존 게임에서 재생한다."""
import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
# 직접 실행하면 space_z.rl.__init__의 다른 RL 폴더 선행 import를 거치지 않는다.
sys.path.insert(0, str(PACKAGE.parent))
sys.path.insert(0, str(PACKAGE.parents[2]))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces

from space_z.config import EnemyAIConfig, EnvConfig, GameConfig, ObsConfig, RewardConfig, TerrainConfig
from best_model_execute.decoders import FactorizedActionDecoder, RotationContinuousActionDecoder
from best_model_execute.environment import TankEnv
from best_model_execute.rewards import RewardCalculator
from best_model_execute.vector_observation import VectorObservation


def read_manifest():
    """동봉한 모델 출처와 학습 당시의 게임·관찰 설정을 읽는다."""
    return json.loads((PACKAGE / 'models.json').read_text(encoding='utf-8'))


def load_model(model_name, manifest):
    """선정 파일의 무결성을 확인하고 CPU에서 정책을 복원한다."""
    entry = manifest['models'][model_name]
    checkpoint = PACKAGE / entry['file']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Model checksum mismatch: {checkpoint.name}')
    model = PPO.load(checkpoint, device='cpu')
    if getattr(model, 'training_method', None) != 'pure_ppo':
        raise ValueError('Only the packaged pure PPO models are supported')
    return model


def make_environment(model_name, manifest, max_seconds=400, zoom=0.35, headless=False):
    """동봉된 설정으로 기본 게임과 모델별 행동 해석기를 연결한다."""
    options = dict(manifest['game_config'])
    enemy_options = EnemyAIConfig(**options.pop('enemy_ai'))
    terrain_options = TerrainConfig(**options.pop('terrain'))
    game_config = GameConfig(**options, enemy_ai=enemy_options, terrain=terrain_options)
    game_config.camera_zoom = zoom
    observation_config = ObsConfig(**manifest['observation_config'])
    observation_builder = VectorObservation(observation_config)
    action_decoder = FactorizedActionDecoder() if model_name == 'discrete' else RotationContinuousActionDecoder()
    reward_calculator = RewardCalculator(RewardConfig())
    frame_skip = manifest['models'][model_name]['frame_skip']
    maximum_steps = math.ceil(max_seconds * game_config.fps / frame_skip)
    environment_config = EnvConfig(frame_skip=frame_skip, max_steps=maximum_steps)
    return TankEnv(
        game_config=game_config, observation_builder=observation_builder, action_decoder=action_decoder,
        reward_calculator=reward_calculator, env_config=environment_config,
        render_mode=None if headless else 'human',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['discrete', 'rotation'], default='rotation')
    parser.add_argument('--seed', type=int, default=223)#61001)
    parser.add_argument('--max-seconds', type=float, default=400)
    parser.add_argument('--zoom', type=float, default=0.35)
    parser.add_argument('--headless', action='store_true', help='화면 없이 게임을 진행한다')
    args = parser.parse_args()
    if args.seed < 0 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Use a nonnegative seed and positive finite duration')
    if not math.isfinite(args.zoom) or args.zoom <= 0:
        parser.error('Zoom must be finite and positive')
    torch.set_num_threads(1)
    manifest = read_manifest()
    model = load_model(args.model, manifest)
    print(f'Model: {manifest["models"][args.model]["name"]}', flush=True)
    with closing(make_environment(args.model, manifest, args.max_seconds, args.zoom, args.headless)) as environment:
        check_for_correct_spaces(environment, model.observation_space, model.action_space)
        info = {}
        interrupted = False
        try:
            observation, info = environment.reset(seed=args.seed)
            while not environment._closed:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    break
        except KeyboardInterrupt:
            interrupted = True
        result = dict(model=args.model, seed=args.seed, score=info.get('score', 0),
                      kills=info.get('kills', 0), seconds=info.get('frames', 0) / 60,
                      ship=environment.game.player.job, interrupted=interrupted)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
