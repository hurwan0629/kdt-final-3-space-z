"""새 분기의 체크포인트를 실제 게임 창에서 실행한다."""
import argparse
import torch
from stable_baselines3 import PPO

from .environment import make_environment
from .policy import VectorOnlyPolicy
from .storage import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', type=str)
    parser.add_argument('--seed', type=int, default=710101)
    parser.add_argument('--max-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error('max-seconds must be positive')
    checkpoint = args.checkpoint or str(ROOT / 'experiments/002_full_game_pilot/checkpoints/best.zip')
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    if not isinstance(model.policy, VectorOnlyPolicy):
        raise ValueError('This player requires a vector-only checkpoint')
    environment = make_environment(args.max_seconds, render_mode='human')
    try:
        observation, info = environment.reset(seed=args.seed)
        while not environment.unwrapped._closed:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                print(f'seed={args.seed}, score={info["score"]}, survival={info["frames"]/60:.2f}s')
                break
    finally:
        environment.close()


if __name__ == '__main__':
    main()
