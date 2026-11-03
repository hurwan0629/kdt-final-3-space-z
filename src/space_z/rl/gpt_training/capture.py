"""실제 게임 화면과 그때의 점수·행동을 이미지 및 JSON으로 남긴다."""
import argparse
import math
from contextlib import closing
from pathlib import Path

import pygame
from stable_baselines3 import PPO
import torch

from .environment import game_configuration, make_environment
from .train import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=101)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--max-steps', type=int)
    duration.add_argument('--max-seconds', type=float, default=200)
    parser.add_argument('--frame-skip', type=int)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--stage', choices=['aim', 'duel', 'arena', 'full', 'range', 'moving_range'], default='full')
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    directory = args.output.resolve()
    if not directory.is_relative_to(ROOT) or directory.exists() or args.interval <= 0 or args.seed < 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error('Use a new output directory inside gpt_training and positive lengths')
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or (args.frame_skip is not None and args.frame_skip <= 0):
        parser.error('Game duration and frame skip must be positive')
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device='cpu')
    sampled = args.stochastic if args.stochastic is not None else getattr(model, 'stochastic_inference', False)
    if sampled:
        torch.manual_seed(1000003 + args.seed)
    records = []
    decision_frame_skip = args.frame_skip or getattr(model, 'decision_frame_skip', 4)
    max_steps = args.max_steps or math.ceil(args.max_seconds * game_configuration(args.stage).fps / decision_frame_skip)
    with closing(make_environment(args.stage, max_steps, pure_rl=True, frame_skip=decision_frame_skip)) as environment:
        environment.unwrapped.game_config.camera_zoom = 0.35
        observation, info = environment.reset(seed=args.seed)
        def screenshot(step, action):
            base = environment.unwrapped
            base.render_mode = 'human'
            base.render()
            base.render_mode = None
            path = directory / f'step_{step:06d}.png'
            pygame.image.save(pygame.display.get_surface(), str(path))
            player = base.game.player
            records.append({'step': step, 'image': path.name, 'score': info['score'], 'kills': info['kills'],
                            'seconds': info['frames']/60, 'hp': player.hp, 'position': [player.x, player.y],
                            'angle': player.angle, 'action': action, 'ship': player.job,
                            'stat_levels': dict(base.game.progression.stat_levels)})
            write_json(directory / 'frames.json', records)

        screenshot(0, None)
        for step in range(1, max_steps+1):
            action = model.predict(observation, deterministic=not sampled)[0]
            observation, _, terminated, truncated, info = environment.step(action)
            if step % args.interval == 0 or terminated or truncated:
                screenshot(step, action.tolist())
            if terminated or truncated:
                break
    print(f'Captured {len(records)} frames: score={info["score"]}, seconds={info["frames"]/60:.2f}', flush=True)


if __name__ == '__main__':
    main()
