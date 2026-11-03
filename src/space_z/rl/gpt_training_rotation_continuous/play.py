"""연속 회전 모델을 게임 화면에서 재생하거나 실제 화면을 캡처한다."""
import argparse
from contextlib import closing
import hashlib
import math
import os
from pathlib import Path

from .train import ROOT, write_json
from .environment import RotationTrainingEnvironment
from .policy import RotationContinuousPolicy

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='?', default=ROOT / 'selected_model.zip')
    parser.add_argument('--seed', type=int, default=505)
    parser.add_argument('--max-seconds', type=int, default=400)
    parser.add_argument('--zoom', type=float, default=0.35)
    parser.add_argument('--capture-directory', type=Path)
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.seed < 0 or args.max_seconds <= 0 or not math.isfinite(args.zoom) or args.zoom <= 0:
        parser.error('Use nonnegative seed and positive duration/zoom')
    capture = args.capture_directory.resolve() if args.capture_directory else None
    if capture is not None:
        if not capture.is_relative_to(ROOT) or capture.exists():
            parser.error('Capture directory must be new and inside this experiment folder')
        capture.mkdir(parents=True)
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    import pygame
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device='cpu')
    sampled = args.stochastic if args.stochastic is not None else getattr(model, 'stochastic_inference', False)
    if sampled:
        torch.manual_seed(1000003 + args.seed)
    if not isinstance(model.policy, RotationContinuousPolicy):
        parser.error('Use a rotation-continuous checkpoint')
    with closing(RotationTrainingEnvironment(max_steps=args.max_seconds * 30, reward_profile='legacy')) as environment:
        check_for_correct_spaces(environment, model.observation_space, model.action_space)
        environment.unwrapped.game_config.camera_zoom = args.zoom
        environment.unwrapped.render_mode = None if capture else 'human'
        if capture:
            write_json(capture / 'metadata.json', dict(
                checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                decoder_source_sha256=hashlib.sha256((ROOT.parent / 'actions/rotation_continuous_action_decoder.py').read_bytes()).hexdigest(),
                seed=args.seed, replay_mode='sampled' if sampled else 'greedy',
                policy_seed=1000003 + args.seed if sampled else None,
                max_seconds=args.max_seconds, frame_skip=2, camera_zoom=args.zoom,
            ))
        observation, info = environment.reset(seed=args.seed)
        records = []
        try:
            for step in range(1, args.max_seconds * 30 + 1):
                action = model.predict(observation, deterministic=not sampled)[0]
                observation, _, terminated, truncated, info = environment.step(action)
                if capture and (step % 3000 == 0 or terminated or truncated):
                    base = environment.unwrapped
                    base.render_mode = 'human'
                    base.render()
                    base.render_mode = None
                    filename = f'step_{step:06d}.png'
                    pygame.image.save(pygame.display.get_surface(), str(capture / filename))
                    records.append(dict(step=step, image=filename, score=info['score'], kills=info['kills'],
                                        seconds=info['frames'] / 60, hp=base.game.player.hp,
                                        action=action.tolist(), ship=base.game.player.job))
                    write_json(capture / 'frames.json', records)
                if terminated or truncated or environment.unwrapped._closed:
                    break
        except KeyboardInterrupt:
            pass
        print(dict(score=info['score'], kills=info['kills'], seconds=info['frames'] / 60))


if __name__ == '__main__':
    main()
