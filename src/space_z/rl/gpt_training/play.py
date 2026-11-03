"""새 학습기의 체크포인트를 같은 관찰·성장 규칙으로 화면에서 재생한다."""
import argparse
import math
from contextlib import closing
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces
import torch

from .environment import game_configuration, make_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path(__file__).resolve().parent / "pure_ppo_model.zip")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--stage", choices=["harvest", "combat", "full", "aim", "duel", "arena", "range", "moving_range"], default="full")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--max-steps', type=int)
    duration.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--frame-skip', type=int)
    parser.add_argument("--zoom", type=float, default=0.35)
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if (args.max_steps is not None and args.max_steps <= 0) or args.seed < 0 or args.zoom <= 0:
        parser.error("Invalid seed, episode length or zoom")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or (args.frame_skip is not None and args.frame_skip <= 0):
        parser.error('Game duration and frame skip must be positive')
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device="cpu")
    sampled = args.stochastic if args.stochastic is not None else getattr(model, 'stochastic_inference', False)
    if sampled:
        torch.manual_seed(1000003 + args.seed)
    pure_rl = getattr(model, "training_method", None) == "pure_ppo"
    decision_frame_skip = args.frame_skip or getattr(model, 'decision_frame_skip', 4)
    max_steps = args.max_steps or math.ceil(args.max_seconds * game_configuration(args.stage).fps / decision_frame_skip)
    with closing(make_environment(args.stage, max_steps, pure_rl=pure_rl, frame_skip=decision_frame_skip)) as environment:
        check_for_correct_spaces(environment, model.observation_space, model.action_space)
        environment.unwrapped.game_config.camera_zoom = args.zoom
        environment.unwrapped.render_mode = "human"
        info = {}
        try:
            observation, info = environment.reset(seed=args.seed)
            while not environment.unwrapped._closed:
                action, _ = model.predict(observation, deterministic=not sampled)
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    break
        except KeyboardInterrupt:
            pass
        print(f"score={info.get('score', 0)} kills={info.get('kills', 0)} seconds={info.get('frames', 0)/60:.1f}")


if __name__ == "__main__":
    main()
