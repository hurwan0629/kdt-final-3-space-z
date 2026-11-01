"""학습에 사용하지 않은 시드에서 체크포인트와 교사를 순차 비교한다."""
import argparse
import hashlib
import math
from pathlib import Path

from stable_baselines3 import PPO
import torch

from .environment import game_configuration, teacher_action
from .train import ROOT, evaluate, policy_action, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--stage', choices=['full', 'aim', 'duel', 'arena', 'range', 'moving_range'], default='full')
    parser.add_argument("--seeds", type=int, nargs="+", default=[404, 505, 606])
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--max-steps', type=int)
    duration.add_argument('--max-seconds', type=float, default=400)
    parser.add_argument('--frame-skip', type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("Output must be a new file inside gpt_training")
    if (args.max_steps is not None and args.max_steps <= 0) or any(seed < 0 for seed in args.seeds) or not (args.checkpoints or args.teacher):
        parser.error("Invalid evaluation conditions")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Game duration must be finite and positive')
    if args.frame_skip is not None and args.frame_skip <= 0:
        parser.error('Frame skip must be positive')
    torch.set_num_threads(1)
    fps = game_configuration(args.stage).fps
    reports = []
    if args.teacher:
        teacher_frame_skip = args.frame_skip or 4
        max_steps = args.max_steps or math.ceil(args.max_seconds * fps / teacher_frame_skip)
        report = {"policy": "teacher", **evaluate(teacher_action, args.seeds, max_steps, frame_skip=teacher_frame_skip)}
        reports.append(report)
        write_json(output, reports)
        print(report, flush=True)
    for path in args.checkpoints:
        checkpoint = path.resolve()
        model = PPO.load(checkpoint, device="cpu")
        sampled = args.stochastic if args.stochastic is not None else getattr(model, 'stochastic_inference', False)
        decision_frame_skip = args.frame_skip or getattr(model, 'decision_frame_skip', 4)
        max_steps = args.max_steps or math.ceil(args.max_seconds * fps / decision_frame_skip)
        report = {
            "policy": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'execution_mode': 'sampled' if sampled else 'greedy',
            **evaluate(lambda observation: policy_action(model, observation, deterministic=not sampled), args.seeds, max_steps,
                       stage=args.stage, pure_rl=getattr(model, "training_method", None) == "pure_ppo",
                       policy_seed=1000003 if sampled else None, frame_skip=decision_frame_skip),
        }
        reports.append(report)
        write_json(output, reports)
        print(report, flush=True)


if __name__ == "__main__":
    main()
