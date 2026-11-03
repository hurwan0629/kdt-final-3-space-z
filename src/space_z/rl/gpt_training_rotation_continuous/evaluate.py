"""지정 체크포인트를 같은 기본 게임 조건에서 비교하고 JSON으로 저장한다."""
import argparse
import hashlib
from pathlib import Path

import torch
from stable_baselines3 import PPO

from .train import ROOT, evaluate, write_json
from .policy import RotationContinuousPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--max-seconds', type=int, default=400)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stochastic', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists() or args.max_seconds <= 0 or min(args.seeds) < 0:
        parser.error('Use a new output inside this experiment folder, positive duration and nonnegative seeds')
    torch.set_num_threads(1)
    reports = []
    for checkpoint in args.checkpoints:
        model = PPO.load(checkpoint, device='cpu')
        sampled = args.stochastic if args.stochastic is not None else getattr(model, 'stochastic_inference', False)
        decoder_source = (ROOT.parent / 'actions/rotation_continuous_action_decoder.py'
                          if isinstance(model.policy, RotationContinuousPolicy)
                          else ROOT.parent / 'gpt_training/environment.py')
        report = dict(checkpoint=str(checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      decoder_source=str(decoder_source),
                      decoder_source_sha256=hashlib.sha256(decoder_source.read_bytes()).hexdigest(),
                      **evaluate(model, args.seeds, args.max_seconds, deterministic=not sampled))
        reports.append(report)
        write_json(output, reports)
        print(checkpoint.name, report['summary'], flush=True)


if __name__ == '__main__':
    main()
