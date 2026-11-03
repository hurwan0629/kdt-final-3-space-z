"""고정한 체크포인트들을 같은 시드에서 평가하고 TensorBoard에 기록한다."""
import argparse
import time

from .experiment import PPO, torch, evaluate, parent_metadata
from .metrics import episode_metrics
from .storage import ROOT, append_json, now, write_json
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='004_pretraining_comparison')
    parser.add_argument('--checkpoints', nargs='+', default=[
        'experiments/002_full_game_pilot/checkpoints/final.zip',
        'experiments/003_training_duration/checkpoints/step_000196608.zip',
        'experiments/003_training_duration/checkpoints/final.zip'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[730001 + 101 * i for i in range(20)])
    args = parser.parse_args()
    if not args.name or '/' in args.name or '\\' in args.name or args.name in ('.', '..'):
        parser.error('Name must be one directory name')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    parents = [parent_metadata(ROOT / name) for name in args.checkpoints]
    directory = ROOT / 'evaluations' / args.name
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', dict(checkpoints=parents, seeds=args.seeds,
               max_seconds=120, deterministic=True, created_at=now(), role='diagnostic_comparison_no_training'))
    state = dict(status='evaluating', completed_episodes=0, total_episodes=len(parents)*len(args.seeds))
    write_json(directory / 'state.json', state)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = []
    started = time.monotonic()
    try:
        for parent in parents:
            model = PPO.load(ROOT / parent['checkpoint'], device='cpu')
            label = f'step_{model.num_timesteps:09d}'
            episodes = []
            with SummaryWriter(str(directory / 'tensorboard' / label)) as writer:
                for index, seed in enumerate(args.seeds):
                    episode = evaluate(model, directory, model.num_timesteps, seeds=[seed])['episodes'][0]
                    episodes.append(episode)
                    append_json(directory / 'episodes.jsonl', dict(checkpoint=parent, **episode))
                    for name, value in episode_metrics(episode).items():
                        writer.add_scalar('episodes/' + name, value, index + 1)
                    state.update(completed_episodes=state['completed_episodes']+1,
                                 checkpoint=parent['checkpoint'], seed=seed, updated_at=now())
                    write_json(directory / 'state.json', state)
                metrics = [episode_metrics(episode) for episode in episodes]
                summary = {name:dict(mean=float(np.mean([row[name] for row in metrics])),
                                    median=float(np.median([row[name] for row in metrics])),
                                    p25=float(np.percentile([row[name] for row in metrics], 25)))
                           for name in metrics[0]}
                for name, values in summary.items():
                    writer.add_scalar('mean/' + name, values['mean'], model.num_timesteps)
                result = dict(checkpoint=parent, summary=summary, episodes=episodes)
                results.append(result)
                write_json(directory / (label + '.json'), result)
                print(label, 'score=', summary['outcomes/score']['mean'],
                      'survival=', summary['outcomes/survival_seconds']['mean'], flush=True)
            del model
        write_json(directory / 'comparison.json', dict(seeds=args.seeds, results=results))
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic()-started, updated_at=now())
        write_json(directory / 'state.json', state)


if __name__ == '__main__':
    main()
