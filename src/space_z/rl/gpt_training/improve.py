"""실험 가설을 기록하고 실제 기본 게임 점수로 순수 PPO 후보를 고른다."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from .train import ROOT, ExperimentLog, append_json, evaluate, policy_action, save_model, write_json
from .environment import EnvironmentSettings, game_configuration, make_environment
from .networks import GrowthMaskedPolicy, SpatialFeaturesExtractor
from .score_reward import reward_settings

import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter


def training_environment(stage, max_steps, reward_profile, frame_skip=4):
    return Monitor(make_environment(stage, max_steps, pure_rl=True, reward_profile=reward_profile, frame_skip=frame_skip))


def checkpoint_ranking(report, objective):
    """초반 점수의 평균과 하위 성적을 함께 보거나 기존 최종 점수로 비교한다."""
    summary = report['summary']
    survival = summary['survival_seconds']['mean']
    if objective == 'early_stability':
        early = summary['score_at_120_seconds']
        return [early['mean'] + early['p25'], early['p25'], summary['score_at_200_seconds']['mean'], survival]
    completed = sum(episode['truncated'] for episode in report['episodes']) / len(report['episodes'])
    return [summary['score']['mean'], survival, completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--stage', choices=['aim', 'duel', 'arena', 'full', 'range', 'moving_range'], required=True)
    parser.add_argument('--steps', type=int, default=262144)
    parser.add_argument('--eval-interval', type=int, default=65536)
    parser.add_argument('--num-envs', type=int, default=6)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--episode-steps', type=int, default=1500)
    parser.add_argument('--eval-steps', type=int, default=3000)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[101, 202, 303, 404, 505, 606])
    parser.add_argument('--selection-objective', choices=['score', 'early_stability'], default='score')
    parser.add_argument('--seed', type=int, default=2042)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--entropy', type=float, default=0.01)
    parser.add_argument('--reward-profile', choices=['score_v2', 'score_v3', 'score_v4', 'score_v5', 'score_v6', 'score_v7'], default='score_v2')
    parser.add_argument('--aim-residual', action='store_true')
    parser.add_argument('--freeze-base-actor', action='store_true')
    parser.add_argument('--unified-aim-input', action='store_true')
    parser.add_argument('--independent-aim', action='store_true')
    parser.add_argument('--freeze-aim', action='store_true')
    parser.add_argument('--ignore-aim-health', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.freeze_base_actor and not args.aim_residual:
        parser.error('Freezing the base actor requires --aim-residual')
    if (args.independent_aim or args.freeze_aim or args.ignore_aim_health) and not args.aim_residual:
        parser.error('Aim configuration requires --aim-residual')
    if args.freeze_base_actor and args.freeze_aim:
        parser.error('At least one actor component must remain trainable')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('Name must be a single experiment directory')
    if not 1 <= args.num_envs <= 6:
        parser.error('Use at most 6 workers; Windows launcher + coordinator + workers <= 8')
    if min(args.steps, args.eval_interval, args.episode_steps, args.eval_steps, args.frame_skip) <= 0 or args.seed < 0:
        parser.error('Lengths must be positive and seed nonnegative')
    if not 0 < args.learning_rate < 1 or not 0 <= args.entropy < 1 or min(args.eval_seeds) < 0:
        parser.error('Invalid optimizer or evaluation settings')
    if args.selection_objective == 'early_stability' and (args.eval_steps * args.frame_skip < 200 * game_configuration('full').fps or len(args.eval_seeds) < 4):
        parser.error('Early-stability selection requires at least 200 seconds and four validation seeds')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    directory = ROOT / 'experiments' / args.name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'checkpoints').mkdir()
    with ZipFile(directory / 'source_code.zip', 'w', ZIP_DEFLATED) as archive:
        for source in sorted(ROOT.glob('*.py')):
            archive.write(source, source.name)
    manifest = {
        'name': args.name, 'goal': 'Increase early scoring speed and lower-quartile consistency in unchanged default GameConfig' if args.selection_objective == 'early_stability' else 'Increase actual score in unchanged default GameConfig',
        'hypothesis': args.hypothesis, 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'training_method': 'pure_ppo', 'demonstration_steps': 0,
        'versions': {'python': sys.version, 'torch': torch.__version__,
                     'stable_baselines3': stable_baselines3.__version__, 'numpy': np.__version__},
        'training_game': asdict(game_configuration(args.stage)),
        'evaluation_game': asdict(game_configuration('full')),
        'training_environment': asdict(EnvironmentSettings(args.stage, args.episode_steps, pure_rl=True, reward_profile=args.reward_profile, frame_skip=args.frame_skip)),
        'range_rules': {'movement_locked': True, 'shape_distance': [220, 480], 'shape_max_hp': 7,
                        'velocity_components': [-3, 3] if args.stage == 'moving_range' else [0, 0],
                        'relocate_distance': [180, 600] if args.stage == 'moving_range' else None} if args.stage in ('range', 'moving_range') else None,
        'reward': asdict(reward_settings(args.reward_profile)),
        'policy': 'GrowthMaskedPolicy + SpatialFeaturesExtractor; original 5139 inputs; all weights PPO-trained',
        'selection_rule': '120-second mean score + p25, then p25, 200-second mean score, survival' if args.selection_objective == 'early_stability' else 'mean game score, then mean survival, then time-limit completion fraction',
        'source_sha256': hashlib.sha256((directory / 'source_code.zip').read_bytes()).hexdigest(),
        'evaluation_note': 'Evaluation uses unchanged full game and original legacy reward wrapper; select actual game score, not evaluation reward',
    }
    write_json(directory / 'experiment.json', manifest)
    environments = None
    model = None
    started = time.monotonic()
    try:
        factories = [partial(training_environment, args.stage, args.episode_steps, args.reward_profile, args.frame_skip) for _ in range(args.num_envs)]
        environments = SubprocVecEnv(factories, start_method='spawn') if args.num_envs > 1 else DummyVecEnv(factories)
        environments.seed(args.seed)
        if args.resume:
            model = PPO.load(args.resume.resolve(), env=environments, device='cpu',
                             learning_rate=args.learning_rate, gamma=manifest['reward']['gamma'],
                             tensorboard_log=str(directory / 'tensorboard'))
            if getattr(model, 'training_method', None) != 'pure_ppo' or not isinstance(model.policy, GrowthMaskedPolicy):
                raise ValueError('Only resume pure-PPO structured-policy checkpoints')
            model.ent_coef = args.entropy
            manifest['parent_sha256'] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
        else:
            model = PPO(
                GrowthMaskedPolicy, environments, device='cpu', seed=args.seed, verbose=1,
                n_steps=512, batch_size=512, n_epochs=4, learning_rate=args.learning_rate,
                gamma=manifest['reward']['gamma'], gae_lambda=0.95, ent_coef=args.entropy, clip_range=0.2,
                target_kl=0.025, tensorboard_log=str(directory / 'tensorboard'),
                policy_kwargs={'features_extractor_class': SpatialFeaturesExtractor,
                               'net_arch': {'pi': [128, 128], 'vf': [128, 128]}},
            )
        # load()가 부모의 seed를 복원하므로 이번 실험의 환경·탐색 seed를 다시 적용한다.
        model.seed = args.seed
        model.set_random_seed(args.seed)
        model.decision_frame_skip = args.frame_skip
        model.training_method = 'pure_ppo'
        if args.aim_residual:
            model.policy.enable_aim_residual()
            model.policy_kwargs['aim_residual'] = True
        if args.independent_aim:
            manifest['aim_output_reinitialized'] = not model.policy.independent_aim
            model.policy.enable_independent_aim()
            model.policy_kwargs['independent_aim'] = True
            model.policy_kwargs['unified_aim_input'] = True
        model.policy.set_aim_frozen(args.freeze_aim)
        model.policy_kwargs['freeze_aim'] = args.freeze_aim
        manifest['independent_aim'] = model.policy.independent_aim
        manifest['aim_frozen'] = args.freeze_aim
        manifest['aim_residual'] = model.policy.use_aim_residual
        model.policy.set_actor_frozen(args.freeze_base_actor)
        model.policy_kwargs['freeze_base_actor'] = args.freeze_base_actor
        manifest['base_actor_frozen'] = args.freeze_base_actor
        if args.unified_aim_input:
            model.policy.unified_aim_input = True
            model.policy_kwargs['unified_aim_input'] = True
        manifest['unified_aim_input'] = model.policy.unified_aim_input
        if args.ignore_aim_health:
            model.policy.ignore_aim_health = True
            model.policy_kwargs['ignore_aim_health'] = True
        manifest['ignore_aim_health'] = model.policy.ignore_aim_health
        model.research_version = 'spatial_' + args.reward_profile
        manifest['initial_timesteps'] = model.num_timesteps
        manifest['ppo'] = {key: getattr(model, key) for key in ['n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'ent_coef', 'target_kl']}
        assert model.gamma == manifest['reward']['gamma']
        assert model.rollout_buffer.gamma == model.gamma
        callback = ExperimentLog(directory, args.eval_interval)
        callback.next_checkpoint = model.num_timesteps + args.eval_interval
        with SummaryWriter(str(directory / 'tensorboard/evaluation')) as writer:
            writer.add_text('experiment/goal', manifest['goal'], model.num_timesteps)
            writer.add_text('experiment/hypothesis', args.hypothesis, model.num_timesteps)
            def assess(label):
                evaluated_checkpoint = directory / 'checkpoints' / f'evaluated_{model.num_timesteps:09d}.zip'
                save_model(model, evaluated_checkpoint)
                checkpoint_hash = hashlib.sha256(evaluated_checkpoint.read_bytes()).hexdigest()
                report = evaluate(partial(policy_action, model), args.eval_seeds, args.eval_steps, pure_rl=True, frame_skip=args.frame_skip)
                record = {'label': label, 'timesteps': model.num_timesteps,
                          'checkpoint': str(evaluated_checkpoint.relative_to(ROOT)),
                          'checkpoint_sha256': checkpoint_hash, **report}
                append_json(directory / 'evaluations.jsonl', record)
                for metric, statistics in report['summary'].items():
                    writer.add_scalar(f'evaluation/{metric}', statistics['mean'], model.num_timesteps)
                    writer.add_scalar(f'evaluation_p25/{metric}', statistics['p25'], model.num_timesteps)
                ranking = checkpoint_ranking(report, args.selection_objective)
                if ranking > manifest.get('best_ranking', [-1, -1, -1]):
                    temporary_best = directory / 'checkpoints/best.tmp.zip'
                    shutil.copyfile(evaluated_checkpoint, temporary_best)
                    temporary_best.replace(directory / 'checkpoints/best.zip')
                    manifest.update(best_ranking=ranking, best_timesteps=model.num_timesteps, best_label=label,
                                    best_checkpoint=str(evaluated_checkpoint.relative_to(ROOT)), best_sha256=checkpoint_hash)
                write_json(directory / 'experiment.json', manifest)
                print(f'EVALUATION {label} steps={model.num_timesteps}: {json.dumps(report["summary"])}', flush=True)

            assess('before')
            final_step = model.num_timesteps + args.steps
            while model.num_timesteps < final_step:
                model.learn(min(args.eval_interval, final_step-model.num_timesteps), callback=callback,
                            reset_num_timesteps=False, tb_log_name='ppo')
                save_model(model, directory / 'checkpoints/latest.zip')
                assess('progress')
                if (directory / 'stop.request').exists():
                    break
            save_model(model, directory / 'checkpoints/final.zip')
            manifest['status'] = 'stopped' if (directory / 'stop.request').exists() else 'completed'
            if manifest['status'] == 'stopped':
                manifest['stop_reason'] = (directory / 'stop.request').read_text(encoding='utf-8-sig').strip()
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        if model is not None:
            save_model(model, directory / 'checkpoints/interrupted.zip')
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        if environments is not None:
            environments.close()
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        manifest.update(timesteps=model.num_timesteps if model else 0,
                        elapsed_seconds=time.monotonic()-started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'experiment.json', manifest)
        append_json(ROOT / 'experiments.jsonl', {'name': args.name, 'status': manifest['status'], 'timesteps': manifest['timesteps']})


if __name__ == '__main__':
    main()
