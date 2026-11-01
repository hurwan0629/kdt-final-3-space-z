"""회전 연속 PPO를 구간별 학습하고 기본 게임의 실제 점수로 평가한다."""
import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from space_z.rl.gpt_training.train import ExperimentLog, append_json, save_model, write_json
from space_z.rl.gpt_training.environment import game_configuration, make_environment
from space_z.rl.gpt_training.score_reward import reward_settings
from .environment import RotationTrainingEnvironment
from .policy import RotationContinuousPolicy


def training_environment(stage, reward_profile, max_steps):
    return Monitor(RotationTrainingEnvironment(stage, max_steps, reward_profile))


def evaluate(model, seeds, max_seconds=200, deterministic=True):
    """기본 게임·같은 seed·30Hz 제어로 평가하고 사망 이후 점수는 유지한다."""
    episodes = []
    for seed in seeds:
        hybrid = isinstance(model.policy, RotationContinuousPolicy)
        environment = (RotationTrainingEnvironment(max_steps=int(max_seconds * 30), reward_profile='legacy')
                       if hybrid else make_environment('full', int(max_seconds * 30), pure_rl=True, frame_skip=2))
        with closing(environment), torch.random.fork_rng(devices=[]):
            if not deterministic:
                torch.manual_seed(1000003 + int(seed))
            observation, info = environment.reset(seed=int(seed))
            milestones = {}
            rotations = []
            while True:
                action = model.predict(observation, deterministic=deterministic)[0]
                rotations.append(float(action[1]))
                observation, _, terminated, truncated, info = environment.step(action)
                for seconds in (60, 120, 200):
                    if seconds <= max_seconds and seconds not in milestones and info['frames'] >= seconds * 60:
                        milestones[seconds] = info['score']
                if terminated or truncated:
                    break
            episode = dict(seed=int(seed), score=info['score'], kills=info['kills'],
                           survival_seconds=info['frames'] / 60, terminated=terminated, truncated=truncated,
                           ship_class=environment.unwrapped.game.player.job,
                           stat_levels=dict(environment.unwrapped.game.progression.stat_levels),
                           **info['experiment_episode'])
            for seconds in (60, 120, 200):
                if seconds <= max_seconds:
                    episode[f'score_at_{seconds}_seconds'] = milestones.get(seconds, info['score'])
            if hybrid:
                episode['mean_absolute_rotation'] = float(np.abs(rotations).mean())
            episodes.append(episode)
    metrics = ['score', 'kills', 'survival_seconds', 'shots', 'shapes_destroyed', 'damage_taken']
    metrics += [key for key in episodes[0] if key.startswith('score_at_')]
    summary = {key: dict(mean=float(np.mean([episode[key] for episode in episodes])),
                         p25=float(np.percentile([episode[key] for episode in episodes], 25)),
                         median=float(np.median([episode[key] for episode in episodes]))) for key in metrics}
    return dict(max_seconds=max_seconds, frame_skip=2, stage='full', episodes=episodes, summary=summary,
                execution_mode='greedy' if deterministic else 'sampled', policy_seed=None if deterministic else 1000003)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--stage', choices=['aim', 'duel', 'full'], default='full')
    parser.add_argument('--reward-profile', choices=['score_v5', 'score_v7'], default='score_v7')
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--entropy', type=float, default=0.002)
    parser.add_argument('--freeze-body', action='store_true')
    parser.add_argument('--freeze-aim', action='store_true')
    parser.add_argument('--stochastic-evaluation', action='store_true')
    parser.add_argument('--seed', type=int, default=4101)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[41001, 41506, 42001, 42506])
    parent = parser.add_mutually_exclusive_group()
    parent.add_argument('--resume', type=Path)
    parent.add_argument('--initialize-discrete', type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('Name must be a single experiment directory')
    if args.steps <= 0 or args.steps % 2000 or args.seed < 0 or min(args.eval_seeds) < 0:
        parser.error('Steps must be positive multiples of 2000; seeds must be nonnegative')
    if not 0 < args.learning_rate < 1 or not 0 <= args.entropy < 1:
        parser.error('Invalid optimizer settings')
    if args.freeze_body and args.freeze_aim:
        parser.error('At least one actor component must remain trainable')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = ROOT / 'experiments' / args.name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'checkpoints').mkdir()
    with ZipFile(directory / 'source_code.zip', 'w', ZIP_DEFLATED) as archive:
        for pattern in ['*.py', '*.ps1', 'protocol.json', 'environment.lock.txt']:
            for source in ROOT.glob(pattern):
                archive.write(source, source.name)
        decoder = ROOT.parent / 'actions/rotation_continuous_action_decoder.py'
        archive.write(decoder, 'actions/' + decoder.name)
        for filename in ['environment.py', 'networks.py', 'score_reward.py', 'train.py']:
            source = ROOT.parent / 'gpt_training' / filename
            archive.write(source, 'gpt_training/' + filename)
        protected_sources = json.loads((ROOT / 'scope_before.json').read_text(encoding='utf-8'))
        for source_path, expected_hash in protected_sources.items():
            source = Path(source_path)
            if source.suffix == '.py':
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f'Protected source changed during study: {source}')
                archive.write(source, 'shared/' + source.as_posix())
        archive.write(ROOT.parent / 'rewards.py', 'shared/src/space_z/rl/rewards.py')
    manifest = dict(name=args.name, hypothesis=args.hypothesis, status='running',
                    started_at=datetime.now(timezone.utc).isoformat(),
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    observation_size=5139, training_method='pure_ppo', demonstration_steps=0,
                    action_format=['move_category_9', 'tanh_gaussian_rotation', 'bernoulli_fire', 'stat_category_6', 'evolution_category_5'],
                    process_limit=8, environment_workers=4, max_owned_python_processes=6,
                    training_game=asdict(game_configuration(args.stage)), evaluation_game=asdict(game_configuration('full')),
                    reward=asdict(reward_settings(args.reward_profile)),
                    source_sha256=hashlib.sha256((directory / 'source_code.zip').read_bytes()).hexdigest(),
                    versions=dict(python=sys.version, torch=torch.__version__))
    write_json(directory / 'experiment.json', manifest)
    environments = None
    model = None
    started = time.monotonic()
    try:
        environments = SubprocVecEnv([partial(training_environment, args.stage, args.reward_profile, 12000) for _ in range(4)], start_method='spawn')
        gamma = manifest['reward']['gamma']
        if args.resume:
            model = PPO.load(args.resume, env=environments, device='cpu', seed=args.seed,
                             gamma=gamma, learning_rate=args.learning_rate, tensorboard_log=str(directory / 'tensorboard'))
            if not isinstance(model.policy, RotationContinuousPolicy):
                raise ValueError('Resume requires a rotation-continuous checkpoint')
        else:
            model = PPO(RotationContinuousPolicy, environments, seed=args.seed, device='cpu', verbose=1,
                        n_steps=500, batch_size=250, n_epochs=4, learning_rate=args.learning_rate,
                        gamma=gamma, gae_lambda=0.95, ent_coef=args.entropy, target_kl=0.025,
                        tensorboard_log=str(directory / 'tensorboard'))
            if args.initialize_discrete:
                previous = PPO.load(args.initialize_discrete, device='cpu')
                if getattr(previous, 'training_method', None) != 'pure_ppo':
                    raise ValueError('Weight initialization requires a pure-PPO parent')
                model.policy.initialize_from_discrete(previous.policy)
                manifest['parent_training_timesteps'] = previous.num_timesteps
                del previous
        parent_path = args.resume or args.initialize_discrete
        if parent_path:
            manifest['parent_sha256'] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        model.seed = args.seed
        model.set_random_seed(args.seed)
        model.training_method = 'pure_ppo'
        model.decision_frame_skip = 2
        model.stochastic_inference = args.stochastic_evaluation
        model.ent_coef = args.entropy
        assert model.n_steps == 500 and model.n_envs == 4
        assert model.gamma == model.rollout_buffer.gamma == gamma
        for module in [model.policy.features_extractor, model.policy.mlp_extractor.policy_net, model.policy.action_net]:
            for parameter in module.parameters():
                parameter.requires_grad_(not args.freeze_body)
        for parameter in model.policy.aim_net.parameters():
            parameter.requires_grad_(not args.freeze_aim)
        model.policy.log_std.requires_grad_(not args.freeze_aim)
        manifest['initial_timesteps'] = model.num_timesteps
        manifest['ppo'] = {key: getattr(model, key) for key in ['n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'learning_rate', 'ent_coef', 'target_kl']}
        callback = ExperimentLog(directory, 50000)
        callback.next_checkpoint = model.num_timesteps + 50000
        with SummaryWriter(str(directory / 'tensorboard/evaluation')) as writer:
            writer.add_text('experiment/hypothesis', args.hypothesis, model.num_timesteps)
            def assess(label):
                checkpoint = directory / 'checkpoints' / f'evaluated_{model.num_timesteps:09d}.zip'
                save_model(model, checkpoint)
                report = evaluate(model, args.eval_seeds, deterministic=not args.stochastic_evaluation)
                record = dict(label=label, timesteps=model.num_timesteps, checkpoint=checkpoint.name,
                              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), **report)
                append_json(directory / 'evaluations.jsonl', record)
                summary = report['summary']
                ranking = [summary['score_at_120_seconds']['mean'] + summary['score_at_120_seconds']['p25'],
                           summary['score']['mean'], summary['survival_seconds']['mean']]
                if ranking > manifest.get('best_ranking', [-1, -1, -1]):
                    shutil.copyfile(checkpoint, directory / 'checkpoints/best.zip')
                    manifest.update(best_ranking=ranking, best_timesteps=model.num_timesteps, best_sha256=record['checkpoint_sha256'])
                for metric, values in summary.items():
                    writer.add_scalar('evaluation/' + metric, values['mean'], model.num_timesteps)
                    writer.add_scalar('evaluation_p25/' + metric, values['p25'], model.num_timesteps)
                writer.add_scalar('policy/rotation_std', model.policy.log_std.exp().item(), model.num_timesteps)
                writer.flush()
                write_json(directory / 'experiment.json', manifest)
                print('EVALUATION', label, model.num_timesteps, json.dumps(summary), flush=True)
            assess('before')
            final_step = model.num_timesteps + args.steps
            while model.num_timesteps < final_step:
                model.learn(min(100000, final_step - model.num_timesteps), callback=callback,
                            reset_num_timesteps=False, tb_log_name='ppo')
                save_model(model, directory / 'checkpoints/latest.zip')
                assess('progress')
                if (directory / 'stop.request').exists():
                    break
            save_model(model, directory / 'checkpoints/final.zip')
            manifest['status'] = 'stopped' if model.num_timesteps < final_step else 'completed'
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
        manifest.update(timesteps=model.num_timesteps if model else 0, elapsed_seconds=time.monotonic() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        manifest['additional_steps'] = manifest['timesteps'] - manifest.get('initial_timesteps', 0)
        write_json(directory / 'experiment.json', manifest)


if __name__ == '__main__':
    main()
