"""Model factories, isolated evaluation and atomic model/statistics snapshots."""
import copy
from contextlib import contextmanager
import importlib
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack

from .environment import TrainingEnv
from .policy import SpaceFeatures
from .settings import TrainSettings


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(_clean(value), ensure_ascii=False, allow_nan=False), encoding='utf-8')
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _one_env(cfg, rank=0, evaluation=False):
    env = Monitor(TrainingEnv(max_steps=cfg.eval_steps if evaluation else cfg.max_steps,
                              curriculum=cfg.curriculum, evaluation=evaluation, gamma=cfg.gamma,
                              score_objective=cfg.objective == 'score100',
                              corner_recovery=cfg.corner_recovery,full_growth=cfg.full_growth))
    return env


def base_vec(cfg, subprocess=False, evaluation=False):
    count = 1 if evaluation else cfg.workers
    factories = [partial(_one_env, cfg, i, evaluation) for i in range(count)]
    vec = SubprocVecEnv(factories, start_method='spawn') if subprocess and count > 1 else DummyVecEnv(factories)
    vec.seed(900001 if evaluation else cfg.seed)
    return vec


def stack(norm, history):
    order={'grid':'first','vector':'last'}
    if 'recovery' in norm.observation_space.spaces:
        order['recovery']='last'
    return VecFrameStack(norm, n_stack=history, channels_order=order)


def make_vec_env(cfg, subprocess=False, evaluation=False):
    cfg.validate()
    raw = base_vec(cfg, subprocess, evaluation)
    # Off-policy algorithms obtain get_original_obs() from VecNormalize. It must
    # therefore contain the FULL history, not an unstacked single observation.
    norm = VecNormalize(stack(raw, cfg.history), norm_obs=not cfg.full_growth, norm_reward=False,
                        norm_obs_keys=['vector'], clip_obs=10.)
    norm.training = not evaluation
    return norm


def normalizer(env):
    while not isinstance(env, VecNormalize):
        env = env.venv
    return env


def learning_rate(progress_remaining, initial=.0003):
    floor = min(.00005, initial)
    return floor + (initial-floor) * max(0., progress_remaining)


def make_model(cfg, env):
    torch.set_num_threads(cfg.torch_threads)
    kwargs = dict(features_extractor_class=SpaceFeatures, normalize_images=False)
    policy='MultiInputPolicy'
    if cfg.full_growth:
        from .growth import FullGrowthPolicy,GrowthFeatures
        policy=FullGrowthPolicy
        kwargs.update(features_extractor_class=GrowthFeatures,combat_recovery=cfg.corner_recovery)
    if cfg.algorithm == 'ppo':
        kwargs.update(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        return PPO(policy, env, learning_rate=partial(learning_rate, initial=cfg.learning_rate), n_steps=cfg.rollout_steps,
                   batch_size=min(cfg.batch_size, cfg.rollout_steps*cfg.workers), n_epochs=cfg.epochs,
                   gamma=cfg.gamma, gae_lambda=.95, clip_range=.15, ent_coef=.02, vf_coef=.5,
                   max_grad_norm=.5, target_kl=.025, policy_kwargs=kwargs, seed=cfg.seed, device='cpu')
    kwargs.update(net_arch=[256, 128])
    return DQN('MultiInputPolicy', env, learning_rate=.0001, gamma=cfg.gamma,
               buffer_size=2000, learning_starts=3000, batch_size=128, train_freq=4,
               gradient_steps=1, target_update_interval=2000, exploration_fraction=.25,
               exploration_final_eps=.03, policy_kwargs=kwargs, seed=cfg.seed, device='cpu')


def publish_checkpoint(model, env, run_dir, cfg, best=False, evaluation=None):
    run_dir = Path(run_dir)
    directory = run_dir/'checkpoints'/f'{model.num_timesteps:012d}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    model.save(directory/'model.zip')
    normalizer(env).save(directory/'obs_stats.pkl')
    entry = {'directory': directory.relative_to(run_dir).as_posix(),
             'step': model.num_timesteps, 'settings': cfg.to_dict(), 'format': 3}
    if evaluation is not None:
        entry['evaluation'] = {key: evaluation[key] for key in
                              ('mean', 'std', 'kills', 'seconds', 'horizon', 'metric', 'score')
                              if key in evaluation}
    atomic_json(directory/'manifest.json', entry)
    atomic_json(run_dir/'latest.json', entry)
    if best:
        atomic_json(run_dir/'best.json', entry)
    # Keep a bounded history, plus whichever generation holds the best model.
    preserved = {entry['directory']}
    best_entry = read_json(run_dir/'best.json', {})
    if best_entry.get('directory'):
        preserved.add(best_entry['directory'])
    generations = sorted((run_dir/'checkpoints').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    preserved.update(p.relative_to(run_dir).as_posix() for p in generations[:3])
    for p in generations[3:]:
        if p.is_dir() and p.relative_to(run_dir).as_posix() not in preserved:
            try:
                shutil.rmtree(p)
            except OSError:
                pass  # A viewer may still have a file open on Windows.
    return entry


def checkpoint_path(run_dir, entry):
    root = Path(run_dir).resolve()
    directory = (root/entry['directory']).resolve()
    if root not in directory.parents or not (directory/'manifest.json').is_file():
        raise ValueError('Invalid or incomplete checkpoint generation')
    return directory


@contextmanager
def legacy_checkpoint_modules():
    """Resolve pre-rename pickle classes only while loading saved models."""
    legacy = 'space_z.rl.' + 'training'
    current = __package__
    aliases = {legacy: importlib.import_module(current)}
    for name in ('policy', 'growth', 'runtime', 'settings'):
        aliases[f'{legacy}.{name}'] = importlib.import_module(f'{current}.{name}')
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_checkpoint(run_dir, entry, training=False, subprocess=False, cfg=None, evaluation=False):
    if entry.get('format') != 3:
        raise ValueError('Checkpoint uses an incompatible observation pipeline; start a new run')
    directory = checkpoint_path(run_dir, entry)
    cfg = cfg or TrainSettings(**entry['settings'])
    raw = base_vec(cfg, subprocess=subprocess, evaluation=evaluation)
    try:
        raw.env_method('set_training_step', entry['step'])
        env = VecNormalize.load(str(directory/'obs_stats.pkl'), stack(raw, cfg.history))
        env.training, env.norm_reward = training, False
        cls = PPO if cfg.algorithm == 'ppo' else DQN
        with legacy_checkpoint_modules():
            model = cls.load(directory/'model.zip', env=env, device='cpu')
    except Exception:
        raw.close()
        raise
    return model, env


def evaluate(model, training_env, cfg, random_policy=False, heartbeat=None, seed_start=None):
    """Held-out evaluation: fixed full difficulty and a copy of frozen statistics."""
    env = make_vec_env(cfg, evaluation=True)
    seed_start = cfg.evaluation_seed if seed_start is None else seed_start
    training_norm=normalizer(training_env)
    normalizer(env).norm_obs=training_norm.norm_obs
    if training_norm.norm_obs:
        normalizer(env).obs_rms=copy.deepcopy(training_norm.obs_rms)
    episodes = []
    rng = np.random.default_rng(918273)
    try:
        for i in range(cfg.eval_episodes):
            env.seed(seed_start+i)
            obs = env.reset()
            while True:
                if heartbeat is not None and not heartbeat():
                    return None
                if random_policy:
                    action=np.array([[rng.integers(int(n)) for n in env.action_space.nvec]]) if cfg.full_growth else [rng.integers(env.action_space.n)]
                else:
                    action=model.predict(obs, deterministic=True)[0]
                obs, _, done, info = env.step(action)
                if done[0]:
                    episodes.append(info[0]['episode_metrics'])
                    break
        returns = [e['score'] if cfg.objective == 'score100' else e['task_return'] for e in episodes]
        return {'step': model.num_timesteps, 'kind': 'random' if random_policy else 'model',
                'mean': float(np.mean(returns)), 'std': float(np.std(returns)),
                'kills': float(np.mean([e['kills'] for e in episodes])),
                'seconds': float(np.mean([e['seconds'] for e in episodes])),
                'score': float(np.mean([e['score'] for e in episodes])),
                'metric': 'score' if cfg.objective == 'score100' else 'reward',
                'level':float(np.mean([e['level'] for e in episodes])),
                'max_level':max(e['level'] for e in episodes),
                'stat_upgrades':float(np.mean([e['stat_upgrades'] for e in episodes])),
                'evolutions':float(np.mean([e['evolutions'] for e in episodes])),
                'masteries':float(np.mean([e['masteries'] for e in episodes])),
                'ships_observed':sorted({ship for e in episodes for ship in e['ship_history']}),
                'episodes': episodes, 'horizon': cfg.eval_steps, 'seeds': list(range(seed_start, seed_start+cfg.eval_episodes))}
    finally:
        env.close()
