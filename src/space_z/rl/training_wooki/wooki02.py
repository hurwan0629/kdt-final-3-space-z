"""Fresh PPO growth training without scripted combat, with hurt/stall penalties.

Run: python -m space_z.rl.training_wooki.wooki02
Resume this variant only: append --resume. Stop safely via control.json command=stop.
Reward: score delta / 200 - death * .5 - actual damage * .01 - wall stall * .01.
"""
from pathlib import Path
import os
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'space_z.rl.training_wooki'

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import argparse
from collections import deque
from functools import partial
import traceback

import numpy as np
import torch
from gymnasium import spaces
from filelock import FileLock
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize

from .environment import TrainingEnv
from .growth import STAT_COUNT, legal_growth_masks
from .policy import SpaceFeatures
from .runtime import atomic_json, read_json, publish_checkpoint, checkpoint_path, learning_rate
from .settings import TrainSettings
from .wooki01 import LiveCallback

RULES = {'variant': 'wooki02', 'damage_per_hp': .01, 'stall_per_step': .01,
         'stall_window_steps': 12, 'stall_span_px': 18., 'wall_clearance_px': 8.,
         'scripted_combat': False, 'pretrained_combat': False}


def penalty_components(damage_taken, stalled):
    return {'hurt': -RULES['damage_per_hp'] * max(0., float(damage_taken)),
            'wall_stall': -RULES['stall_per_step'] if stalled else 0.}


class WallStallTracker:
    """Measure 12 uninterrupted movement attempts; never choose an escape action."""
    def __init__(self, position):
        self.positions = deque([np.asarray(position, dtype=float)], maxlen=13)
        self.attempts = 0
        self.span = 0.

    def update(self, position, attempted, near_wall):
        position = np.asarray(position, dtype=float)
        if not attempted or not near_wall:
            self.positions.clear()
            self.positions.append(position)
            self.attempts = 0
            self.span = 0.
            return False
        self.positions.append(position)
        self.attempts += 1
        self.span = float(np.linalg.norm(np.ptp(np.asarray(self.positions), axis=0)))
        return self.attempts >= 12 and self.span < RULES['stall_span_px']

    def observation(self):
        return np.array([min(self.attempts / 12., 1.),
                         min(self.span / 18., 1.)], dtype=np.float32)


class PenaltyTrainingEnv(TrainingEnv):
    def __init__(self, **kwargs):
        super().__init__(score_objective=True, full_growth=True, corner_recovery=False, **kwargs)
        self.observation_space.spaces['stall'] = spaces.Box(0., 1., (2,), np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        p = self.env.game.player
        self.stall_tracker = WallStallTracker((p.x, p.y))
        self.stall_steps = 0
        obs['stall'] = self.stall_tracker.observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        p = self.env.game.player
        # Read-only geometric clearance, including map borders and breakable walls.
        near_wall = not self.env.game.terrain.circle_fits(
            p.x, p.y, p.radius + RULES['wall_clearance_px'])
        stalled = self.stall_tracker.update((p.x, p.y), int(action[0]) // 6 != 0, near_wall)
        self.stall_steps += int(stalled)
        parts = penalty_components(info['events']['damage_taken'], stalled)
        extra = sum(parts.values())
        reward += extra
        self.episode_return += extra
        self.task_return += extra
        info['reward_components'].update(parts)
        info['task_reward'] += extra
        for key, value in parts.items():
            self._components[key] = self._components.get(key, 0.) + value
        obs['stall'] = self.stall_tracker.observation()
        info.update(wall_stalled=bool(stalled), wall_near=bool(near_wall))
        if terminated or truncated:
            info['episode_metrics'].update(
                **{'return': self.episode_return, 'task_return': self.task_return,
                   'components': dict(self._components), 'wall_stall_steps': self.stall_steps})
        return obs, float(reward), terminated, truncated, info


class LearnedFeatures(SpaceFeatures):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self._features_dim += int(np.prod(observation_space['stall'].shape))

    def forward(self, observations):
        return torch.cat((super().forward(observations), observations['stall']), dim=1)


class LearnedGrowthPolicy(MultiInputActorCriticPolicy):
    """All 54 combat choices are learned, with no tactical masks or frozen prior."""
    def _distribution_and_value(self, obs):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        output = self.action_net(pi)
        combat = output[:, :54]
        stat = output[:, 54:54 + STAT_COUNT]
        evolution = output[:, 54 + STAT_COUNT:]
        stat_mask, evolution_mask = legal_growth_masks(obs['vector'])
        # Keep the original growth spending preference; only combat assistance is removed.
        wait_bias = torch.zeros_like(stat)
        wait_bias[:, 0] = stat_mask[:, 1:].any(1) * 2.
        stat = (stat - wait_bias).masked_fill(~stat_mask, -1e4)
        evolution = evolution.masked_fill(~evolution_mask, -1e4)
        distribution = self.action_dist.proba_distribution(
            action_logits=torch.cat((combat, stat, evolution), dim=1))
        return distribution, self.value_net(vf)

    def get_distribution(self, obs):
        return self._distribution_and_value(obs)[0]

    def forward(self, obs, deterministic=False):
        distribution, values = self._distribution_and_value(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution, values = self._distribution_and_value(obs)
        return values, distribution.log_prob(actions), distribution.entropy()


def one_env(cfg, evaluation=False):
    return Monitor(PenaltyTrainingEnv(max_steps=cfg.eval_steps if evaluation else cfg.max_steps,
                   curriculum=cfg.curriculum, evaluation=evaluation, gamma=cfg.gamma))


def raw_env(cfg, evaluation=False, subprocess=True):
    count = 1 if evaluation else cfg.workers
    factories = [partial(one_env, cfg, evaluation) for _ in range(count)]
    env = SubprocVecEnv(factories, start_method='spawn') if subprocess and count > 1 else DummyVecEnv(factories)
    env.seed(cfg.evaluation_seed if evaluation else cfg.seed)
    return env


def stacked(env, cfg):
    return VecFrameStack(env, n_stack=cfg.history,
                         channels_order={'grid': 'first', 'vector': 'last', 'stall': 'last'})


def make_env(cfg, evaluation=False, subprocess=True):
    env = VecNormalize(stacked(raw_env(cfg, evaluation, subprocess), cfg),
                       norm_obs=False, norm_reward=False)
    env.training = not evaluation
    return env


def make_model(cfg, env):
    return PPO(LearnedGrowthPolicy, env,
               policy_kwargs=dict(features_extractor_class=LearnedFeatures,
                                  net_arch=dict(pi=[128, 128], vf=[128, 128]), normalize_images=False),
               learning_rate=partial(learning_rate, initial=cfg.learning_rate),
               n_steps=cfg.rollout_steps, batch_size=cfg.batch_size, n_epochs=cfg.epochs,
               gamma=cfg.gamma, gae_lambda=.95, clip_range=.15, ent_coef=.02,
               vf_coef=.5, max_grad_norm=.5, target_kl=.025, seed=cfg.seed, device='cpu')


def load_model(directory, cfg, subprocess=True):
    entry = read_json(directory / 'latest.json')
    if not entry or entry.get('format') != 3:
        raise ValueError('No valid wooki02 checkpoint')
    path = checkpoint_path(directory, entry)
    raw = raw_env(cfg, subprocess=subprocess)
    try:
        env = VecNormalize.load(str(path / 'obs_stats.pkl'), stacked(raw, cfg))
        env.training, env.norm_reward = True, False
        model = PPO.load(path / 'model.zip', env=env, device='cpu')
        if not isinstance(model.policy, LearnedGrowthPolicy):
            raise ValueError('Only wooki02 models may be resumed here')
        return model, env
    except Exception:
        raw.close()
        raise


def evaluate(model, cfg, random_policy=False, heartbeat=None):
    env = make_env(cfg, evaluation=True, subprocess=False)
    episodes = []
    rng = np.random.default_rng(918273)
    try:
        for i in range(cfg.eval_episodes):
            env.seed(cfg.evaluation_seed + i)
            obs = env.reset()
            while True:
                if heartbeat is not None and not heartbeat():
                    return None
                action = np.array([[rng.integers(int(n)) for n in env.action_space.nvec]]) if random_policy else model.predict(obs, deterministic=True)[0]
                obs, _, done, infos = env.step(action)
                if done[0]:
                    episodes.append(infos[0]['episode_metrics'])
                    break
        scores = [e['score'] for e in episodes]
        return dict(step=model.num_timesteps, kind='random' if random_policy else 'model',
                    mean=float(np.mean(scores)), std=float(np.std(scores)), score=float(np.mean(scores)),
                    kills=float(np.mean([e['kills'] for e in episodes])),
                    seconds=float(np.mean([e['seconds'] for e in episodes])),
                    horizon=cfg.eval_steps, metric='score', episodes=episodes,
                    reward_mean=float(np.mean([e['return'] for e in episodes])))
    finally:
        env.close()


class PenaltyCallback(LiveCallback):
    def assess(self):
        self.state('evaluating', True)
        result = evaluate(self.model, self.cfg, heartbeat=lambda: self.control('evaluating'))
        if result is not None:
            self.evaluations.append(result)
            better = result['mean'] > self.best
            if better:
                self.best = result['mean']
            publish_checkpoint(self.model, self.training_env, self.directory, self.cfg,
                               best=better, evaluation=result)
            self.append_csv('evaluations.csv', {k: result[k] for k in
                            ('step', 'kind', 'mean', 'std', 'kills', 'seconds', 'horizon')})
            self.last_eval = self.model.num_timesteps
        self.state('training', True)


def run_training(directory, cfg, resume=False, subprocess=True):
    cfg.validate()
    if cfg.algorithm != 'ppo' or not cfg.full_growth or cfg.corner_recovery or cfg.objective != 'score100':
        raise ValueError('wooki02 requires growth PPO with score rewards and no combat recovery')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / 'job.lock'), timeout=0):
        marker = directory / 'wooki02.json'
        if resume:
            if read_json(marker) != RULES:
                raise ValueError('Resume requires a wooki02 run with matching reward rules')
        elif any(directory.iterdir()) and any(p.name != 'job.lock' for p in directory.iterdir()):
            raise ValueError('New training requires an empty run folder; use --resume for wooki02')
        atomic_json(marker, RULES)
        torch.set_num_threads(cfg.torch_threads)
        model, env = load_model(directory, cfg, subprocess) if resume else (None, make_env(cfg, subprocess=subprocess))
        callback = None
        try:
            if model is None:
                model = make_model(cfg, env)
            previous = read_json(directory / 'status.json', {}) if resume else {}
            callback = PenaltyCallback(directory, cfg, previous)
            model.set_logger(__import__('stable_baselines3.common.logger', fromlist=['configure']).configure(None, []))
            callback.init_callback(model)
            callback.start_step = model.num_timesteps
            atomic_json(directory / 'config.json', cfg.to_dict())
            atomic_json(directory / 'control.json', {'command': 'run'})
            publish_checkpoint(model, env, directory, cfg)
            callback.state('baseline', True)
            if callback.baseline is None:
                callback.baseline = evaluate(model, cfg, random_policy=True,
                                             heartbeat=lambda: callback.control('baseline'))
                if callback.baseline is not None:
                    atomic_json(directory / 'random_baseline.json', callback.baseline)
            remaining = cfg.total_steps - model.num_timesteps
            if remaining > 0 and callback.control():
                model.learn(remaining, callback=callback, reset_num_timesteps=False)
            if not callback.stop_requested:
                callback.assess()
            publish_checkpoint(model, env, directory, cfg)
            callback.state('stopped' if callback.stop_requested else 'completed', True)
        except KeyboardInterrupt:
            if model is not None:
                publish_checkpoint(model, env, directory, cfg)
            if callback is not None:
                callback.state('stopped', True)
            else:
                atomic_json(directory / 'status.json', {'status': 'stopped', 'pid': os.getpid()})
        except Exception as error:
            atomic_json(directory / 'status.json', dict(status='failed', error=str(error),
                                                       traceback=traceback.format_exc(), pid=os.getpid()))
            raise
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parents[4] / 'runs/ppo-wooki02')
    parser.add_argument('--steps', type=int, default=2_000_000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.resume:
        saved = read_json(args.run_dir / 'config.json')
        if not saved:
            parser.error('No saved wooki02 configuration')
        cfg = TrainSettings(**saved)
        cfg.total_steps = args.steps
    else:
        cfg = TrainSettings(total_steps=args.steps, workers=args.workers, full_growth=True,
                            corner_recovery=False, objective='score100', curriculum=False,
                            max_steps=9000, eval_steps=1500, eval_episodes=6, eval_every=16384,
                            gamma=.9995, torch_threads=1)
    if args.smoke:
        cfg.total_steps = 64
        cfg.workers, cfg.rollout_steps, cfg.batch_size, cfg.epochs = 1, 16, 16, 1
        cfg.max_steps, cfg.eval_steps, cfg.eval_episodes = 16, 8, 1
    run_training(args.run_dir, cfg.validate(), resume=args.resume)


if __name__ == '__main__':
    # Persist importable class paths even when launched as a script or with -m.
    from space_z.rl.training_wooki.wooki02 import main as entrypoint
    entrypoint()
