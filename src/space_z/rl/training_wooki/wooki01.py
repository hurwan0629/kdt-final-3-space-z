"""Run a resumable job. All dashboard data comes from actual callbacks/evaluations."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from filelock import FileLock

from .runtime import (atomic_json, read_json, make_vec_env, make_model, evaluate,
                      load_checkpoint, publish_checkpoint)
from .settings import TrainSettings, stage_for_step


class LiveCallback(BaseCallback):
    def __init__(self, directory, cfg, previous=None):
        super().__init__()
        self.directory, self.cfg = Path(directory), cfg
        previous = previous or {}
        self.episodes = deque(previous.get('episodes', []), maxlen=120)
        self.evaluations = deque(previous.get('evaluations', []), maxlen=100)
        self.history = deque(previous.get('history', []), maxlen=500)
        self.baseline = previous.get('baseline')
        means = [e['mean'] for e in self.evaluations if e.get('kind') == 'model']
        # The best may predate the bounded plot history. The pointer's embedded
        # score also survives a crash between checkpoint and status publication.
        stored_best = read_json(self.directory/'best.json', {}).get('evaluation', {}).get('mean')
        candidates = means + [previous.get('best'), stored_best]
        self.best = max((v for v in candidates if isinstance(v, (int, float)) and np.isfinite(v)),
                        default=-float('inf'))
        self.started = time.monotonic()
        self.last_write = self.last_checkpoint = self.last_eval = 0
        self.start_step = 0
        self.stop_requested = False
        self.action_counts = np.zeros(54, dtype=np.int64)
        self.episode_count = int(previous.get('episode_count', 0))

    def append_csv(self, name, row):
        path = self.directory/name
        exists = path.exists() and path.stat().st_size > 0
        with path.open('a', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def state(self, status='training', force=False):
        now = time.monotonic()
        if not force and now-self.last_write < 1.:
            return
        self.last_write = now
        elapsed = now-self.started
        step = int(self.model.num_timesteps)
        fps = max(0, step-self.start_step)/max(elapsed, .01)
        loss = {k.split('/')[-1]: float(v) for k, v in self.model.logger.name_to_value.items()
                if k.startswith('train/') and isinstance(v, (int, float, np.number))}
        data = {'status': status, 'pid': os.getpid(), 'updated': time.time(),
                'algorithm': self.cfg.algorithm, 'step': step, 'target': self.cfg.total_steps,
                'workers': self.cfg.workers, 'elapsed': elapsed, 'fps': fps,
                'eta': (self.cfg.total_steps-step)/fps if fps > 0 else None,
                'stage': stage_for_step(step, self.cfg.curriculum),
                'updates': int(getattr(self.model, '_n_updates', 0)), 'loss': loss,
                'episode_count': self.episode_count, 'episodes': list(self.episodes),
                'evaluations': list(self.evaluations), 'history': list(self.history),
                'baseline': self.baseline, 'best': self.best if np.isfinite(self.best) else None,
                'action_counts': self.action_counts.tolist(), 'settings': self.cfg.to_dict()}
        try:
            atomic_json(self.directory/'status.json', data)
        except OSError as error:
            warnings.warn(f'Monitor status could not be saved; training continues: {error}')

    def control(self, status='training'):
        command = read_json(self.directory/'control.json', {}).get('command', 'run')
        while command == 'pause':
            self.state('paused')
            time.sleep(.15)
            command = read_json(self.directory/'control.json', {}).get('command', 'run')
        self.stop_requested = command == 'stop'
        self.state('stopping' if self.stop_requested else status)
        return not self.stop_requested

    def assess(self):
        self.state('evaluating', True)
        result = evaluate(self.model, self.training_env, self.cfg,
                          heartbeat=lambda: self.control('evaluating'))
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

    def _on_training_start(self):
        self.start_step = self.last_checkpoint = self.model.num_timesteps
        self.last_eval = self.model.num_timesteps
        self.state('starting', True)

    def _on_rollout_start(self):
        self.training_env.env_method('set_training_step', self.model.num_timesteps)
        if self.cfg.algorithm == 'ppo':
            progress = min(1., self.model.num_timesteps/self.cfg.total_steps)
            self.model.ent_coef = min(self.model.ent_coef, .02 - .016*progress)
        if self.model.num_timesteps-self.last_checkpoint >= self.cfg.checkpoint_every:
            publish_checkpoint(self.model, self.training_env, self.directory, self.cfg)
            self.last_checkpoint = self.model.num_timesteps
        if self.model.num_timesteps-self.last_eval >= self.cfg.eval_every and self.control():
            self.assess()
        if self.model.num_timesteps > self.start_step:
            point = {'step': self.model.num_timesteps,
                     'episode_mean': float(np.mean([e['score'] if self.cfg.objective == 'score100' else e['task_return'] for e in self.episodes])) if self.episodes else None}
            self.history.append(point)
        self.state(force=True)

    def _on_step(self):
        actions=np.asarray(self.locals.get('actions', []),dtype=int)
        if self.cfg.full_growth and actions.size:
            actions=actions[:,0]
        for action in actions.ravel():
            self.action_counts[action] += 1
        for info in self.locals.get('infos', []):
            if 'episode_metrics' in info:
                ep = dict(info['episode_metrics'], global_step=self.num_timesteps)
                self.episodes.append(ep)
                self.episode_count += 1
                self.append_csv('episodes.csv', dict(step=self.num_timesteps,
                    **{k: ep[k] for k in ('return', 'task_return', 'kills', 'damage', 'hurt',
                                         'shapes', 'seconds', 'death', 'stage', 'seed')},
                    **({'score': ep['score']} if self.cfg.objective == 'score100' else {}),
                    **({'corner_recoveries': ep['corner_recoveries']} if self.cfg.corner_recovery else {}),
                    **({k:ep[k] for k in ('level','stat_upgrades','evolutions','masteries','upgrade_points')}
                       if self.cfg.full_growth else {}),
                    components=json.dumps(ep['components'])))
        return self.control()


def _run_training(directory, cfg, resume=False, subprocess=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    previous = read_json(directory/'status.json', {}) if resume else {}
    if resume:
        entry = read_json(directory/'latest.json')
        if not entry:
            raise ValueError('No valid latest checkpoint exists to resume')
        if entry['settings']['algorithm'] != cfg.algorithm:
            raise ValueError('Cannot resume a checkpoint with another algorithm')
        model, env = load_checkpoint(directory, entry, training=True, subprocess=subprocess, cfg=cfg)
        # Keep DQN replay memory as well as network/optimizer state across resumes.
        replay = directory/'replay_buffer.pkl'
        if cfg.algorithm == 'dqn' and replay.exists():
            model.load_replay_buffer(replay)
    else:
        env = make_vec_env(cfg, subprocess=subprocess)
        model = make_model(cfg, env)
    callback = LiveCallback(directory, cfg, previous)
    model.set_logger(__import__('stable_baselines3.common.logger', fromlist=['configure']).configure(None, []))
    callback.init_callback(model)
    atomic_json(directory/'config.json', cfg.to_dict())
    try:
        publish_checkpoint(model, env, directory, cfg)
        callback.start_step = model.num_timesteps
        callback.state('baseline', True)
        if callback.baseline is None and callback.control('baseline'):
            callback.baseline = evaluate(model, env, cfg, random_policy=True,
                                         heartbeat=lambda: callback.control('baseline'))
            atomic_json(directory/'random_baseline.json', callback.baseline)
        if callback.control():
            remaining = cfg.total_steps-model.num_timesteps
            if remaining > 0:
                model.learn(remaining, callback=callback, reset_num_timesteps=False)
            if not callback.stop_requested:
                callback.assess()
        publish_checkpoint(model, env, directory, cfg)
        if cfg.algorithm == 'dqn':
            model.save_replay_buffer(directory/'replay_buffer.pkl')
        callback.state('stopped' if callback.stop_requested else 'completed', True)
    except KeyboardInterrupt:
        publish_checkpoint(model, env, directory, cfg)
        callback.state('stopped', True)
    except Exception as error:
        callback.state('failed', True)
        state = read_json(directory/'status.json', {})
        state['error'], state['traceback'] = str(error), traceback.format_exc()
        atomic_json(directory/'status.json', state)
        raise
    finally:
        env.close()


def run_training(directory, cfg, resume=False, subprocess=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent two launchers from overwriting the same run's checkpoints/status.
    with FileLock(str(directory/'job.lock'), timeout=0):
        return _run_training(directory, cfg, resume, subprocess)


def parser(require_run=True):
    p = argparse.ArgumentParser(description='Space Z live reinforcement learning')
    p.add_argument('--run-dir', required=require_run)
    p.add_argument('--algorithm', choices=['ppo', 'dqn'], default='ppo')
    p.add_argument('--steps', type=int, default=2000000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-curriculum', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--full-growth',action='store_true')
    p.add_argument('--init-combat',help='Best existing combat run to transfer into a new growth run')
    return p


def settings_from_args(args):
    cfg = TrainSettings(algorithm=args.algorithm, total_steps=args.steps,
                        workers=args.workers if args.algorithm == 'ppo' else 1,
                        seed=args.seed, curriculum=not args.no_curriculum)
    if args.full_growth:
        cfg.full_growth=True;cfg.corner_recovery=True;cfg.curriculum=False
        cfg.objective='score100';cfg.max_steps=9000;cfg.eval_steps=1500;cfg.gamma=.9995
    if args.resume:
        saved = read_json(Path(args.run_dir)/'config.json')
        if not saved:
            raise ValueError('No saved experiment configuration')
        cfg = TrainSettings(**saved)
        cfg.total_steps = args.steps
    if args.smoke:
        cfg.total_steps = 64 if cfg.algorithm == 'ppo' else 3200
        cfg.workers, cfg.rollout_steps, cfg.batch_size, cfg.epochs = 1, 16, 16, 1
        cfg.max_steps, cfg.eval_steps, cfg.eval_episodes = 32, 8, 1
        cfg.eval_every, cfg.checkpoint_every = 100000, 32
    return cfg.validate()


def main():
    args = parser().parse_args()
    try:
        if args.init_combat:
            if args.resume:
                raise ValueError('Use --init-combat for a new run or --resume for an existing run')
            from .growth import migrate_combat
            cfg=migrate_combat(args.init_combat,args.run_dir)
            cfg.total_steps=args.steps;cfg.workers=args.workers
            atomic_json(Path(args.run_dir)/'config.json',cfg.to_dict())
            run_training(args.run_dir,cfg,resume=True)
            return
        run_training(args.run_dir, settings_from_args(args), resume=args.resume)
    except Exception as error:
        state = read_json(Path(args.run_dir)/'status.json', {})
        # A competing launcher must not overwrite the active job's status.
        if state.get('pid') in (None, os.getpid()):
            state.update(status='failed', error=str(error), traceback=traceback.format_exc())
            atomic_json(Path(args.run_dir)/'status.json', state)
        raise


if __name__ == '__main__':
    main()
