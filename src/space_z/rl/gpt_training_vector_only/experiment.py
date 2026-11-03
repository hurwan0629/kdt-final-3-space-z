"""새 정책의 예비 학습·평가·기록을 순서대로 실행한다."""
import argparse
from collections import deque
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import psutil
import torch
import stable_baselines3
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from .environment import REWARD_COEFFICIENTS, make_environment, environment_configuration
from .policy import VectorOnlyPolicy
from .metrics import collect_telemetry, empty_telemetry, episode_metrics
from .storage import ROOT, append_json, atomic_replace, event, now, sha256, write_json

PPO_OPTIONS = dict(n_steps=256, batch_size=256, n_epochs=4, learning_rate=0.0003,
                   gamma=0.9975, gae_lambda=0.95, ent_coef=0.02, clip_range=0.15,
                   target_kl=0.025)
EVALUATION_SEEDS = [710101, 710202, 710303]
REWARD_COEFFICIENT_NAMES = set(REWARD_COEFFICIENTS)


def update_report():
    """갱신은 jsonl 원본에서 언제든 재생성 가능한 비필수 대시보드라, 저장 실패가 학습을 죽이게 하지 않는다."""
    from .report import build_report
    try:
        build_report()
    except OSError as error:
        print('REPORT_UPDATE_FAILED', repr(error), flush=True)


def save_model(model, path):
    temporary = path.with_name(path.stem + '.writing.zip')
    model.save(temporary)
    atomic_replace(temporary, path)


def parse_reward_overrides(entries):
    if not entries:
        return None
    overrides = {}
    for entry in entries:
        key, separator, value = entry.partition('=')
        if not separator or key not in REWARD_COEFFICIENT_NAMES:
            raise ValueError(f'Invalid --reward-overrides entry: {entry!r}')
        overrides[key] = float(value)
    return overrides


def check_process_budget(maximum):
    """학습 본체·자식·Python 실행기를 포함한 프로세스 상한을 확인한다."""
    process = psutil.Process()
    launcher = process.parent()
    children = process.children(recursive=True)
    count = 1 + len(children) + int(launcher is not None and launcher.name().lower().startswith('python'))
    if count > maximum:
        raise RuntimeError(f'Owned process budget exceeded: {count} > {maximum}')
    return dict(owned_process_count=count, worker_pids=[child.pid for child in children])


def new_experiment(name, title, kind, goal, config, parent=None):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Experiment name must be one directory name')
    directory = ROOT / 'experiments' / name
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ['logs', 'checkpoints', 'replays']:
        (directory / folder).mkdir()
    write_json(directory / 'config.json', config)
    protected = json.loads((ROOT / 'agent_context/protected_sources.json').read_text(encoding='utf-8'))
    project = ROOT.parents[3]
    for source, expected in protected.items():
        if sha256(project / source) != expected:
            raise RuntimeError(f'Protected source changed before experiment: {source}')
    with ZipFile(directory / 'source_snapshot.zip', 'w', ZIP_DEFLATED) as archive:
        for source in ROOT.glob('*.py'):
            archive.write(source, 'branch/' + source.name)
        template = ROOT / 'report_template.html'
        if template.exists():
            archive.write(template, 'branch/' + template.name)
        archive.write(ROOT / 'reports/architecture.html', 'reports/architecture.html')
        for source in protected:
            archive.write(project / source, source.replace('\\', '/'))
    manifest = dict(id=name, title=title, kind=kind, goal=goal, created_at=now(),
                    initialization='resume' if parent else 'random', parent=parent,
                    input_contract='original_vector_5139_only',
                    architecture=dict(name='vector_only_cross_attention_v1', width=64, heads=4,
                                      context_tokens=143, blocks=1, action_parameters=22),
                    architecture_sha256=sha256(ROOT / 'reports/architecture.html'),
                    config_sha256=sha256(directory / 'config.json'),
                    source_snapshot_sha256=sha256(directory / 'source_snapshot.zip'),
                    shared_source_sha256=protected,
                    versions=dict(python=sys.version, torch=torch.__version__,
                                  stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__),
                    process_policy=('SubprocVecEnv workers' if config.get('parallel') else 'DummyVecEnv in one process')
                                   + f'; evaluation is sequential; maximum {config.get("maximum_owned_processes", 18)} owned processes',
                    distribution_dependency='Only RotationHybridDistribution is reused; no parent policy or aim_features is called')
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'state.json', dict(status='prepared', timesteps=0, updated_at=now(), process_id=os.getpid()))
    event(directory, 'prepared', goal, initialization=manifest['initialization'], parent=parent)
    return directory


class TrainingJsonWriter(KVWriter):
    """TensorBoard와 같은 시점의 스칼라 지표를 JSONL로 남긴다."""

    def __init__(self, directory):
        self.directory = directory

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for name, value in key_values.items():
            if isinstance(value, Real) and math.isfinite(float(value)):
                metrics[name] = float(value)
        append_json(self.directory / 'logs/training.jsonl', dict(time=now(), timesteps=int(step), metrics=metrics))

    def close(self):
        pass


class EpisodeRecorder(BaseCallback):
    """에피소드 결과와 학습 진행 상태를 기록한다."""

    def __init__(self, directory, environment_count, start_time, maximum_owned_processes=18):
        super().__init__()
        self.directory = directory
        self.start_time = start_time
        self.maximum_owned_processes = maximum_owned_processes
        self.components = [dict() for _ in range(environment_count)]
        self.reward_inputs = [dict() for _ in range(environment_count)]
        self.shots = [0] * environment_count
        self.telemetry = [empty_telemetry() for _ in range(environment_count)]
        self.recent_metrics = deque(maxlen=20)
        self.episode_writer = SummaryWriter(str(directory / 'logs/tensorboard/episodes'))

    def _on_step(self):
        for index, info in enumerate(self.locals['infos']):
            collect_telemetry(self.telemetry[index], info, self.locals['actions'][index])
            for name, value in info['reward_components'].items():
                self.components[index][name] = self.components[index].get(name, 0) + value
            for name, value in info['reward_inputs'].items():
                self.reward_inputs[index][name] = self.reward_inputs[index].get(name, 0) + value
            self.shots[index] += info['events']['player_shots_fired']
            if 'episode' in info:
                record = dict(time=now(), timesteps=self.num_timesteps, environment_index=index,
                              score=info['score'], kills=info['kills'], survival_seconds=info['frames'] / 60,
                              episode_reward=info['episode']['r'], episode_steps=info['episode']['l'],
                              shots_fired=self.shots[index], job=info['job'], reward_components=self.components[index],
                              reward_inputs=self.reward_inputs[index],
                              telemetry=self.telemetry[index],
                              truncated=bool(info.get('TimeLimit.truncated', False)))
                append_json(self.directory / 'logs/episodes.jsonl', record)
                scalars = episode_metrics(record)
                self.recent_metrics.append(scalars)
                for name, value in scalars.items():
                    self.logger.record_mean(name, value)
                    self.episode_writer.add_scalar('episode/' + name, value, self.num_timesteps)
                    mean = sum(row[name] for row in self.recent_metrics) / len(self.recent_metrics)
                    self.episode_writer.add_scalar('recent20/' + name, mean, self.num_timesteps)
                self.components[index] = {}
                self.reward_inputs[index] = {}
                self.shots[index] = 0
                self.telemetry[index] = empty_telemetry()
        return not (self.directory / 'stop.request').exists()

    def _on_rollout_end(self):
        process_usage = check_process_budget(self.maximum_owned_processes)
        self.logger.record('resources/owned_python_processes', process_usage['owned_process_count'])
        state = json.loads((self.directory / 'state.json').read_text(encoding='utf-8'))
        state.update(status='training', timesteps=self.num_timesteps, updated_at=now(),
                     additional_timesteps=self.num_timesteps - state.get('initial_timesteps', 0),
                     process_id=os.getpid(), elapsed_seconds=time.monotonic() - self.start_time,
                     owned_process_count=process_usage['owned_process_count'],
                     maximum_owned_processes=self.maximum_owned_processes)
        write_json(self.directory / 'state.json', state)


def evaluate(model, directory, timesteps, seeds=EVALUATION_SEEDS, max_seconds=120, capture=False, reward_overrides=None):
    environment = make_environment(max_seconds, reward_overrides=reward_overrides)
    episodes = []
    renderer = None
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    try:
        if capture:
            import pygame
            from space_z.rendering.renderer import Renderer
            pygame.display.init()
            pygame.font.init()
            renderer = Renderer(environment.unwrapped.game_config)
        for episode_index, seed in enumerate(seeds):
            observation, info = environment.reset(seed=seed)
            total_reward = 0.0
            components = {}
            reward_inputs = {}
            shots = 0
            screenshots = []
            telemetry = empty_telemetry()
            for step in range(round(max_seconds * 30)):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(action)
                collect_telemetry(telemetry, info, action)
                total_reward += reward
                shots += info['events']['player_shots_fired']
                for name, value in info['reward_components'].items():
                    components[name] = components.get(name, 0) + value
                for name, value in info['reward_inputs'].items():
                    reward_inputs[name] = reward_inputs.get(name, 0) + value
                if renderer is not None and episode_index == 0 and (step == 0 or step == 1799 or terminated or truncated):
                    renderer.draw(environment.unwrapped.game, show_controls=False)
                    filename = f'{timesteps:09d}_seed_{seed}_frame_{info["frames"]:06d}.png'
                    pygame.image.save(renderer.screen, directory / 'replays' / filename)
                    screenshots.append('replays/' + filename)
                if terminated or truncated:
                    break
            episodes.append(dict(seed=seed, score=info['score'], kills=info['kills'],
                                 survival_seconds=info['frames'] / 60, steps=step + 1, job=info['job'],
                                 episode_reward=total_reward, shots_fired=shots, reward_components=components,
                                 reward_inputs=reward_inputs,
                                 telemetry=telemetry,
                                 terminated=bool(terminated), truncated=bool(truncated), screenshots=screenshots))
    finally:
        environment.close()
        if renderer is not None:
            pygame.display.quit()
        model.policy.set_training_mode(was_training)
    summary = {}
    for metric in ['score', 'kills', 'survival_seconds', 'episode_reward', 'shots_fired']:
        values = [episode[metric] for episode in episodes]
        summary[metric] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                               p25=float(np.percentile(values, 25)), minimum=float(min(values)))
    return dict(summary=summary, episodes=episodes, deterministic=True, max_seconds=max_seconds)


def parent_metadata(checkpoint, reward_overrides=None):
    """동일 분기의 체크포인트와 고정 설정을 확인한다. reward_overrides가 있으면 보상 계수 필드만 예외로 허용한다."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or not checkpoint.is_relative_to(ROOT / 'experiments'):
        raise ValueError('Resume checkpoint must exist inside this branch experiments directory')
    with ZipFile(checkpoint) as archive:
        data = json.loads(archive.read('data'))
    if data.get('input_contract') != 'original_vector_5139_only':
        raise ValueError('Parent does not declare the vector-only input contract')
    config = json.loads((checkpoint.parent.parent / 'config.json').read_text(encoding='utf-8'))
    actual_environment = config['environment']
    expected_environment = json.loads(json.dumps(environment_configuration(reward_overrides)))
    if reward_overrides:
        actual_environment = {key: value for key, value in actual_environment.items() if key != 'reward_coefficients'}
        expected_environment = {key: value for key, value in expected_environment.items() if key != 'reward_coefficients'}
    if actual_environment != expected_environment or config['ppo'] != PPO_OPTIONS:
        raise ValueError('Parent environment or PPO settings differ from the continuation settings')
    return dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), sha256=sha256(checkpoint),
                timesteps=data['num_timesteps'], experiment_id=checkpoint.parent.parent.name,
                optimizer_preserved=True, environment_reset_on_resume=True)


def verify_loaded_parent(model, checkpoint, environment_count=4):
    """가중치와 Adam 이력이 체크포인트 그대로 복원됐는지 확인한다."""
    with ZipFile(checkpoint) as archive:
        expected_weights = torch.load(BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
        expected_optimizer = torch.load(BytesIO(archive.read('policy.optimizer.pth')), map_location='cpu', weights_only=True)
    actual_weights = model.policy.state_dict()
    assert expected_weights.keys() == actual_weights.keys()
    assert all(torch.equal(value, actual_weights[name]) for name, value in expected_weights.items())

    def equal(first, second):
        if isinstance(first, torch.Tensor):
            return isinstance(second, torch.Tensor) and torch.equal(first, second)
        if isinstance(first, dict):
            return first.keys() == second.keys() and all(equal(first[key], second[key]) for key in first)
        if isinstance(first, (list, tuple)):
            return len(first) == len(second) and all(equal(a, b) for a, b in zip(first, second))
        return first == second

    assert equal(expected_optimizer, model.policy.optimizer.state_dict())
    assert isinstance(model.policy, VectorOnlyPolicy) and model.n_envs == environment_count
    for name, expected in PPO_OPTIONS.items():
        actual = getattr(model, name)
        assert (actual(1) if callable(actual) else actual) == expected, name
    return dict(parent_weights=dict(passed=True, tensors=len(actual_weights)),
                parent_optimizer=dict(passed=True, parameter_states=len(expected_optimizer['state'])),
                parent_hyperparameters=dict(passed=True, settings=len(PPO_OPTIONS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='002_full_game_pilot')
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluation-interval', type=int, default=32768)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--max-owned-processes', type=int, default=18)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    parser.add_argument('--title')
    parser.add_argument('--goal')
    parser.add_argument('--evaluation-seeds', type=int, nargs='+', default=EVALUATION_SEEDS)
    parser.add_argument('--confirmation-seeds', type=int, nargs='+', default=[720101,720202,720303,720404,720505,720606])
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='부모 복원·검증 뒤 lr_schedule과 옵티마이저 lr만 명시적으로 덮어씀. 기본값 None은 override 없음.')
    parser.add_argument('--reward-overrides', type=str, nargs='+', default=None,
                        help='score_gain=0.03 enemy_damage=0.015 형태. 지정한 키만 REWARD_COEFFICIENTS 기본값 위에 덮어씀.')
    args = parser.parse_args()
    if not 2 <= args.max_owned_processes <= 18:
        parser.error('Use a process budget between 2 and 18')
    if not 1 <= args.num_envs <= 16 or (args.parallel and args.num_envs + 2 > args.max_owned_processes):
        parser.error('Environment workers plus trainer and launcher must fit the process budget')
    try:
        reward_overrides = parse_reward_overrides(args.reward_overrides)
    except ValueError as error:
        parser.error(str(error))
    rollout_size = args.num_envs * PPO_OPTIONS['n_steps']
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else rollout_size * (6 if args.parallel else 16)
    if checkpoint_interval <= 0 or checkpoint_interval % rollout_size:
        parser.error('Checkpoint interval must be a positive multiple of the rollout size')
    if args.steps <= 0 or args.steps % rollout_size:
        parser.error('Steps must be a positive multiple of the rollout size')
    if args.evaluation_interval <= 0 or args.evaluation_interval % checkpoint_interval:
        parser.error('Evaluation interval must be a positive multiple of the checkpoint interval')
    checks_file = ROOT / 'experiments/001_contract_checks/checks.json'
    if not checks_file.exists() or not all(result['passed'] for result in json.loads(checks_file.read_text(encoding='utf-8')).values()):
        parser.error('Run the recorded contract checks before training')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parent = parent_metadata(args.resume, reward_overrides) if args.resume else None
    initial_timesteps = parent['timesteps'] if parent else 0
    target_timesteps = initial_timesteps + args.steps
    config = dict(training_seed=args.seed, requested_steps=args.steps, number_of_environments=args.num_envs,
                  parallel=args.parallel, rollout_size=rollout_size, maximum_owned_processes=args.max_owned_processes,
                  initial_timesteps=initial_timesteps, target_timesteps=target_timesteps, parent=parent,
                  environment=environment_configuration(reward_overrides), ppo=PPO_OPTIONS,
                  learning_rate_override=args.learning_rate,
                  training_episode_limit_seconds=120,
                  evaluation=dict(seeds=args.evaluation_seeds, max_seconds=120, deterministic=True,
                                  role='development_only', interval_steps=args.evaluation_interval),
                  confirmation=dict(seeds=args.confirmation_seeds,
                                    max_seconds=120, role='held_out_until_model_selection') if parent else None,
                  checkpoint_interval_steps=checkpoint_interval,
                  model_selection='highest development mean game score, then mean survival; final is kept separately')
    title = '학습량 증가 · 같은 조건에서 이어 학습' if parent else '원본 벡터만으로 기본 게임 예비 학습'
    goal = (f'{parent["experiment_id"]}의 체크포인트에서 구조·보상·게임·PPO 설정을 유지하고 이어 학습한다. 환경 수 {args.num_envs}, 병렬 실행 {args.parallel}, rollout {rollout_size}스텝.'
            if parent else '목표 선택·조준 보조 입력 없이 기본 게임에서 점수 획득과 생존을 PPO로 학습할 수 있는지 살펴본다.')
    title = args.title or title
    goal = args.goal or goal
    directory = new_experiment(args.name, title, 'training', goal, config, parent=parent)
    started = time.monotonic()
    environments = None
    model = None
    recorder = None
    state = dict(status='prepared', timesteps=initial_timesteps, initial_timesteps=initial_timesteps,
                 additional_timesteps=0, requested_steps=args.steps, target_timesteps=target_timesteps,
                 updated_at=now(), process_id=os.getpid())
    best_rank = None
    try:
        factories = [lambda: Monitor(make_environment(120, reward_overrides=reward_overrides)) for _ in range(args.num_envs)]
        environments = SubprocVecEnv(factories, start_method='spawn') if args.parallel else DummyVecEnv(factories)
        process_usage = check_process_budget(args.max_owned_processes)
        event(directory, '프로세스 확인', '실측한 자체 Python 프로세스 수를 기록했다.',
              **process_usage)
        if parent:
            model = PPO.load(args.resume, env=environments, device='cpu', seed=args.seed)
            checks = verify_loaded_parent(model, args.resume, args.num_envs)
            write_json(directory / 'checks.json', checks)
            event(directory, 'parent_restored', '부모 정책·Adam·PPO 설정을 복원했다. 게임 에피소드는 새로 시작하고 환경 수 변경은 config에 기록했다.', checks=checks)
            if args.learning_rate is not None and args.learning_rate != PPO_OPTIONS['learning_rate']:
                previous_rate = PPO_OPTIONS['learning_rate']
                new_rate = args.learning_rate
                model.lr_schedule = lambda progress_remaining: new_rate
                for group in model.policy.optimizer.param_groups:
                    group['lr'] = args.learning_rate
                event(directory, 'learning_rate_overridden',
                      '부모 가중치·Adam 복원 검증을 마친 뒤 학습률만 명시적으로 바꿨다. 부모 체크포인트·config는 바꾸지 않는다.',
                      previous=previous_rate, new=args.learning_rate)
        else:
            model = PPO(VectorOnlyPolicy, environments, seed=args.seed, device='cpu', verbose=1, **PPO_OPTIONS)
        model.training_method = 'pure_ppo'
        model.decision_frame_skip = 2
        model.input_contract = 'original_vector_5139_only'
        model.set_logger(configure(str(directory / 'logs/tensorboard'), ['stdout', 'tensorboard']))
        model.logger.output_formats.append(TrainingJsonWriter(directory))
        model.logger.output_formats.append(HumanOutputFormat(str(directory / 'logs/console.log')))
        recorder = EpisodeRecorder(directory, args.num_envs, started, args.max_owned_processes)
        event(directory, 'started', '기본 게임의 기존 관찰만 사용하며 보조 목표 선택·시야선 보상은 사용하지 않는다.',
              policy_parameters=sum(parameter.numel() for parameter in model.policy.parameters()),
              process_id=os.getpid(), environment_processes=args.num_envs if args.parallel else 0,
              environment_count=args.num_envs, rollout_size=rollout_size)

        def assess(label, capture=False):
            nonlocal best_rank
            checkpoint = directory / 'checkpoints' / f'step_{model.num_timesteps:09d}.zip'
            save_model(model, checkpoint)
            state.update(status='evaluating', timesteps=model.num_timesteps,
                         additional_timesteps=model.num_timesteps-initial_timesteps,
                         latest_checkpoint=checkpoint.relative_to(directory).as_posix(), updated_at=now())
            write_json(directory / 'state.json', state)
            event(directory, 'evaluation_started', label, timesteps=model.num_timesteps)
            update_report()
            result = evaluate(model, directory, model.num_timesteps, seeds=args.evaluation_seeds, capture=capture,
                              reward_overrides=reward_overrides)
            record = dict(time=now(), label=label, timesteps=model.num_timesteps,
                          checkpoint=checkpoint.relative_to(directory).as_posix(), checkpoint_sha256=sha256(checkpoint), **result)
            append_json(directory / 'logs/evaluations.jsonl', record)
            for metric, summary in result['summary'].items():
                model.logger.record('evaluation/' + metric, summary['mean'])
            for name in environment_configuration(reward_overrides)['reward_coefficients']:
                mean = np.mean([episode['reward_components'][name] for episode in result['episodes']])
                model.logger.record('evaluation_reward_components/' + name, float(mean))
            details = [episode_metrics(episode) for episode in result['episodes']]
            for name in details[0]:
                model.logger.record('evaluation_details/' + name, float(np.mean([row[name] for row in details])))
            model.logger.dump(model.num_timesteps)
            rank = (result['summary']['score']['mean'], result['summary']['survival_seconds']['mean'])
            if best_rank is None or rank > best_rank:
                best_rank = rank
                shutil.copyfile(checkpoint, directory / 'checkpoints/best.zip')
                state.update(best_timesteps=model.num_timesteps, best_checkpoint='checkpoints/best.zip', best_score=rank[0])
            write_json(directory / 'state.json', state)
            event(directory, 'evaluation_finished', label, timesteps=model.num_timesteps, summary=result['summary'])
            print('EVALUATION', model.num_timesteps, json.dumps(result['summary']), flush=True)
            update_report()

        assess('before', capture=True)
        while model.num_timesteps < target_timesteps:
            state.update(status='training', updated_at=now())
            write_json(directory / 'state.json', state)
            event(directory, 'training_block', '같은 설정으로 이어 학습한다.', start_step=model.num_timesteps)
            model.learn(min(checkpoint_interval, target_timesteps - model.num_timesteps), callback=recorder,
                        reset_num_timesteps=False, progress_bar=False)
            save_model(model, directory / 'checkpoints/latest.zip')
            state.update(timesteps=model.num_timesteps, additional_timesteps=model.num_timesteps-initial_timesteps,
                         latest_checkpoint='checkpoints/latest.zip', updated_at=now())
            write_json(directory / 'state.json', state)
            if (directory / 'stop.request').exists():
                state['status'] = 'stopped'
                event(directory, 'stopped', (directory / 'stop.request').read_text(encoding='utf-8-sig'))
                break
            if (model.num_timesteps-initial_timesteps) % args.evaluation_interval == 0 or model.num_timesteps >= target_timesteps:
                assess('after' if model.num_timesteps >= target_timesteps else 'progress', capture=model.num_timesteps >= target_timesteps)
            update_report()
        save_model(model, directory / 'checkpoints/final.zip')
        if state['status'] != 'stopped':
            state['status'] = 'completed'
        state['final_checkpoint'] = 'checkpoints/final.zip'
        if parent and state['status'] == 'completed':
            state.update(status='confirming', updated_at=now())
            write_json(directory / 'state.json', state)
            event(directory, 'confirmation_started', '개발 평가로 후보를 고정한 뒤 config에 사전 선언한 확인 시드에서 부모와 비교한다.', selected_timesteps=state['best_timesteps'])
            update_report()
            confirmation = dict(seeds=config['confirmation']['seeds'], max_seconds=120,
                                selected_timesteps=state['best_timesteps'],
                                parent_checkpoint=parent, selected_checkpoint='checkpoints/best.zip',
                                selected_sha256=sha256(directory / 'checkpoints/best.zip'))
            for label, path in [('parent', args.resume), ('selected', directory / 'checkpoints/best.zip')]:
                candidate = PPO.load(path, device='cpu')
                confirmation[label] = evaluate(candidate, directory, candidate.num_timesteps,
                                               seeds=config['confirmation']['seeds'], max_seconds=120,
                                               reward_overrides=reward_overrides)
                with SummaryWriter(str(directory / 'logs/tensorboard/confirmation' / label)) as writer:
                    details = [episode_metrics(episode) for episode in confirmation[label]['episodes']]
                    for index, values in enumerate(details):
                        for name, value in values.items():
                            writer.add_scalar('episodes/' + name, value, index + 1)
                    for name in details[0]:
                        writer.add_scalar('mean/' + name, float(np.mean([row[name] for row in details])), candidate.num_timesteps)
                del candidate
            write_json(directory / 'confirmation.json', confirmation)
            event(directory, 'confirmation_finished', '새 시드 비교를 기록했다. 이 결과를 보고 후보를 다시 고르거나 추가 학습하지 않는다.',
                  parent_summary=confirmation['parent']['summary'], selected_summary=confirmation['selected']['summary'])
            state['status'] = 'completed'
        event(directory, state['status'], '계획한 학습·평가를 종료했다. 평가 시드와 상한은 config에 기록했다.', timesteps=model.num_timesteps)
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        if model is not None:
            save_model(model, directory / 'checkpoints/interrupted.zip')
        event(directory, 'interrupted', '사용자 중단. 메모리에 남은 모델은 interrupted.zip으로 저장했다.')
    except Exception as error:
        state.update(status='failed', error=repr(error))
        event(directory, 'failed', repr(error))
        raise
    finally:
        if environments is not None:
            environments.close()
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        if recorder is not None:
            recorder.episode_writer.close()
        state.update(timesteps=model.num_timesteps if model is not None else 0,
                     additional_timesteps=(model.num_timesteps-initial_timesteps) if model is not None else 0,
                     updated_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(directory / 'state.json', state)
        update_report()


if __name__ == '__main__':
    main()
