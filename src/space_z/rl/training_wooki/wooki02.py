"""Fresh PPO growth training without scripted combat, with hurt/stall penalties.

Run: python -m space_z.rl.training_wooki.wooki02
Resume this variant only: append --resume. Stop safely via control.json command=stop.
Reward: score delta / 200 - death * .5 - actual damage * .01 - wall stall * .01.
"""

# 별도 학습 변형: wooki01의 기록 기능을 재사용하고, 피격·벽 정체 벌점을 추가한다.

from pathlib import Path
import os
import sys

# 파일 직접 실행 시에도 저장 모델의 클래스가 정식 패키지 경로로 import되도록 한다.
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
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from .environment import TrainingEnv
from .growth import STAT_COUNT, legal_growth_masks
from .policy import SpaceFeatures
from .runtime import (
    atomic_json,
    read_json,
    publish_checkpoint,
    checkpoint_path,
    learning_rate,
)
from .settings import TrainSettings
from .wooki01 import LiveCallback

# 벌점 계수와 실험 구분값을 저장해 재개 시 같은 규칙인지 확인한다.
RULES = {
    'variant': 'wooki02',
    'damage_per_hp': .01,
    'stall_per_step': .01,
    'stall_window_steps': 12,
    'stall_span_px': 18.,
    'wall_clearance_px': 8.,
    'scripted_combat': False,
    'pretrained_combat': False,
}


def penalty_components(damage_taken, stalled):
    # 실제 받은 피해와 벽 정체 여부를 각각 음수 보상 항목으로 반환한다.
    return {
        'hurt': -RULES['damage_per_hp'] * max(0., float(damage_taken)),
        'wall_stall': -RULES['stall_per_step'] if stalled else 0.,
    }


class WallStallTracker:
    """Measure 12 uninterrupted movement attempts; never choose an escape action."""

    def __init__(self, position):
        # 12번 연속 이동 시도의 변위를 보기 위해 시작점을 포함한 위치 13개를 보관한다.
        self.positions = deque([np.asarray(position, dtype=float)], maxlen=13)
        self.attempts = 0
        self.span = 0.

    def update(self, position, attempted, near_wall):
        # 벽 근처에서 이동을 연속 시도했는데 위치 범위가 작으면 정체로 판정한다.
        position = np.asarray(position, dtype=float)

        # 연속 조건이 깨지면 위치 이력과 시도 횟수를 새로 시작한다.
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
        # 연속 시도 횟수와 위치 범위를 각각 0~1로 제한한 2차원 관찰을 반환한다.
        return np.array(
            [min(self.attempts / 12., 1.), min(self.span / 18., 1.)],
            dtype=np.float32,
        )


class PenaltyTrainingEnv(TrainingEnv):
    def __init__(self, **kwargs):
        # 성장·점수 목표 환경에 정체 관찰만 추가하며, 강제 탈출 기능은 사용하지 않는다.
        super().__init__(
            score_objective=True, full_growth=True, corner_recovery=False, **kwargs
        )
        self.observation_space.spaces['stall'] = spaces.Box(0., 1., (2,), np.float32)

    def reset(self, **kwargs):
        # 기본 게임 초기화 후 시작 위치로 정체 추적기와 누적 정체 횟수를 초기화한다.
        obs, info = super().reset(**kwargs)
        p = self.env.game.player
        self.stall_tracker = WallStallTracker((p.x, p.y))
        self.stall_steps = 0
        obs['stall'] = self.stall_tracker.observation()
        return obs, info

    def step(self, action):
        # 기본 진행 결과에 피격·정체 벌점을 더하고, 관찰과 에피소드 지표에도 반영한다.
        obs, reward, terminated, truncated, info = super().step(action)
        p = self.env.game.player
        # 맵 경계와 파괴 가능한 벽을 포함해 기체 주변 여유 공간만 조회한다.
        near_wall = not self.env.game.terrain.circle_fits(
            p.x, p.y, p.radius + RULES['wall_clearance_px']
        )
        stalled = self.stall_tracker.update(
            (p.x, p.y), int(action[0]) // 6 != 0, near_wall
        )
        self.stall_steps += int(stalled)
        parts = penalty_components(info['events']['damage_taken'], stalled)

        # 1) 추가 벌점을 반환 보상과 에피소드 누적 보상 양쪽에 반영한다.
        extra = sum(parts.values())
        reward += extra
        self.episode_return += extra
        self.task_return += extra
        info['reward_components'].update(parts)
        info['task_reward'] += extra
        for key, value in parts.items():
            self._components[key] = self._components.get(key, 0.) + value

        # 2) 정체 상태를 관찰에 넣고, 종료 지표가 있으면 추가 벌점까지 포함해 갱신한다.
        obs['stall'] = self.stall_tracker.observation()
        info.update(wall_stalled=bool(stalled), wall_near=bool(near_wall))
        if terminated or truncated:
            info['episode_metrics'].update(
                **{
                    'return': self.episode_return,
                    'task_return': self.task_return,
                    'components': dict(self._components),
                    'wall_stall_steps': self.stall_steps,
                }
            )
        return obs, float(reward), terminated, truncated, info


class LearnedFeatures(SpaceFeatures):
    def __init__(self, observation_space):
        # 기본 벡터·격자 특징 크기에 정체 관찰 크기를 더한다.
        super().__init__(observation_space)
        self._features_dim += int(np.prod(observation_space['stall'].shape))

    def forward(self, observations):
        # 기본 신경망 특징 뒤에 정체 관찰을 그대로 이어 붙인다.
        return torch.cat((super().forward(observations), observations['stall']), dim=1)


class LearnedGrowthPolicy(MultiInputActorCriticPolicy):
    """All 54 combat choices are learned, with no tactical masks or frozen prior."""

    def _distribution_and_value(self, obs):
        # 신경망 출력을 전투·스탯·전직으로 나누고, 성장 선택만 가능 여부로 제한한다.
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        output = self.action_net(pi)

        # 54개 전투 선택 뒤에 스탯·전직 선택 점수가 이어진다.
        combat = output[:, :54]
        stat = output[:, 54 : 54 + STAT_COUNT]
        evolution = output[:, 54 + STAT_COUNT :]
        stat_mask, evolution_mask = legal_growth_masks(obs['vector'])
        # 기존 성장 선택 선호는 유지하며, 전투 행동에는 사전 정책의 도움을 적용하지 않는다.
        wait_bias = torch.zeros_like(stat)
        wait_bias[:, 0] = stat_mask[:, 1:].any(1) * 2.
        stat = (stat - wait_bias).masked_fill(~stat_mask, -1e4)
        evolution = evolution.masked_fill(~evolution_mask, -1e4)
        distribution = self.action_dist.proba_distribution(
            action_logits=torch.cat((combat, stat, evolution), dim=1)
        )
        return distribution, self.value_net(vf)

    def get_distribution(self, obs):
        # 계산한 전투·성장 행동 분포를 반환한다.
        return self._distribution_and_value(obs)[0]

    def forward(self, obs, deterministic=False):
        # 분포에서 행동을 선택하고 상태 가치·로그 확률과 함께 반환한다.
        distribution, values = self._distribution_and_value(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        # PPO 갱신에 필요한 주어진 행동의 로그 확률·엔트로피·가치를 반환한다.
        distribution, values = self._distribution_and_value(obs)
        return values, distribution.log_prob(actions), distribution.entropy()


def one_env(cfg, evaluation=False):
    # 학습·평가 길이에 맞는 벌점 환경을 만들고 에피소드 기록용 Monitor로 감싼다.
    return Monitor(
        PenaltyTrainingEnv(
            max_steps=cfg.eval_steps if evaluation else cfg.max_steps,
            curriculum=cfg.curriculum,
            evaluation=evaluation,
            gamma=cfg.gamma,
        )
    )


def raw_env(cfg, evaluation=False, subprocess=True):
    # 평가는 하나, 학습은 workers개의 환경을 만들고 해당 seed를 지정한다.
    count = 1 if evaluation else cfg.workers
    factories = [partial(one_env, cfg, evaluation) for _ in range(count)]
    env = (
        SubprocVecEnv(factories, start_method='spawn')
        if subprocess and count > 1
        else DummyVecEnv(factories)
    )
    env.seed(cfg.evaluation_seed if evaluation else cfg.seed)
    return env


def stacked(env, cfg):
    # 격자·벡터·정체 관찰을 각 축에 history개씩 쌓는다.
    return VecFrameStack(
        env,
        n_stack=cfg.history,
        channels_order={'grid': 'first', 'vector': 'last', 'stall': 'last'},
    )


def make_env(cfg, evaluation=False, subprocess=True):
    # 이력 관찰을 구성하고, 관찰·보상 정규화를 끈 VecNormalize로 감싼다.
    env = VecNormalize(
        stacked(raw_env(cfg, evaluation, subprocess), cfg),
        norm_obs=False,
        norm_reward=False,
    )
    env.training = not evaluation
    return env


def make_model(cfg, env):
    # 학습된 전투 사전 정책 없이 전투와 성장을 함께 학습하는 PPO를 만든다.
    return PPO(
        LearnedGrowthPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=LearnedFeatures,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            normalize_images=False,
        ),
        learning_rate=partial(learning_rate, initial=cfg.learning_rate),
        n_steps=cfg.rollout_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.epochs,
        gamma=cfg.gamma,
        gae_lambda=.95,
        clip_range=.15,
        ent_coef=.02,
        vf_coef=.5,
        max_grad_norm=.5,
        target_kl=.025,
        seed=cfg.seed,
        device='cpu',
    )


def load_model(directory, cfg, subprocess=True):
    # 정규화 래퍼와 PPO를 복원하고, wooki02용 정책인지 확인한다.

    # 복원할 세대와 정책 종류를 검사해 다른 학습 변형의 모델이 섞이지 않게 한다.
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
    # 별도 환경과 고정 seed에서 점수 기준 평가를 수행하고 평균·개별 기록을 반환한다.
    env = make_env(cfg, evaluation=True, subprocess=False)
    episodes = []
    rng = np.random.default_rng(918273)
    try:
        # 같은 평가 seed 집합으로 무작위 기준선과 학습 정책의 게임 점수를 비교한다.
        for i in range(cfg.eval_episodes):
            env.seed(cfg.evaluation_seed + i)
            obs = env.reset()
            while True:
                if heartbeat is not None and not heartbeat():
                    return None
                action = (
                    np.array([[rng.integers(int(n)) for n in env.action_space.nvec]])
                    if random_policy
                    else model.predict(obs, deterministic=True)[0]
                )
                obs, _, done, infos = env.step(action)
                if done[0]:
                    episodes.append(infos[0]['episode_metrics'])
                    break
        scores = [e['score'] for e in episodes]
        return dict(
            step=model.num_timesteps,
            kind='random' if random_policy else 'model',
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            score=float(np.mean(scores)),
            kills=float(np.mean([e['kills'] for e in episodes])),
            seconds=float(np.mean([e['seconds'] for e in episodes])),
            horizon=cfg.eval_steps,
            metric='score',
            episodes=episodes,
            reward_mean=float(np.mean([e['return'] for e in episodes])),
        )
    finally:
        env.close()


class PenaltyCallback(LiveCallback):
    def assess(self):
        # wooki02 평가 결과로 최고 점수·체크포인트·평가 CSV를 갱신한다.
        self.state('evaluating', True)
        result = evaluate(
            self.model, self.cfg, heartbeat=lambda: self.control('evaluating')
        )
        if result is not None:
            self.evaluations.append(result)
            better = result['mean'] > self.best
            if better:
                self.best = result['mean']
            publish_checkpoint(
                self.model,
                self.training_env,
                self.directory,
                self.cfg,
                best=better,
                evaluation=result,
            )
            self.append_csv(
                'evaluations.csv',
                {
                    k: result[k]
                    for k in (
                        'step',
                        'kind',
                        'mean',
                        'std',
                        'kills',
                        'seconds',
                        'horizon',
                    )
                },
            )
            self.last_eval = self.model.num_timesteps
        self.state('training', True)


def run_training(directory, cfg, resume=False, subprocess=True):
    # 실험 규칙 확인 → 모델 준비 → 기준선 평가 → 학습 → 최종 저장 순서로 실행한다.
    cfg.validate()
    if (
        cfg.algorithm != 'ppo'
        or not cfg.full_growth
        or cfg.corner_recovery
        or cfg.objective != 'score100'
    ):
        raise ValueError(
            'wooki02 requires growth PPO with score rewards and no combat recovery'
        )
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # 1) 실행 폴더를 잠그고 신규 폴더 또는 재개의 규칙 일치 여부를 확인한다.
    with FileLock(str(directory / 'job.lock'), timeout=0):
        marker = directory / 'wooki02.json'
        if resume:
            if read_json(marker) != RULES:
                raise ValueError(
                    'Resume requires a wooki02 run with matching reward rules'
                )
        elif any(directory.iterdir()) and any(
            p.name != 'job.lock' for p in directory.iterdir()
        ):
            raise ValueError(
                'New training requires an empty run folder; use --resume for wooki02'
            )
        atomic_json(marker, RULES)
        torch.set_num_threads(cfg.torch_threads)

        # 2) 모델·환경을 복원하거나 새 환경을 만들고 기록 콜백을 연결한다.
        model, env = (
            load_model(directory, cfg, subprocess)
            if resume
            else (None, make_env(cfg, subprocess=subprocess))
        )
        callback = None
        try:
            if model is None:
                model = make_model(cfg, env)
            previous = read_json(directory / 'status.json', {}) if resume else {}
            callback = PenaltyCallback(directory, cfg, previous)
            model.set_logger(
                __import__(
                    'stable_baselines3.common.logger', fromlist=['configure']
                ).configure(None, [])
            )
            callback.init_callback(model)
            callback.start_step = model.num_timesteps
            atomic_json(directory / 'config.json', cfg.to_dict())
            atomic_json(directory / 'control.json', {'command': 'run'})
            publish_checkpoint(model, env, directory, cfg)
            callback.state('baseline', True)

            # 3) 기준선이 없는 경우에만 무작위 행동 평가를 수행한다.
            if callback.baseline is None:
                callback.baseline = evaluate(
                    model,
                    cfg,
                    random_policy=True,
                    heartbeat=lambda: callback.control('baseline'),
                )
                if callback.baseline is not None:
                    atomic_json(directory / 'random_baseline.json', callback.baseline)

            # 4) 남은 step만 학습하고 평가·체크포인트·종료 상태를 저장한다.
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
                atomic_json(
                    directory / 'status.json', {'status': 'stopped', 'pid': os.getpid()}
                )
        except Exception as error:
            atomic_json(
                directory / 'status.json',
                dict(
                    status='failed',
                    error=str(error),
                    traceback=traceback.format_exc(),
                    pid=os.getpid(),
                ),
            )
            raise
        finally:
            env.close()


def main():
    # 신규·재개 설정을 고르고, 선택적으로 짧은 검증 실행 설정을 적용해 학습을 시작한다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=Path(__file__).resolve().parents[4] / 'runs/ppo-wooki02',
    )
    parser.add_argument('--steps', type=int, default=2_000_000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    # 재개 시에는 저장 설정을 복원하고, 목표 누적 step만 명령행 값으로 바꾼다.
    if args.resume:
        saved = read_json(args.run_dir / 'config.json')
        if not saved:
            parser.error('No saved wooki02 configuration')
        cfg = TrainSettings(**saved)
        cfg.total_steps = args.steps
    else:
        cfg = TrainSettings(
            total_steps=args.steps,
            workers=args.workers,
            full_growth=True,
            corner_recovery=False,
            objective='score100',
            curriculum=False,
            max_steps=9000,
            eval_steps=1500,
            eval_episodes=6,
            eval_every=16384,
            gamma=.9995,
            torch_threads=1,
        )

    # 짧은 검증 실행에서는 환경 수·수집량·에피소드 길이를 줄인다.
    if args.smoke:
        cfg.total_steps = 64
        cfg.workers, cfg.rollout_steps, cfg.batch_size, cfg.epochs = 1, 16, 16, 1
        cfg.max_steps, cfg.eval_steps, cfg.eval_episodes = 16, 8, 1
    run_training(args.run_dir, cfg.validate(), resume=args.resume)


if __name__ == '__main__':
    # 파일 직접 실행·-m 실행 모두 저장 모델의 클래스 경로를 import 가능한 형태로 유지한다.
    from space_z.rl.training_wooki.wooki02 import main as entrypoint

    entrypoint()
