"""Reproducible experiment settings, separate from the playable game's defaults."""
from dataclasses import asdict, dataclass


@dataclass
class TrainSettings:
    algorithm: str = 'ppo'
    total_steps: int = 2_000_000
    workers: int = 4
    seed: int = 42
    evaluation_seed: int = 900001
    objective: str = 'combat'
    corner_recovery: bool = False
    history: int = 4
    max_steps: int = 3000
    eval_steps: int = 1500
    eval_episodes: int = 3
    eval_every: int = 50000
    checkpoint_every: int = 8192
    rollout_steps: int = 512
    batch_size: int = 256
    epochs: int = 6
    curriculum: bool = True
    torch_threads: int = 2
    gamma: float = .995
    learning_rate: float = .0003

    def validate(self):
        if self.algorithm not in ('ppo', 'dqn'):
            raise ValueError('algorithm must be ppo or dqn')
        if self.objective not in ('combat', 'score100'):
            raise ValueError('objective must be combat or score100')
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError('evaluation_seed must be a nonnegative integer')
        for key in ('total_steps', 'workers', 'history', 'max_steps', 'eval_steps',
                    'eval_episodes', 'eval_every', 'checkpoint_every', 'rollout_steps',
                    'batch_size', 'epochs', 'torch_threads'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.algorithm == 'ppo' and self.rollout_steps * self.workers < 2:
            raise ValueError('PPO needs at least two rollout samples')
        if not 0 < self.gamma < 1 or self.learning_rate <= 0:
            raise ValueError('invalid discount factor or learning rate')
        return self

    def to_dict(self):
        return asdict(self)


STAGE_NAMES = ('기초 전투 · 적 3대', '장애물 적응 · 적 6대', '원래 게임 · 적 9대')


def stage_for_step(step, curriculum=True):
    return (0 if step < 100000 else 1 if step < 350000 else 2) if curriculum else 2
