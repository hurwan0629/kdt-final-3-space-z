"""Reproducible experiment settings, separate from the playable game's defaults."""

# 실험 설정과 난이도 단계 선택을 담당한다. 게임 자체의 기본 설정과는 별개다.

from dataclasses import asdict, dataclass


@dataclass
class TrainSettings:
    # 실행 대상과 난수: total_steps는 재개 이후에도 목표 누적 step이다.
    algorithm: str = 'ppo'  # 사용할 학습 알고리즘: ppo 또는 dqn
    total_steps: int = 2_000_000  # 모든 환경을 합산한 목표 누적 step
    workers: int = 4  # 동시에 실행할 학습 환경 수
    seed: int = 42  # 학습 모델·환경의 난수 seed
    evaluation_seed: int = 900001  # 평가 에피소드 seed 집합의 시작값
    objective: str = 'combat'  # 기본 전투 보상 또는 게임 점수 목표(score100)
    corner_recovery: bool = False  # 모서리 탈출 상태·정책 사용 여부
    full_growth: bool = False  # 스탯 강화·전직·숙련을 포함한 성장 학습 여부

    # 관찰 이력과 에피소드 길이: 한 환경 step은 기본 게임 4프레임을 진행한다.
    history: int = 4  # 함께 입력할 연속 관찰 수
    max_steps: int = 3000  # 학습 에피소드의 최대 환경 step
    eval_steps: int = 1500  # 평가 에피소드의 최대 환경 step
    eval_episodes: int = 3  # 평가 한 번에 실행할 에피소드 수
    eval_every: int = 50000  # 평가 간격의 누적 step 기준값
    checkpoint_every: int = 8192  # 체크포인트 간격의 누적 step 기준값

    # PPO 갱신과 실행 자원: 저장·평가 시점은 데이터 수집 주기 경계에서 확인한다.
    rollout_steps: int = 512  # PPO가 갱신 전 환경 하나에서 수집할 step 수
    batch_size: int = 256  # 한 번의 경사 계산에 사용할 표본 수
    epochs: int = 6  # 수집한 표본을 PPO 갱신에 반복 사용할 횟수
    curriculum: bool = True  # 누적 step에 따른 단계별 난이도 사용 여부
    torch_threads: int = 2  # 현재 프로세스에서 PyTorch가 사용할 스레드 수
    gamma: float = .995  # 미래 보상의 할인율
    learning_rate: float = .0003  # PPO 학습률 스케줄의 시작값

    def validate(self):
        # 선택 가능한 모드와 수치 범위를 검사하며, 유효하면 현재 설정 객체를 반환한다.
        if self.algorithm not in ('ppo', 'dqn'):
            raise ValueError('algorithm must be ppo or dqn')
        if self.full_growth and self.algorithm != 'ppo':
            raise ValueError('Full growth uses factorized PPO actions')
        if self.objective not in ('combat', 'score100'):
            raise ValueError('objective must be combat or score100')
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError('evaluation_seed must be a nonnegative integer')
        for key in (
            'total_steps',
            'workers',
            'history',
            'max_steps',
            'eval_steps',
            'eval_episodes',
            'eval_every',
            'checkpoint_every',
            'rollout_steps',
            'batch_size',
            'epochs',
            'torch_threads',
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.algorithm == 'ppo' and self.rollout_steps * self.workers < 2:
            raise ValueError('PPO needs at least two rollout samples')
        if not 0 < self.gamma < 1 or self.learning_rate <= 0:
            raise ValueError('invalid discount factor or learning rate')
        return self

    def to_dict(self):
        # 설정 필드를 JSON 저장에 사용할 딕셔너리로 변환한다.
        return asdict(self)


STAGE_NAMES = ('기초 전투 · 적 3대', '장애물 적응 · 적 6대', '원래 게임 · 적 9대')


def stage_for_step(step, curriculum=True):
    # 누적 학습 step으로 0~2단계를 고른다. 커리큘럼을 끄면 항상 마지막 단계다.
    return (0 if step < 100000 else 1 if step < 350000 else 2) if curriculum else 2
