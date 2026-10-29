"""DQN_03 학습 실행: python DQN_03.py   (프로젝트 루트에서 실행, 아래 '실행' 참고)

목표: 오래 살아남으면서 점수도 올린다. 두 가지가 모두 올라야 좋은 모델로 인정한다.

설계 원칙
  1) 사람이 짠 전술·성장 규칙을 넣지 않는다. 이동·조준·발사, 스탯 강화, 전직 선택을 모델이 직접 배운다.
  2) 적 AI의 설정값(감지 거리·유지 거리·발사 간격 등)을 보상·관찰·규칙에 쓰지 않는다.
     게임이 알려 주는 결과(피해·점수·사망)로만 학습한다.
  3) 평가는 보상과 분리한다. 게임 원래 값(생존 시간·점수·처치·레벨)만 쓰고,
     학습에 쓰지 않은 시드에서 무작위 정책 기준선과 같은 조건으로 비교한다.
     최고 모델은 학습 중 평가 시드로 고르고, 최종 검증(--play)은 선택에 쓰지 않은 별도 시드로 한다.

알고리즘: Double + Dueling + N-step + 우선순위 리플레이(PER)를 쓰는 분기(branching) DQN.
  - 출력 3갈래: 이동·조준·발사(54) / 스탯 강화(6) / 전직(5). 1620개를 한 번에 고르지 않는다.
  - 관찰 구조를 이용하는 네트워크: 지형은 CNN, 적·천체·탄환은 슬롯 공유 인코더.
  - 환경 여러 개를 프로세스로 병렬 실행하고, 리플레이는 GPU 메모리에 둔다.

실행
  학습:  python src/space_z/rl/heeseung/DQN03/DQN_03.py --save-dir <저장 폴더>
  평가:  python src/space_z/rl/heeseung/DQN03/DQN_03.py --play <dqn03_best.pt> --eval-episodes 10 --baseline
추가 의존성: torch. 프로젝트 설치는 python -m pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from space_z.config import GameConfig, RewardConfig
from space_z.game.actions import N_MOVES, N_ROTS
from space_z.rl.actions.discrete_action_decoder import DiscreteActionDecoder
from space_z.rl.environment import TankEnv
from space_z.rl.observations.vector_observation import VectorObservation

# --------------------------------------------------------------------------
# 하이퍼파라미터 (여기서 수정한다. 실행할 때 --옵션을 주면 이 값보다 우선한다)
# --------------------------------------------------------------------------
# 학습 길이: TRAIN_EPISODES가 0보다 크면 전체 환경이 끝낸 판 수의 합에서 끝나고, 0이면 TRAIN_STEPS에서 끝난다.
TRAIN_EPISODES = 0  # 학습할 에피소드 수(모든 환경의 합). 0이면 아래 step 기준
TRAIN_STEPS = 300000  # 학습할 환경 step 수(모든 환경의 합, TRAIN_EPISODES가 0일 때만 사용)
SEED = 42  # 학습 난수 시드

NUM_ENVS = 0  # 병렬 환경(프로세스) 수. 0이면 CPU 코어 수에 맞춰 자동(최대 8). 코랩 기본은 코어 2개

LEARNING_RATE = 1e-4  # Adam 학습률
ADAM_EPS = 1e-8  # Adam 안정화 값(PyTorch 기본). 보상이 작을 때 이 값이 크면 학습이 사실상 멈춘다
GAMMA = 0.99  # 할인율
N_STEP = 3  # N-step 반환: 몇 step 뒤까지의 보상을 합쳐서 학습할지 (1이면 1-step)
BATCH_SIZE = 256  # 한 번의 경사 계산에 쓰는 경험 수
BUFFER_SIZE = 100000  # 리플레이 버퍼 크기. GPU 메모리 약 2GB(10만 개 기준)
LEARN_START = 5000  # 버퍼에 이 개수 이상 쌓이면 학습 시작
TRAIN_EVERY = 8  # 모든 환경 step이 이만큼 모일 때마다 학습 1회
TAU = 0.005  # 타깃 신경망이 온라인을 따라가는 비율(학습 1회마다 적용)
DOUBLE_DQN = True  # Double DQN 사용 여부
REWARD_SCALE = 1.0  # 학습용 보상 배율(1.0=원본). 보상이 이미 작아서 줄이지 않는다

PER_ALPHA = 0.6  # 우선순위 반영 정도(0이면 균등 샘플링)
PER_BETA_START = 0.4  # 중요도 보정 시작 값
PER_BETA_STEPS = 500000  # 중요도 보정이 1.0이 되는 step 수

EPS_END = 0.05  # ε 최솟값 (탐험 확률의 하한)
EPS_DECAY_STEPS = 150000  # ε이 1.0에서 EPS_END까지 줄어드는 step 수(모든 환경의 합)

LOG_EVERY = 10  # 몇 에피소드마다 학습 상황을 출력할지
EVAL_EVERY_STEPS = 50000  # 몇 step마다 평가할지 (0이면 평가 끔)
EVAL_EPISODES = 5  # 평가할 때 돌리는 에피소드 수
EVAL_SEED_BASE = 90000000  # 학습 중 평가(최고 모델 선택)에 쓰는 시드 시작값. 학습 시드와 겹치지 않는 큰 값
TEST_SEED_BASE = 95000000  # --play 최종 검증용 시드. 모델 선택에 쓰지 않은 시드라야 선택 편향 없이 실력을 잰다
SAVE_EVERY_STEPS = 50000  # 몇 step마다 dqn03_last.pt를 저장할지
SAVE_DIR = "trained_models"  # 체크포인트·로그 저장 폴더 (코랩은 Drive 경로 권장)

# --------------------------------------------------------------------------
# 보상 (승인된 표). 변수 이름은 공통 RewardConfig의 필드와 같고, score_gain만 추가했다.
# 환경이 계산하는 기본 보상은 쓰지 않고 이 값들로 이 파일에서 직접 계산한다.
# --------------------------------------------------------------------------
score_gain = 0.01  # 오른 점수 1점당 보상 (적 처치 +2.0, 엘리트 +4.0, 사각형 +0.1, 오각형 +1.3)
shape_damage = 0.001  # 천체 피해 중간 신호: 가중 피해(점수 환산)에 곱함. 파괴 점수 보상의 약 10%
shape_destroy = 0.0  # 천체 파괴 보상. score_gain과 중복되므로 0
enemy_damage = 0.003  # 적 피해 중간 신호: 적에게 준 피해량에 곱함 (체력 60 적 1대분 +0.18)
enemy_kill = 0.0  # 적 처치 보상. score_gain과 중복되므로 0
damage_taken = -0.05  # 피격 패널티: 받은 피해량에 곱함 (체력 100 전부 잃으면 -5)
death = -10.0  # 플레이어 사망 시 1회 (시간 제한 종료에는 적용하지 않음)
time_step = 0.005  # 생존 보상: step마다 더함 (1분 생존 +4.5)
shot_cost = 0.0  # 발사 비용: 두지 않음

# 화면 표시(--play 평가에서만 사용). 학습은 병렬 프로세스라 화면 없이 진행한다.
RENDER_MODE = None  # 화면 없이 평가
# RENDER_MODE = "human"  # --play로 저장한 모델의 플레이를 화면으로 볼 때 켠다 (위의 RENDER_MODE = None 줄은 지운다)

# 배속(RENDER_MODE="human"일 때만 의미가 있다). 1.0=실시간(15 step/초), 4.0=4배속, 0=제한 없음.
RENDER_SPEED = 4.0


# --------------------------------------------------------------------------
# 장치·난수
# --------------------------------------------------------------------------
def pick_device() -> torch.device:
    """사용할 수 있는 가장 빠른 장치(cuda → mps → cpu)를 반환한다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def set_seed(seed: int) -> None:
    """NumPy·torch 난수를 고정한다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 환경·행동·보상
# --------------------------------------------------------------------------
N_CONTROL = N_MOVES * N_ROTS * 2  # 이동 × 조준 회전 × 발사 = 54
N_STAT = 6  # 스탯 강화: 0=선택 안 함, 1~5=체력·공격력·사거리·발사속도·이동속도


class NoReward:
    """환경이 계산하는 기본 보상을 쓰지 않기 위한 자리 채우기. 보상은 compute_reward로 직접 계산한다."""

    def calculate(self, events: dict) -> float:
        return 0.0


def make_env(render_mode: str | None = None) -> TankEnv:
    """벡터 관찰·이산 행동(이동·조준·발사 + 스탯 + 전직)의 학습 환경을 만든다."""
    env = TankEnv(
        GameConfig(),
        VectorObservation(),
        DiscreteActionDecoder(enable_stat_upgrades=True, enable_evolution=True),
        NoReward(),
        render_mode=render_mode,
    )
    if render_mode == "human":
        # 환경 코드는 그대로 두고 속도 제한값(metadata)만 바꾼다. 0이면 pygame 시계가 제한하지 않는다.
        base_fps = env.metadata["render_fps"]
        env.metadata["render_fps"] = 0 if RENDER_SPEED <= 0 else max(1, round(base_fps * RENDER_SPEED))
    return env


def n_evolution_choices(env: TankEnv) -> int:
    """전직 선택 번호 개수(0=선택 안 함 포함). 환경의 전체 행동 수에서 구한다."""
    return int(env.action_space.n) // (N_CONTROL * N_STAT)


def compose_action(control: int, stat: int, evolution: int, n_evolution: int) -> int:
    """분기 세 개의 선택을 환경이 받는 행동 번호 하나로 합친다. (규칙 없이 번호만 조합한다)"""
    return (int(control) * N_STAT + int(stat)) * n_evolution + int(evolution)


def compute_reward(events: dict, score_delta: float) -> float:
    """승인된 보상표대로 한 환경 step의 보상을 계산한다."""
    return float(
        score_gain * score_delta
        + shape_damage * events["shape_damage_weighted"]
        + shape_destroy * events["shape_destroyed_score"]
        + enemy_damage * events["enemy_damage"]
        + enemy_kill * events["enemies_killed"]
        + damage_taken * events["damage_taken"]
        + (death if events["died"] else 0.0)
        + time_step
        - shot_cost * events["player_shots_fired"]
    )


# --------------------------------------------------------------------------
# 병렬 환경: 환경마다 별도 프로세스에서 step을 실행한다
# --------------------------------------------------------------------------
class WorkerError:
    """워커에서 난 오류를 메인 프로세스로 전달하는 상자."""

    def __init__(self, text: str) -> None:
        self.text = text


def _worker(conn, seed_base: int) -> None:
    """환경 하나를 맡아 명령(reset·step·close)을 처리한다. 판이 끝나면 바로 새 판을 시작한다."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)  # 코어를 환경끼리 나눠 쓰므로 워커마다 스레드 1개
    env = None
    try:
        env = make_env(None)
        n_evolution = n_evolution_choices(env)
        episode, ep_return, prev_score = 0, 0.0, 0
        while True:
            command, data = conn.recv()
            if command == "reset":
                obs, info = env.reset(seed=seed_base + episode)
                ep_return, prev_score = 0.0, info["score"]
                conn.send(obs)
            elif command == "step":
                control, stat, evolution = data
                obs, _, terminated, truncated, info = env.step(
                    compose_action(control, stat, evolution, n_evolution))
                reward = compute_reward(info["events"], info["score"] - prev_score)
                prev_score = info["score"]
                ep_return += reward
                stats, after_obs = None, obs
                if terminated or truncated:
                    stats = {"survival": info["frames"] / env.game_config.fps, "return": ep_return,
                             "score": info["score"], "kills": info["kills"], "level": info["level"],
                             "steps": info["steps"]}
                    episode += 1
                    after_obs, info = env.reset(seed=seed_base + episode)  # 다음 판 첫 관찰
                    ep_return, prev_score = 0.0, info["score"]
                conn.send((obs, reward, terminated, truncated, stats, after_obs))
            else:
                break
    except Exception:
        conn.send(WorkerError(traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class ParallelEnvs:
    """환경 여러 개를 프로세스로 띄워 한꺼번에 reset·step 한다."""

    def __init__(self, num_envs: int, seed: int) -> None:
        context = mp.get_context("spawn")  # 윈도·리눅스(코랩) 공통으로 안전한 방식
        self.conns, self.procs = [], []
        for index in range(num_envs):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, seed + index * 1_000_003), daemon=True)
            process.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(process)

    def _receive(self, conn):
        message = conn.recv()
        if isinstance(message, WorkerError):
            raise RuntimeError(f"환경 워커 오류:\n{message.text}")
        return message

    def reset(self) -> np.ndarray:
        for conn in self.conns:
            conn.send(("reset", None))
        return np.stack([self._receive(conn) for conn in self.conns])

    def step(self, actions: np.ndarray) -> list[tuple]:
        for conn, action in zip(self.conns, actions):
            conn.send(("step", tuple(int(x) for x in action)))
        return [self._receive(conn) for conn in self.conns]

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.procs:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


# --------------------------------------------------------------------------
# 리플레이 버퍼: GPU 메모리 + 우선순위 샘플링(PER)
# --------------------------------------------------------------------------
class ReplayBuffer:
    """경험 (s, 분기 행동 3개, N-step 보상, ns, terminated, 할인)을 원형으로 저장하고 우선순위로 뽑는다.

    관찰이 5139차원이라 float16으로 저장해 메모리를 아낀다(관찰 값은 [-1, 1] 범위).
    DEVICE가 GPU면 전부 GPU에 두어 샘플링 때 복사가 없다. 새 경험은 현재 최대 우선순위를 받는다.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.states = torch.empty((capacity, obs_dim), dtype=torch.float16, device=DEVICE)
        self.next_states = torch.empty((capacity, obs_dim), dtype=torch.float16, device=DEVICE)
        self.actions = torch.empty((capacity, 3), dtype=torch.int64, device=DEVICE)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=DEVICE)
        self.terminated = torch.empty(capacity, dtype=torch.float32, device=DEVICE)
        self.discounts = torch.empty(capacity, dtype=torch.float32, device=DEVICE)  # 다음 상태 가치에 곱할 γ^k
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=DEVICE)
        self.max_priority = torch.ones((), dtype=torch.float32, device=DEVICE)
        self.pos = 0  # 다음에 쓸 위치
        self.size = 0  # 현재 저장된 경험 수

    def add_batch(self, items: list[tuple]) -> None:
        """(s, (제어, 스탯, 전직), r, ns, terminated, discount) 여러 개를 한 번에 저장한다."""
        count = len(items)
        if count == 0:
            return
        index = (self.pos + torch.arange(count, device=DEVICE)) % self.capacity
        self.states[index] = torch.as_tensor(np.stack([item[0] for item in items]), dtype=torch.float16, device=DEVICE)
        self.actions[index] = torch.as_tensor([item[1] for item in items], dtype=torch.int64, device=DEVICE)
        self.rewards[index] = torch.as_tensor([item[2] for item in items], dtype=torch.float32, device=DEVICE)
        self.next_states[index] = torch.as_tensor(np.stack([item[3] for item in items]), dtype=torch.float16, device=DEVICE)
        self.terminated[index] = torch.as_tensor([float(item[4]) for item in items], dtype=torch.float32, device=DEVICE)
        self.discounts[index] = torch.as_tensor([item[5] for item in items], dtype=torch.float32, device=DEVICE)
        self.priorities[index] = self.max_priority
        self.pos = (self.pos + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int, beta: float) -> tuple[torch.Tensor, ...]:
        """우선순위에 비례해 batch_size개를 뽑는다. 중요도 가중치(편향 보정)도 함께 반환한다."""
        weights_pa = self.priorities[: self.size] ** PER_ALPHA
        probs = weights_pa / weights_pa.sum()
        index = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self.size * probs[index]) ** (-beta)
        weights = weights / weights.max()
        return (
            index,
            self.states[index].float(),
            self.actions[index],
            self.rewards[index].unsqueeze(1),
            self.next_states[index].float(),
            self.terminated[index].unsqueeze(1),
            self.discounts[index].unsqueeze(1),
            weights.unsqueeze(1),
        )

    def update_priorities(self, index: torch.Tensor, td_error: torch.Tensor) -> None:
        """학습에서 나온 TD 오차 크기로 우선순위를 갱신한다."""
        priority = td_error.abs().detach().squeeze(1) + 1e-5
        self.priorities[index] = priority
        self.max_priority = torch.maximum(self.max_priority, priority.max())

    def __len__(self) -> int:
        return self.size


def store_nstep(
    out: list,
    window: deque,
    next_state: np.ndarray,
    terminated: bool,
    episode_over: bool,
    n_step: int,
    gamma: float,
) -> None:
    """window의 (s, 행동, r) 앞쪽부터 N-step 경험으로 묶어 out 목록에 추가한다.

    window가 가득 차면 가장 오래된 하나를 내보낸다. 판이 끝나면(episode_over) 남은 것도 모두 내보낸다.
    남은 step 수 k만큼만 합치고 할인은 γ^k다. 시간 제한은 terminated=False라 다음 상태에서 이어 붙인다.
    """
    def pop_oldest() -> None:
        ret = sum(gamma ** i * item[2] for i, item in enumerate(window))
        out.append((window[0][0], window[0][1], ret, next_state, terminated, gamma ** len(window)))
        window.popleft()

    if len(window) == n_step:
        pop_oldest()
    if episode_over:
        while window:
            pop_oldest()


# --------------------------------------------------------------------------
# 신경망: 관찰 구조를 이용하는 분기(branching) Dueling Q 네트워크
# --------------------------------------------------------------------------
def make_obs_spec(n_evolution: int) -> dict:
    """VectorObservation의 구간 정보와 분기별 행동 개수를 신경망이 쓸 수 있는 형태로 정리한다."""
    builder = VectorObservation()
    config = builder.config
    slices = builder.slices
    return {
        "self": (slices["self"].start, slices["self"].stop),
        "progression": (slices["progression"].start, slices["progression"].stop),
        "enemies": (slices["enemies"].start, slices["enemies"].stop, config.n_enemies, config.ENEMY_DIM),
        "shapes": (slices["shapes"].start, slices["shapes"].stop, config.n_shapes, config.SHAPE_DIM),
        "bullets": (slices["bullets"].start, slices["bullets"].stop, config.n_bullets, config.BULLET_DIM),
        "grid": (slices["grid"].start, slices["grid"].stop, *builder.grid_shape),
        "size": int(builder.observation_space.shape[0]),
        "branches": (N_CONTROL, N_STAT, n_evolution),
    }


class EntityEncoder(nn.Module):
    """개체(적·천체·탄환) 슬롯마다 같은 MLP를 적용한다. 존재 표시(첫 값)가 0인 슬롯은 출력을 0으로 만든다."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * x[..., :1]


class BranchingQNetwork(nn.Module):
    """지형은 CNN, 개체는 슬롯 공유 인코더, 자기 상태·성장은 MLP로 읽어 합친 뒤 분기별 Q를 낸다.

    상태 가치 V(s)는 모든 분기가 공유하고, 분기마다 행동 이득 A_b(s, a_b)를 따로 낸다.
    Q_b = V + A_b - mean(A_b). 출력은 (제어 Q, 스탯 Q, 전직 Q) 세 텐서다.
    """

    def __init__(self, spec: dict) -> None:
        super().__init__()
        self.spec = spec
        _, _, grid_channels, grid_height, grid_width = spec["grid"]
        self.status = nn.Sequential(
            nn.Linear(spec["self"][1] - spec["self"][0] + spec["progression"][1] - spec["progression"][0], 128),
            nn.ReLU(),
        )
        enemy_slots = spec["enemies"][2]
        self.enemy_encoder = EntityEncoder(spec["enemies"][3], 64, 32)
        self.shape_encoder = EntityEncoder(spec["shapes"][3], 32, 32)
        self.bullet_encoder = EntityEncoder(spec["bullets"][3], 32, 32)
        self.grid = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            grid_features = self.grid(torch.zeros(1, grid_channels, grid_height, grid_width)).shape[1]
        fused = 128 + enemy_slots * 32 + 32 + 64 + grid_features  # 상태 + 적(순서 유지) + 천체(max) + 탄환(max·mean) + 지형
        self.fuse = nn.Sequential(nn.Linear(fused, 512), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantages = nn.ModuleList(
            nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, count)) for count in spec["branches"]
        )

    def _slots(self, x: torch.Tensor, key: str) -> torch.Tensor:
        start, stop, slots, dim = self.spec[key]
        return x[:, start:stop].reshape(x.shape[0], slots, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        spec = self.spec
        status = self.status(torch.cat([x[:, spec["self"][0]:spec["self"][1]],
                                        x[:, spec["progression"][0]:spec["progression"][1]]], dim=1))
        enemies = self.enemy_encoder(self._slots(x, "enemies")).flatten(1)  # 가까운 순서를 유지한다
        shapes = self.shape_encoder(self._slots(x, "shapes")).amax(dim=1)
        bullet_slots = self._slots(x, "bullets")
        bullets = self.bullet_encoder(bullet_slots)
        bullet_count = bullet_slots[..., :1].sum(dim=1).clamp(min=1.0)
        bullet_features = torch.cat([bullets.amax(dim=1), bullets.sum(dim=1) / bullet_count], dim=1)
        grid_start, grid_stop, channels, height, width = spec["grid"]
        grid = self.grid(x[:, grid_start:grid_stop].reshape(x.shape[0], channels, height, width))
        z = self.fuse(torch.cat([status, enemies, shapes, bullet_features, grid], dim=1))
        value = self.value(z)
        outputs = []
        for head in self.advantages:
            advantage = head(z)
            outputs.append(value + advantage - advantage.mean(dim=1, keepdim=True))
        return tuple(outputs)


def dqn_update(
    online: BranchingQNetwork,
    target: BranchingQNetwork,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    beta: float,
    double: bool,
) -> tuple[float, float]:
    """우선순위로 뽑은 배치로 온라인 신경망을 한 번 갱신하고 (손실, 평균 Q)를 반환한다.

    분기 DQN: 다음 상태 가치는 분기별 최선 행동의 타깃 Q를 평균해서 구하고,
    모든 분기가 같은 목표값 y를 향해 학습한다.
    """
    index, states, actions, rewards, next_states, terminated, discounts, weights = buffer.sample(batch_size, beta)
    branch_q = online(states)
    chosen = [q.gather(1, actions[:, b:b + 1]) for b, q in enumerate(branch_q)]  # 실제 선택한 행동의 Q (분기별)
    with torch.no_grad():
        target_q = target(next_states)
        picks_from = online(next_states) if double else target_q  # Double이면 온라인으로 고르고 타깃으로 평가
        picks = [q.argmax(dim=1, keepdim=True) for q in picks_from]
        next_value = torch.stack([tq.gather(1, p) for tq, p in zip(target_q, picks)]).mean(dim=0)
        # rewards는 N-step 보상 합, discounts는 γ^k. 사망이면 미래 가치를 제외하고 시간 제한은 이어 붙인다.
        targets = rewards + discounts * (1 - terminated) * next_value
    losses = [nn.functional.smooth_l1_loss(q, targets, reduction="none") * weights for q in chosen]
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    mean_chosen = torch.stack(chosen).mean(dim=0)
    buffer.update_priorities(index, targets - mean_chosen)
    return float(loss.item()), float(mean_chosen.mean().item())


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """타깃 가중치를 온라인 쪽으로 tau만큼 조금 이동시킨다."""
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.lerp_(online_param, tau)


def greedy_actions(online: BranchingQNetwork, states: np.ndarray) -> np.ndarray:
    """여러 환경의 관찰을 한 번에 넣어 분기마다 Q값이 가장 큰 행동을 (환경 수, 3) 배열로 반환한다."""
    with torch.no_grad():
        tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        return torch.stack([q.argmax(dim=1) for q in online(tensor)], dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# 평가: 보상과 분리된 게임 원래 지표 + 무작위 기준선
# --------------------------------------------------------------------------
def run_episodes(env: TankEnv, choose, seeds: list[int]) -> dict:
    """주어진 시드마다 한 판씩 플레이하고 게임 원래 지표(생존 시간·점수·처치·레벨)를 모은다.

    choose(관찰) -> (제어, 스탯, 전직) 선택을 돌려주는 함수다. 보상은 쓰지 않는다.
    """
    n_evolution = n_evolution_choices(env)
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            control, stat, evolution = choose(obs)
            obs, _, terminated, truncated, info = env.step(compose_action(control, stat, evolution, n_evolution))
        episodes.append({"seed": seed, "survival": info["frames"] / env.game_config.fps, "score": info["score"],
                         "kills": info["kills"], "level": info["level"], "died": bool(terminated)})
    summary = {key: float(np.mean([e[key] for e in episodes])) for key in ("survival", "score", "kills", "level")}
    return {"episodes": episodes, "summary": summary}


def random_chooser(spec: dict, rng: np.random.Generator):
    """기준선: 세 분기를 모두 균등하게 무작위로 고르는 정책."""
    branches = spec["branches"]
    return lambda obs: tuple(int(rng.integers(0, count)) for count in branches)


def model_chooser(model: BranchingQNetwork):
    """학습된 모델이 탐험 없이 분기마다 최선의 행동을 고르는 정책."""
    return lambda obs: tuple(int(x) for x in greedy_actions(model, obs[None])[0])


def quality(summary: dict, baseline: dict) -> tuple[float, float, float]:
    """기준선 대비 생존 시간 비율, 점수 비율, 둘의 기하평균(종합)을 반환한다. 둘 다 올라야 종합이 커진다."""
    survival_ratio = summary["survival"] / max(baseline["survival"], 1e-6)
    score_ratio = (summary["score"] + 1.0) / (baseline["score"] + 1.0)  # 점수 0에서도 비율이 정의되도록 +1
    return survival_ratio, score_ratio, float(np.sqrt(survival_ratio * score_ratio))


# --------------------------------------------------------------------------
# 저장
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, online: BranchingQNetwork) -> None:
    """모델 가중치와 재생에 필요한 형식 정보를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": online.state_dict(), "n_evolution": online.spec["branches"][2]}, path)


def load_checkpoint(path: Path) -> BranchingQNetwork:
    """저장한 모델을 불러온다."""
    data = torch.load(path, map_location=DEVICE)
    model = BranchingQNetwork(make_obs_spec(data["n_evolution"])).to(DEVICE)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    """병렬 환경으로 경험을 모으며 DQN을 학습하고 체크포인트와 로그(CSV)를 저장한다."""
    set_seed(args.seed)
    num_envs = args.num_envs or max(1, min(os.cpu_count() or 1, 8))
    main_env = make_env(None)  # 평가용이자 행동 개수·관찰 크기를 읽는 용도
    n_evolution = n_evolution_choices(main_env)
    spec = make_obs_spec(n_evolution)
    obs_dim = spec["size"]
    goal = f"{args.episodes} 에피소드" if args.episodes else f"{args.steps} step"
    print(f"장치={DEVICE} | 병렬 환경={num_envs}개 | 관찰={obs_dim}차원 | 분기 행동={spec['branches']} | 목표={goal}")

    online = BranchingQNetwork(spec).to(DEVICE)
    target = BranchingQNetwork(spec).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr, eps=ADAM_EPS)
    replay = ReplayBuffer(args.buffer, obs_dim)

    out_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d%H%M%S_dqn03")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["episode", "steps", "env", "ep_steps", "survival_sec", "return", "score", "kills", "level",
                  "epsilon", "loss", "mean_q"])
    eval_file = open(out_dir / "eval.csv", "w", newline="", encoding="utf-8")
    eval_log = csv.writer(eval_file)
    eval_log.writerow(["steps", "survival", "score", "kills", "level", "survival_ratio", "score_ratio", "quality"])

    eval_seeds = [EVAL_SEED_BASE + i for i in range(args.eval_episodes)]  # 학습에 쓰지 않는 시드
    baseline = None
    if args.eval_every_steps > 0:
        baseline = run_episodes(main_env, random_chooser(spec, np.random.default_rng(args.seed)), eval_seeds)["summary"]
        print(f"[기준선: 무작위 정책] 생존 {baseline['survival']:.1f}초 | 점수 {baseline['score']:.1f} | "
              f"처치 {baseline['kills']:.1f} | 레벨 {baseline['level']:.1f}  (평가 시드 {eval_seeds[0]}~, {len(eval_seeds)}판)")

    envs = ParallelEnvs(num_envs, args.seed)
    obs = envs.reset()
    windows = [deque() for _ in range(num_envs)]  # 환경별 N-step 계산용 최근 (s, 행동, r)
    eps_scale = np.linspace(1.0, 0.25, num_envs) if num_envs > 1 else np.ones(1)  # 환경마다 탐험 정도를 다르게
    rng = np.random.default_rng(args.seed + 1)

    recent = {key: deque(maxlen=20) for key in ("return", "survival", "score", "kills", "level")}
    losses: deque[float] = deque(maxlen=200)
    q_means: deque[float] = deque(maxlen=200)
    best_quality = -float("inf")
    steps, episode, updates, update_credit = 0, 0, 0, 0.0
    last_eval_bucket, last_save_bucket = 0, 0
    started = time.time()

    try:
        while not (episode >= args.episodes if args.episodes else steps >= args.steps):
            epsilon = np.maximum(args.eps_end, max(args.eps_end, 1.0 - steps / args.eps_decay_steps) * eps_scale)
            greedy = greedy_actions(online, obs)
            random_actions = np.stack([rng.integers(0, count, size=num_envs) for count in spec["branches"]], axis=1)
            explore = rng.random((num_envs, 3)) < epsilon[:, None]  # 분기마다 독립적으로 탐험
            actions = np.where(explore, random_actions, greedy)

            results = envs.step(actions)
            new_obs = np.empty_like(obs)
            transitions: list[tuple] = []
            for env_id, (next_obs, reward, terminated, truncated, stats, after_obs) in enumerate(results):
                windows[env_id].append((obs[env_id], tuple(int(x) for x in actions[env_id]), reward * args.reward_scale))
                store_nstep(transitions, windows[env_id], next_obs, terminated, terminated or truncated,
                            args.n_step, args.gamma)
                new_obs[env_id] = after_obs
                if stats is not None:
                    episode += 1
                    for key in recent:
                        recent[key].append(stats[key])
                    mean_loss = float(np.mean(losses)) if losses else float("nan")
                    mean_q = float(np.mean(q_means)) if q_means else float("nan")
                    log.writerow([episode, steps + num_envs, env_id, stats["steps"], f"{stats['survival']:.1f}",
                                  f"{stats['return']:.4f}", stats["score"], stats["kills"], stats["level"],
                                  f"{epsilon[env_id]:.3f}", f"{mean_loss:.5f}", f"{mean_q:.4f}"])
                    log_file.flush()
                    if episode % args.log_every == 0:
                        speed = (steps + num_envs) / max(time.time() - started, 1e-9)
                        print(f"에피소드 {episode:5d} | step {steps + num_envs:8d} | "
                              f"최근20 생존 {np.mean(recent['survival']):5.0f}초 점수 {np.mean(recent['score']):6.1f} "
                              f"처치 {np.mean(recent['kills']):4.1f} 레벨 {np.mean(recent['level']):4.1f} "
                              f"보상 {np.mean(recent['return']):7.2f} | ε {epsilon.mean():.2f} | "
                              f"loss {mean_loss:.4f} Q {mean_q:.2f} | {speed:.0f} step/s")
            obs = new_obs
            steps += num_envs
            replay.add_batch(transitions)

            # 모인 step 수에 비례해 학습한다(환경이 많아도 학습 비율은 같다).
            if len(replay) >= args.learn_start:
                update_credit += num_envs / args.train_every
                beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * steps / PER_BETA_STEPS)
                while update_credit >= 1.0:
                    update_credit -= 1.0
                    loss_value, q_value = dqn_update(online, target, replay, optimizer, args.batch, beta, args.double)
                    soft_update(target, online, TAU)
                    losses.append(loss_value)
                    q_means.append(q_value)
                    updates += 1

            if steps // SAVE_EVERY_STEPS > last_save_bucket:
                last_save_bucket = steps // SAVE_EVERY_STEPS
                save_checkpoint(out_dir / "dqn03_last.pt", online)

            if baseline is not None and steps // args.eval_every_steps > last_eval_bucket:
                last_eval_bucket = steps // args.eval_every_steps
                online.eval()
                summary = run_episodes(main_env, model_chooser(online), eval_seeds)["summary"]
                online.train()
                survival_ratio, score_ratio, overall = quality(summary, baseline)
                eval_log.writerow([steps, f"{summary['survival']:.1f}", f"{summary['score']:.1f}",
                                   f"{summary['kills']:.2f}", f"{summary['level']:.2f}",
                                   f"{survival_ratio:.3f}", f"{score_ratio:.3f}", f"{overall:.3f}"])
                eval_file.flush()
                print(f"  [평가 @{steps}] 생존 {summary['survival']:.1f}초 (기준선 대비 ×{survival_ratio:.2f}) | "
                      f"점수 {summary['score']:.1f} (×{score_ratio:.2f}) | 처치 {summary['kills']:.1f} | "
                      f"레벨 {summary['level']:.1f} | 종합 ×{overall:.2f}")
                if overall > best_quality:
                    best_quality = overall
                    save_checkpoint(out_dir / "dqn03_best.pt", online)
                save_checkpoint(out_dir / "dqn03_last.pt", online)
    finally:
        # 직접 중지(Ctrl+C·코랩 중지)해도 마지막 상태를 저장하고 총 판 수를 출력한다.
        save_checkpoint(out_dir / "dqn03_last.pt", online)
        log_file.close()
        eval_file.close()
        envs.close()
        main_env.close()
        print(f"학습 종료: 총 {episode} 에피소드, {steps} step, 학습 {updates}회 | 저장 완료: {out_dir.resolve()}")


def play(args: argparse.Namespace) -> None:
    """저장한 모델을 최종 검증용 시드(TEST_SEED_BASE~)에서 평가한다. --baseline이면 같은 시드의 무작위 정책도 함께 보여 준다.

    학습 중 평가 시드(EVAL_SEED_BASE~)로 최고 모델을 골랐으므로, 선택에 쓰지 않은 시드로 다시 재야 운 좋은 선택이 걸러진다.
    판 수가 적으면 결과가 크게 흔들리니 10판 이상을 권장한다.
    """
    model = load_checkpoint(Path(args.play))
    spec = make_obs_spec(model.spec["branches"][2])
    seeds = [TEST_SEED_BASE + i for i in range(args.eval_episodes)]
    env = make_env(RENDER_MODE)
    try:
        result = run_episodes(env, model_chooser(model), seeds)
        for episode in result["episodes"]:
            print(episode)
        print("[모델 평균]", {key: round(value, 2) for key, value in result["summary"].items()})
        if args.baseline:
            base_env = make_env(None)
            try:
                base = run_episodes(base_env, random_chooser(spec, np.random.default_rng(args.seed)), seeds)["summary"]
            finally:
                base_env.close()
            survival_ratio, score_ratio, overall = quality(result["summary"], base)
            print("[무작위 기준선 평균]", {key: round(value, 2) for key, value in base.items()})
            print(f"기준선 대비: 생존 ×{survival_ratio:.2f} | 점수 ×{score_ratio:.2f} | 종합 ×{overall:.2f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Space Z DQN_03 학습(분기 DQN·병렬 환경·PER)")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES,
                        help="학습할 에피소드 수(모든 환경의 합). 0보다 크면 --steps 대신 이 값으로 끝낸다")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="학습할 환경 step 수(--episodes가 0일 때만 사용)")
    parser.add_argument("--seed", type=int, default=SEED, help="난수 시드")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="병렬 환경 수(0이면 자동)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam 학습률")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="할인율")
    parser.add_argument("--n-step", type=int, default=N_STEP, help="N-step 반환의 N(1이면 1-step)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="배치 크기")
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE, help="리플레이 버퍼 크기")
    parser.add_argument("--learn-start", type=int, default=LEARN_START, help="이 개수 이상 쌓이면 학습 시작")
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY, help="환경 step이 몇 개 모일 때마다 학습할지")
    parser.add_argument("--eps-end", type=float, default=EPS_END, help="ε 최솟값")
    parser.add_argument("--eps-decay-steps", type=int, default=EPS_DECAY_STEPS, help="ε이 1.0에서 최솟값까지 줄어드는 step 수")
    parser.add_argument("--reward-scale", type=float, default=REWARD_SCALE, help="학습용 보상 배율(1.0=원본)")
    parser.add_argument("--double", action=argparse.BooleanOptionalAction, default=DOUBLE_DQN,
                        help="Double DQN 사용(끄려면 --no-double)")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY, help="몇 에피소드마다 출력할지")
    parser.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS, help="몇 step마다 평가할지(0이면 끔)")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="평가 에피소드 수")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="체크포인트·로그 저장 폴더(코랩은 Drive 경로 권장)")
    parser.add_argument("--play", type=str, default=None, help="학습 대신 저장한 .pt 모델을 평가한다")
    parser.add_argument("--baseline", action="store_true", help="--play에서 같은 시드의 무작위 정책도 함께 평가한다")
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0 or args.episodes < 0 or args.num_envs < 0:
        parser.error("--steps는 양수, --episodes·--seed·--num-envs는 0 이상이어야 합니다.")
    if args.n_step < 1:
        parser.error("--n-step은 1 이상이어야 합니다.")
    if args.reward_scale <= 0:
        parser.error("--reward-scale은 양수여야 합니다.")
    play(args) if args.play else train(args)


if __name__ == "__main__":
    main()
