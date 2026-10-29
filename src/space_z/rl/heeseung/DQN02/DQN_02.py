"""DQN_02 학습 실행: python -m space_z.rl.DQN_02

DQN.py보다 성능을 우선한 버전이다. 바뀐 점:
  1) 환경을 여러 프로세스에서 병렬로 실행해 경험을 빨리 모은다(NUM_ENVS).
  2) 평평한 MLP 대신 관찰 구조를 이용하는 네트워크(지형 CNN + 개체별 인코더)를 쓴다.
  3) Double + Dueling + N-step에 우선순위 리플레이(PER)를 더했다.
  4) 리플레이를 GPU 메모리에 두어 샘플링 때 CPU→GPU 복사를 없앴다.
  5) 환경마다 탐험 확률을 다르게 두고, 타깃은 조금씩 따라가게(soft update) 갱신한다.

행동 방식·보상은 DQN.py와 같다(54개 행동 + 규칙 기반 성장, 공통 RewardConfig와 같은 이름의 보상 변수).
추가 의존성: torch. 프로젝트 설치는 python -m pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from space_z.config import GameConfig, RewardConfig
from space_z.config.progression_definitions import STAT_UPGRADES
from space_z.game.actions import N_MOVES, N_ROTS
from space_z.rl.actions.discrete_action_decoder import DiscreteActionDecoder
from space_z.rl.environment import TankEnv
from space_z.rl.evaluate import PolicyEvaluator
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.rewards import RewardCalculator

# --------------------------------------------------------------------------
# 하이퍼파라미터 (여기서 수정한다. 실행할 때 --옵션을 주면 이 값보다 우선한다)
# --------------------------------------------------------------------------
# 학습 길이: TRAIN_EPISODES가 0보다 크면 전체 환경이 끝낸 판 수의 합에서 끝나고, 0이면 TRAIN_STEPS에서 끝난다.
TRAIN_EPISODES = 0  # 학습할 에피소드 수(모든 환경의 합). 0이면 아래 step 기준
TRAIN_STEPS = 300000  # 학습할 환경 step 수(모든 환경의 합, TRAIN_EPISODES가 0일 때만 사용)
SEED = 42  # 난수 시드

NUM_ENVS = 0  # 병렬 환경(프로세스) 수. 0이면 CPU 코어 수에 맞춰 자동(최대 8). 코랩 기본은 코어 2개

LEARNING_RATE = 2e-4  # Adam 학습률
ADAM_EPS = 1.5e-4  # Adam 안정화 값
GAMMA = 0.99  # 할인율
N_STEP = 3  # N-step 반환: 몇 step 뒤까지의 보상을 합쳐서 학습할지 (1이면 1-step)
BATCH_SIZE = 256  # 한 번의 경사 계산에 쓰는 경험 수
BUFFER_SIZE = 100000  # 리플레이 버퍼 크기. GPU 메모리 약 2GB(10만 개 기준)
LEARN_START = 5000  # 버퍼에 이 개수 이상 쌓이면 학습 시작
TRAIN_EVERY = 8  # 모든 환경 step이 이만큼 모일 때마다 학습 1회
TAU = 0.005  # 타깃 신경망이 온라인을 따라가는 비율(학습 1회마다 적용)
REWARD_SCALE = 0.1  # 학습용 보상 배율(1.0=원본). 로그에 찍히는 보상은 원본 그대로

PER_ALPHA = 0.6  # 우선순위 반영 정도(0이면 균등 샘플링)
PER_BETA_START = 0.4  # 중요도 보정 시작 값
PER_BETA_STEPS = 500000  # 중요도 보정이 1.0이 되는 step 수

EPS_END = 0.05  # ε 최솟값 (탐험 확률의 하한)
EPS_DECAY_STEPS = 150000  # ε이 1.0에서 EPS_END까지 줄어드는 step 수(모든 환경의 합)

LOG_EVERY = 10  # 몇 에피소드마다 학습 상황을 출력할지
EVAL_EVERY = 100  # 몇 에피소드마다 평가할지 (0이면 평가 끔)
EVAL_EPISODES = 3  # 평가할 때 돌리는 에피소드 수
SAVE_EVERY_STEPS = 50000  # 몇 step마다 dqn02_last.pt를 저장할지
SAVE_DIR = "trained_models"  # 체크포인트·로그 저장 폴더 (코랩은 Drive 경로 권장)

# --------------------------------------------------------------------------
# 사용자 설정
# --------------------------------------------------------------------------
# 행동 방식: 아래 세 줄 중 한 줄만 주석을 풀고 나머지는 주석 처리한다.
ACTION_MODE = "auto_growth"  # DQN은 54개(이동·회전·발사)만 고르고, 스탯 강화·전직은 아래 규칙이 자동 선택
# ACTION_MODE = "move_only"  # 54개. 스탯 강화·전직을 아예 사용하지 않음
# ACTION_MODE = "full_1620"  # 스탯 강화·전직까지 DQN이 1620개 전부 고름(PPO와 동일)

# 자동 성장 규칙(auto_growth일 때만 사용)
#  - 스탯: 포인트가 있으면 강화 단계가 가장 낮은 스탯부터 올린다(같으면 체력→공격력→사거리→발사속도→이동속도).
#  - 전직: 아래 순서에서 먼저 나오는 선택지를 고른다. 생존 목적이라 가디언 계열(체력·피해 감소)을 우선한다.
#    목록에 없거나 숙련 보상(선택지 1개)이면 첫 번째 선택지를 고른다.
EVOLUTION_PREFERENCE = ("guardian", "aegis", "citadel", "bastion", "colossus")

# 보상: 변수 이름은 공통 RewardConfig(learning.py)의 필드와 같다. 값은 이 파일에서만 쓰며 learning.py 원본은 바뀌지 않는다.
# time_step을 뺀 나머지는 RewardConfig 기본값과 같다. 기본값 -0.0001은 오래 살수록 깎여서 생존이 목적이면 양수로 바꾼다.
shape_damage = 0.002  # 천체 피해 보상 계수: 피해량 × (점수/최대 체력)에 곱함
shape_destroy = 0.01  # 천체 파괴 보상 계수: 파괴한 천체 점수에 곱함
enemy_damage = 0.01  # 적 피해 보상 계수: 적에게 준 피해량에 곱함
enemy_kill = 2.0  # 적 한 대 처치 시 보상
damage_taken = -0.02  # 피격 보상 계수: 받은 피해량에 곱함 (음수면 페널티)
death = -20.0  # 플레이어 사망 시 보상
time_step = 0.01  # step마다 더해지는 값 (생존 보너스). 공통 기본값은 -0.0001
shot_cost = 0.01  # 실제 발사 탄환 1발당 차감할 보상 (0 이상)

# 화면 표시(--play 평가에서만 사용). 학습은 병렬 프로세스라 화면 없이 진행한다.
RENDER_MODE = "human"  # 화면 없이 평가
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
    """파이썬·NumPy·torch 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 환경과 행동 변환 (DQN.py와 같은 규칙)
# --------------------------------------------------------------------------
BASE_ACTIONS = N_MOVES * N_ROTS * 2  # 이동 × 회전 × 발사 = 54개
STAT_CHOICES = len(STAT_UPGRADES) + 1  # 스탯 강화 번호 개수(0=선택 안 함)


def make_env(render_mode: str | None = None) -> TankEnv:
    """벡터 관찰·이산 행동의 학습 환경을 만든다. 보상 계수는 파일 상단의 RewardConfig와 같은 이름의 변수를 쓴다."""
    growth = ACTION_MODE in ("auto_growth", "full_1620")  # auto_growth도 환경의 행동 번호에는 성장 선택이 필요
    env = TankEnv(
        GameConfig(),
        VectorObservation(),
        DiscreteActionDecoder(enable_stat_upgrades=growth, enable_evolution=growth),
        RewardCalculator(RewardConfig(
            shape_damage=shape_damage,
            shape_destroy=shape_destroy,
            enemy_damage=enemy_damage,
            enemy_kill=enemy_kill,
            damage_taken=damage_taken,
            death=death,
            time_step=time_step,
            shot_cost=shot_cost,
        )),
        render_mode=render_mode,
    )
    if render_mode == "human":
        # 환경 코드는 그대로 두고 속도 제한값(metadata)만 바꾼다. 0이면 pygame 시계가 제한하지 않는다.
        base_fps = env.metadata["render_fps"]
        env.metadata["render_fps"] = 0 if RENDER_SPEED <= 0 else max(1, round(base_fps * RENDER_SPEED))
    return env


def n_dqn_actions(env: TankEnv) -> int:
    """DQN이 직접 고르는 행동 개수. auto_growth는 환경이 1620개여도 DQN은 54개만 고른다."""
    return int(env.action_space.n) if ACTION_MODE == "full_1620" else BASE_ACTIONS


def pick_growth(env: TankEnv) -> tuple[int, int]:
    """현재 게임 상태를 읽기만 해서 (스탯 번호, 전직 번호)를 규칙으로 정한다. 0이면 선택 안 함."""
    progression = env.game.progression
    stat, lowest = 0, None
    for number, card in enumerate(STAT_UPGRADES, start=1):
        if progression.can_upgrade_stat(card.key):
            level = progression.stat_levels[card.key]
            if lowest is None or level < lowest:  # 같은 단계면 정의 순서(체력 먼저)를 유지
                stat, lowest = number, level
    evolution = 0
    choices = progression.reward_choices  # 레벨업 때 이미 열려 있다. 여기서 열거나 바꾸지 않는다.
    if choices:
        keys = [ship.key for ship in choices]
        index = next((keys.index(key) for key in EVOLUTION_PREFERENCE if key in keys), 0)
        evolution = index + 1
    return stat, evolution


def to_env_action(action: int, env: TankEnv) -> int:
    """DQN 행동 번호를 환경이 받는 행동 번호로 바꾼다. auto_growth만 규칙으로 고른 성장 선택을 끼워 넣는다."""
    if ACTION_MODE != "auto_growth":
        return action
    stat, evolution = pick_growth(env)
    evolution_count = int(env.action_space.n) // (BASE_ACTIONS * STAT_CHOICES)
    return (action * STAT_CHOICES + stat) * evolution_count + evolution


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
        episode, ep_return = 0, 0.0
        while True:
            command, data = conn.recv()
            if command == "reset":
                obs, _ = env.reset(seed=seed_base + episode)
                ep_return = 0.0
                conn.send(obs)
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(to_env_action(int(data), env))
                ep_return += reward
                stats, after_obs = None, obs
                if terminated or truncated:
                    stats = {"survival": info["frames"] / env.game_config.fps, "return": ep_return,
                             "score": info["score"], "kills": info["kills"], "steps": info["steps"]}
                    episode += 1
                    ep_return = 0.0
                    after_obs, _ = env.reset(seed=seed_base + episode)  # 다음 판 첫 관찰
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
            conn.send(("step", int(action)))
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
    """경험 (s, a, N-step 보상, ns, terminated, 할인)을 원형으로 저장하고 우선순위로 뽑는다.

    관찰이 5139차원이라 float16으로 저장해 메모리를 아낀다(관찰 값은 [-1, 1] 범위).
    DEVICE가 GPU면 전부 GPU에 두어 샘플링 때 복사가 없다. 새 경험은 현재 최대 우선순위를 받는다.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.states = torch.empty((capacity, obs_dim), dtype=torch.float16, device=DEVICE)
        self.next_states = torch.empty((capacity, obs_dim), dtype=torch.float16, device=DEVICE)
        self.actions = torch.empty(capacity, dtype=torch.int64, device=DEVICE)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=DEVICE)
        self.terminated = torch.empty(capacity, dtype=torch.float32, device=DEVICE)
        self.discounts = torch.empty(capacity, dtype=torch.float32, device=DEVICE)  # 다음 상태 가치에 곱할 γ^k
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=DEVICE)
        self.max_priority = torch.ones((), dtype=torch.float32, device=DEVICE)
        self.pos = 0  # 다음에 쓸 위치
        self.size = 0  # 현재 저장된 경험 수

    def add_batch(self, items: list[tuple]) -> None:
        """(s, a, r, ns, terminated, discount) 여러 개를 한 번에 저장한다."""
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
            self.actions[index].unsqueeze(1),
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
    """window의 (s, a, r) 앞쪽부터 N-step 경험으로 묶어 out 목록에 추가한다.

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
# 신경망: 관찰 구조를 이용하는 Dueling Q 네트워크
# --------------------------------------------------------------------------
def make_obs_spec() -> dict:
    """VectorObservation의 구간 정보를 신경망이 쓸 수 있는 형태로 정리한다."""
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
    }


class EntityEncoder(nn.Module):
    """개체(적·천체·탄환) 슬롯마다 같은 MLP를 적용한다. 존재 표시(첫 값)가 0인 슬롯은 출력을 0으로 만든다."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * x[..., :1]


class StructuredQNetwork(nn.Module):
    """지형은 CNN, 개체는 슬롯 공유 인코더, 자기 상태·성장은 MLP로 읽어 합친 뒤 Dueling으로 Q를 낸다."""

    def __init__(self, spec: dict, n_actions: int) -> None:
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
        self.advantage = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def _slots(self, x: torch.Tensor, key: str) -> torch.Tensor:
        start, stop, slots, dim = self.spec[key]
        return x[:, start:stop].reshape(x.shape[0], slots, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        advantage = self.advantage(z)
        return self.value(z) + advantage - advantage.mean(dim=1, keepdim=True)


def dqn_update(
    online: StructuredQNetwork,
    target: StructuredQNetwork,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    beta: float,
) -> tuple[float, float]:
    """우선순위로 뽑은 배치로 온라인 신경망을 한 번 갱신하고 (손실, 평균 Q)를 반환한다. Double DQN을 쓴다."""
    index, states, actions, rewards, next_states, terminated, discounts, weights = buffer.sample(batch_size, beta)
    chosen_q = online(states).gather(1, actions)  # 실제로 선택했던 행동의 Q값만 추출
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1, keepdim=True)  # 온라인으로 고르고
        next_q = target(next_states).gather(1, next_actions)  # 타깃으로 평가한다
        # rewards는 N-step 보상 합, discounts는 γ^k. 사망이면 미래 가치를 제외하고 시간 제한은 이어 붙인다.
        targets = rewards + discounts * (1 - terminated) * next_q
    td_error = targets - chosen_q
    loss = (nn.functional.smooth_l1_loss(chosen_q, targets, reduction="none") * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    buffer.update_priorities(index, td_error)
    return float(loss.item()), float(chosen_q.mean().item())


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """타깃 가중치를 온라인 쪽으로 tau만큼 조금 이동시킨다."""
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.lerp_(online_param, tau)


def greedy_actions(online: StructuredQNetwork, states: np.ndarray) -> np.ndarray:
    """여러 환경의 관찰을 한 번에 넣어 Q값이 가장 큰 행동 번호들을 반환한다."""
    with torch.no_grad():
        tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        return online(tensor).argmax(dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# 저장·평가
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, online: StructuredQNetwork, n_actions: int) -> None:
    """모델 가중치와 재생에 필요한 형식 정보를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": online.state_dict(), "n_actions": n_actions, "action_mode": ACTION_MODE}, path)


def load_checkpoint(path: Path) -> StructuredQNetwork:
    """저장한 모델을 불러온다. 행동 공간 설정이 다르면 오류를 낸다."""
    data = torch.load(path, map_location=DEVICE)
    if data["action_mode"] != ACTION_MODE:
        raise ValueError(f"ACTION_MODE 설정이 다릅니다. 저장 당시 값: {data['action_mode']}")
    model = StructuredQNetwork(make_obs_spec(), data["n_actions"]).to(DEVICE)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


def evaluate(online: StructuredQNetwork, env: TankEnv, seeds: list[int]) -> dict:
    """고정 시드로 탐험 없이(argmax) 평가한다. 평가 중에는 학습하지 않는다."""
    online.eval()
    try:
        selector = lambda obs: to_env_action(int(greedy_actions(online, obs[None])[0]), env)
        return PolicyEvaluator(env, selector).evaluate(seeds)
    finally:
        online.train()


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    """병렬 환경으로 경험을 모으며 DQN을 학습하고 체크포인트와 에피소드 로그(CSV)를 저장한다."""
    set_seed(args.seed)
    num_envs = args.num_envs or max(1, min(os.cpu_count() or 1, 8))
    main_env = make_env(None)  # 평가용이자 행동 개수·관찰 크기를 읽는 용도
    spec = make_obs_spec()
    obs_dim = spec["size"]
    n_actions = n_dqn_actions(main_env)
    goal = f"{args.episodes} 에피소드" if args.episodes else f"{args.steps} step"
    print(f"장치={DEVICE} | 병렬 환경={num_envs}개 | 관찰={obs_dim}차원 | DQN 행동={n_actions}개 | "
          f"ACTION_MODE={ACTION_MODE} | 목표={goal}")

    online = StructuredQNetwork(spec, n_actions).to(DEVICE)
    target = StructuredQNetwork(spec, n_actions).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr, eps=ADAM_EPS)
    replay = ReplayBuffer(args.buffer, obs_dim)

    out_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d%H%M%S_dqn02")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["episode", "steps", "env", "ep_steps", "survival_sec", "return", "score", "kills",
                  "epsilon", "loss", "mean_q"])

    envs = ParallelEnvs(num_envs, args.seed)
    obs = envs.reset()
    windows = [deque() for _ in range(num_envs)]  # 환경별 N-step 계산용 최근 (s, a, r)
    eps_scale = np.linspace(1.0, 0.25, num_envs) if num_envs > 1 else np.ones(1)  # 환경마다 탐험 정도를 다르게

    recent_returns: deque[float] = deque(maxlen=20)
    recent_survival: deque[float] = deque(maxlen=20)
    losses: deque[float] = deque(maxlen=200)
    q_means: deque[float] = deque(maxlen=200)
    best_eval = (-float("inf"), -float("inf"))  # (평가 생존 시간, 평가 보상): 생존이 목적이라 생존 시간이 우선
    steps, episode, updates, update_credit = 0, 0, 0, 0.0
    last_eval_bucket, last_save_bucket = 0, 0
    started = time.time()

    try:
        while not (episode >= args.episodes if args.episodes else steps >= args.steps):
            epsilon = np.maximum(args.eps_end, max(args.eps_end, 1.0 - steps / args.eps_decay_steps) * eps_scale)
            greedy = greedy_actions(online, obs)
            random_actions = np.random.randint(0, n_actions, size=num_envs)
            actions = np.where(np.random.random(num_envs) < epsilon, random_actions, greedy)

            results = envs.step(actions)
            new_obs = np.empty_like(obs)
            transitions: list[tuple] = []
            for env_id, (next_obs, reward, terminated, truncated, stats, after_obs) in enumerate(results):
                windows[env_id].append((obs[env_id], int(actions[env_id]), reward * args.reward_scale))
                store_nstep(transitions, windows[env_id], next_obs, terminated, terminated or truncated,
                            args.n_step, args.gamma)
                new_obs[env_id] = after_obs
                if stats is not None:
                    episode += 1
                    recent_returns.append(stats["return"])
                    recent_survival.append(stats["survival"])
                    mean_loss = float(np.mean(losses)) if losses else float("nan")
                    mean_q = float(np.mean(q_means)) if q_means else float("nan")
                    log.writerow([episode, steps + num_envs, env_id, stats["steps"], f"{stats['survival']:.1f}",
                                  f"{stats['return']:.4f}", stats["score"], stats["kills"],
                                  f"{epsilon[env_id]:.3f}", f"{mean_loss:.5f}", f"{mean_q:.4f}"])
                    log_file.flush()
                    if episode % args.log_every == 0:
                        speed = (steps + num_envs) / max(time.time() - started, 1e-9)
                        print(f"에피소드 {episode:5d} | step {steps + num_envs:8d} | "
                              f"최근 20회 평균 보상 {np.mean(recent_returns):8.2f} | "
                              f"평균 생존 {np.mean(recent_survival):5.0f}초 | ε {epsilon.mean():.2f} | "
                              f"loss {mean_loss:.4f} | Q {mean_q:.2f} | {speed:.0f} step/s")
            obs = new_obs
            steps += num_envs
            replay.add_batch(transitions)

            # 모인 step 수에 비례해 학습한다(환경이 많아도 학습 비율은 같다).
            if len(replay) >= args.learn_start:
                update_credit += num_envs / args.train_every
                beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * steps / PER_BETA_STEPS)
                while update_credit >= 1.0:
                    update_credit -= 1.0
                    loss_value, q_value = dqn_update(online, target, replay, optimizer, args.batch, beta)
                    soft_update(target, online, TAU)
                    losses.append(loss_value)
                    q_means.append(q_value)
                    updates += 1

            if steps // SAVE_EVERY_STEPS > last_save_bucket:
                last_save_bucket = steps // SAVE_EVERY_STEPS
                save_checkpoint(out_dir / "dqn02_last.pt", online, n_actions)

            if args.eval_every > 0 and episode // args.eval_every > last_eval_bucket:
                last_eval_bucket = episode // args.eval_every
                result = evaluate(online, main_env, [10_000 + i for i in range(args.eval_episodes)])
                summary = result["summary"]
                mean_reward = summary["total_reward"]["mean"]
                print(f"  [평가] 보상 {mean_reward:.2f} | 점수 {summary['score']['mean']:.0f}"
                      f" | 처치 {summary['kills']['mean']:.1f} | 생존 {summary['survival_seconds']['mean']:.0f}초")
                score_key = (summary["survival_seconds"]["mean"], mean_reward)
                if score_key > best_eval:
                    best_eval = score_key
                    save_checkpoint(out_dir / "dqn02_best.pt", online, n_actions)
                save_checkpoint(out_dir / "dqn02_last.pt", online, n_actions)
    finally:
        # 직접 중지(Ctrl+C·코랩 중지)해도 마지막 상태를 저장하고 총 판 수를 출력한다.
        save_checkpoint(out_dir / "dqn02_last.pt", online, n_actions)
        log_file.close()
        envs.close()
        main_env.close()
        print(f"학습 종료: 총 {episode} 에피소드, {steps} step, 학습 {updates}회 | 저장 완료: {out_dir.resolve()}")


def play(args: argparse.Namespace) -> None:
    """저장한 모델로 평가한다. RENDER_MODE = "human"으로 바꾸면 화면으로 볼 수 있다."""
    model = load_checkpoint(Path(args.play))
    env = make_env(RENDER_MODE)
    try:
        result = evaluate(model, env, [10_000 + i for i in range(args.eval_episodes)])
    finally:
        env.close()
    for episode in result["episodes"]:
        print(episode)
    print(result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Space Z DQN_02 학습(병렬 환경·구조화 네트워크·PER)")
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
    parser.add_argument("--reward-scale", type=float, default=REWARD_SCALE, help="학습용 보상 배율(1.0=원본). 출력되는 보상은 원본 그대로")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY, help="몇 에피소드마다 출력할지")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="몇 에피소드마다 평가할지(0이면 끔)")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="평가 에피소드 수")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="체크포인트·로그 저장 폴더(코랩은 Drive 경로 권장)")
    parser.add_argument("--play", type=str, default=None, help="학습 대신 저장한 .pt 모델을 평가한다")
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
