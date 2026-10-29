"""DQN 학습 실행: python -m space_z.rl.DQN

학습 길이(에피소드·step)와 하이퍼파라미터는 아래 '하이퍼파라미터' 구간의 상수를 고치거나 --옵션으로 지정한다.

13_MultiModal/04_딥강화학습.ipynb의 DQN(ReplayBuffer·QNetwork·dqn_update·ε-greedy·target 복사)을
Space Z 환경(TankEnv, 벡터 관찰, 이산 행동)에 맞춰 옮긴 것이다.

추가 의존성: torch (코랩에는 기본 설치). 프로젝트 설치는 python -m pip install -e ".[rl]"
로컬에서 학습 화면을 보려면 아래 '화면 표시' 주석을 풀고, 배속은 RENDER_SPEED로 조절한다.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
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
# 학습 길이: TRAIN_EPISODES가 0보다 크면 그 에피소드 수에서 끝나고, 0이면 TRAIN_STEPS에서 끝난다.
TRAIN_EPISODES = 0  # 학습할 에피소드 수 (예: 500). 0이면 아래 step 기준
TRAIN_STEPS = 300000  # 학습할 환경 step 수 (TRAIN_EPISODES가 0일 때만 사용)
SEED = 42  # 난수 시드

LEARNING_RATE = 1e-4  # Adam 학습률
GAMMA = 0.99  # 할인율
N_STEP = 3  # N-step 반환: 몇 step 뒤까지의 보상을 합쳐서 학습할지 (1이면 일반 DQN)
BATCH_SIZE = 64  # 한 번의 경사 계산에 쓰는 경험 수
BUFFER_SIZE = 50000  # 리플레이 버퍼 크기 (5만 개 ≈ 1GB RAM)
LEARN_START = 2000  # 버퍼에 이 개수 이상 쌓이면 학습 시작
TRAIN_EVERY = 4  # 몇 step마다 한 번 학습할지
TARGET_EVERY = 500  # 몇 step마다 Online 가중치를 Target에 복사할지
DOUBLE_DQN = True  # Double DQN 사용 여부
REWARD_SCALE = 0.1  # 학습용 보상 배율(1.0=원본). 로그에 찍히는 보상은 원본 그대로

EPS_END = 0.05  # ε 최솟값 (탐험 확률의 하한)
EPS_DECAY_STEPS = 150_000  # ε이 1.0에서 EPS_END까지 줄어드는 step 수

LOG_EVERY = 5  # 몇 에피소드마다 학습 상황을 출력할지
EVAL_EVERY = 50  # 몇 에피소드마다 평가할지 (0이면 평가 끔)
EVAL_EPISODES = 3  # 평가할 때 돌리는 에피소드 수
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
# time_step이 0.01이면 3000 step을 끝까지 살 때 +30, 사망(death)은 -20.
shape_damage = 0.002  # 천체 피해 보상 계수: 피해량 × (점수/최대 체력)에 곱함
shape_destroy = 0.01  # 천체 파괴 보상 계수: 파괴한 천체 점수에 곱함
enemy_damage = 0.01  # 적 피해 보상 계수: 적에게 준 피해량에 곱함
enemy_kill = 2.0  # 적 한 대 처치 시 보상
damage_taken = -0.02  # 피격 보상 계수: 받은 피해량에 곱함 (음수면 페널티)
death = -20.0  # 플레이어 사망 시 보상
time_step = 0.01  # step마다 더해지는 값 (생존 보너스). 공통 기본값은 -0.0001
shot_cost = 0.01  # 실제 발사 탄환 1발당 차감할 보상 (0 이상)

# 화면 표시(로컬 전용). 코랩은 창이 없으므로 아래 두 줄은 주석 그대로 둔다.
# 화면을 켜면 매 step 그리기 비용 때문에 학습이 느려진다.
RENDER_MODE = None  # 학습: 화면 없이 최대 속도로 진행
# RENDER_MODE = "human"  # 학습 화면 켜기 (위의 RENDER_MODE = None 줄은 지운다)

# 배속(RENDER_MODE="human"일 때만 의미가 있다). 1.0=실시간(15 step/초), 4.0=4배속, 0=제한 없음.
RENDER_SPEED = 4.0


# --------------------------------------------------------------------------
# 재현성·장치
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
# 환경
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# DQN 구성 요소 (노트북의 ReplayBuffer·QNetwork·dqn_update)
# --------------------------------------------------------------------------
class ReplayBuffer:
    """경험 (s, a, N-step 보상, ns, terminated, 할인)을 원형으로 저장하고 무작위로 뽑는다.

    관찰이 5139차원이라 deque 대신 NumPy 배열을 미리 잡고 float16으로 저장해 메모리를 아낀다.
    관찰 값이 [-1, 1] 범위라 float16 정밀도로 충분하다.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.states = np.empty((capacity, obs_dim), dtype=np.float16)
        self.next_states = np.empty((capacity, obs_dim), dtype=np.float16)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)  # 다음 상태 가치에 곱할 γ^k (k=실제 합친 step 수)
        self.pos = 0  # 다음에 쓸 위치
        self.size = 0  # 현재 저장된 경험 수

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, terminated: bool, discount: float) -> None:
        """경험 하나를 저장한다. 가득 차면 가장 오래된 경험을 덮어쓴다."""
        self.states[self.pos] = s
        self.next_states[self.pos] = ns
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.terminated[self.pos] = float(terminated)
        self.discounts[self.pos] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """경험 batch_size개를 무작위로 뽑아 DEVICE의 텐서로 변환한다."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.actions[idx], device=DEVICE).unsqueeze(1),
            torch.as_tensor(self.rewards[idx], device=DEVICE).unsqueeze(1),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.terminated[idx], device=DEVICE).unsqueeze(1),
            torch.as_tensor(self.discounts[idx], device=DEVICE).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self.size


class QNetwork(nn.Module):
    """상태 벡터를 받아 행동별 Q값을 출력하는 Dueling MLP.

    공유 몸통에서 상태 가치 V(s)와 행동 이득 A(s, a)를 따로 계산해 Q = V + A - mean(A)로 합친다.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        advantage = self.advantage(z)
        return self.value(z) + advantage - advantage.mean(dim=1, keepdim=True)


def dqn_update(
    online: QNetwork,
    target: QNetwork,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int = 64,
    double: bool = False,
) -> tuple[float, float]:
    """리플레이 버퍼에서 배치를 뽑아 온라인 신경망을 한 번 갱신하고 (손실, 평균 Q)를 반환한다."""
    states, actions, rewards, next_states, terminated, discounts = buffer.sample(batch_size)
    chosen_q = online(states).gather(1, actions)  # 실제로 선택했던 행동의 Q값만 추출
    with torch.no_grad():
        if double:
            # Double DQN: 온라인으로 행동을 고르고 타깃으로 평가
            next_actions = online(next_states).argmax(dim=1, keepdim=True)
            next_q = target(next_states).gather(1, next_actions)
        else:
            next_q = target(next_states).max(dim=1, keepdim=True).values
        # rewards는 N-step 보상 합, discounts는 γ^k. 사망(terminated)이면 미래 가치를 제외하고
        # 시간 제한(truncated)은 종료로 보지 않는다.
        targets = rewards + discounts * (1 - terminated) * next_q
    loss = nn.functional.smooth_l1_loss(chosen_q, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.item()), float(chosen_q.mean().item())


def store_nstep(
    replay: ReplayBuffer,
    window: deque,
    next_state: np.ndarray,
    terminated: bool,
    episode_over: bool,
    n_step: int,
    gamma: float,
) -> None:
    """window의 (s, a, r) 앞쪽부터 N-step 경험으로 묶어 리플레이에 저장한다.

    window가 가득 차면 가장 오래된 하나를 저장한다. 판이 끝나면(episode_over) 남은 것도 모두 저장한다.
    남은 step 수 k만큼만 합치고 할인은 γ^k다. 시간 제한은 terminated=False라 다음 상태에서 이어 붙인다.
    """
    def store_oldest() -> None:
        ret = sum(gamma ** i * item[2] for i, item in enumerate(window))
        state, action = window[0][0], window[0][1]
        replay.add(state, action, ret, next_state, terminated, gamma ** len(window))
        window.popleft()

    if len(window) == n_step:
        store_oldest()
    if episode_over:
        while window:
            store_oldest()


BASE_ACTIONS = N_MOVES * N_ROTS * 2  # 이동 × 회전 × 발사 = 54개
STAT_CHOICES = len(STAT_UPGRADES) + 1  # 스탯 강화 번호 개수(0=선택 안 함)


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


def greedy_action(online: QNetwork, state: np.ndarray) -> int:
    """Q값이 가장 큰 행동 번호를 반환한다."""
    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(online(tensor).argmax(dim=1).item())


# --------------------------------------------------------------------------
# 학습·평가·저장
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, online: QNetwork, obs_dim: int, n_actions: int) -> None:
    """모델 가중치와 재생에 필요한 형식 정보를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": online.state_dict(), "obs_dim": obs_dim, "n_actions": n_actions,
         "action_mode": ACTION_MODE},
        path,
    )


def load_checkpoint(path: Path) -> QNetwork:
    """저장한 모델을 불러온다. 행동 공간 설정이 다르면 오류를 낸다."""
    data = torch.load(path, map_location=DEVICE)
    if data["action_mode"] != ACTION_MODE:
        raise ValueError(f"ACTION_MODE 설정이 다릅니다. 저장 당시 값: {data['action_mode']}")
    model = QNetwork(data["obs_dim"], data["n_actions"]).to(DEVICE)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


def evaluate(online: QNetwork, env: TankEnv, seeds: list[int]) -> dict:
    """고정 시드로 탐험 없이(argmax) 평가한다. 평가 중에는 학습하지 않는다."""
    online.eval()
    try:
        return PolicyEvaluator(env, lambda obs: to_env_action(greedy_action(online, obs), env)).evaluate(seeds)
    finally:
        online.train()


def train(args: argparse.Namespace) -> None:
    """DQN을 학습하고 체크포인트와 에피소드 로그(CSV)를 저장한다."""
    set_seed(args.seed)
    env = make_env(RENDER_MODE)
    eval_env = make_env(None) if args.eval_every > 0 else None  # 평가는 항상 화면 없이

    obs_dim = int(env.observation_space.shape[0])
    n_actions = n_dqn_actions(env)
    goal = f"{args.episodes} 에피소드" if args.episodes else f"{args.steps} step"
    print(f"장치={DEVICE} | 관찰={obs_dim}차원 | DQN 행동={n_actions}개 | ACTION_MODE={ACTION_MODE} | 목표={goal}")

    online = QNetwork(obs_dim, n_actions).to(DEVICE)
    target = QNetwork(obs_dim, n_actions).to(DEVICE)
    target.load_state_dict(online.state_dict())  # Online 가중치를 Target에 복사
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer, obs_dim)

    out_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d%H%M%S_dqn")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["episode", "steps", "ep_steps", "survival_sec", "return", "score", "kills", "epsilon", "loss", "mean_q"])

    returns: deque[float] = deque(maxlen=20)  # 최근 20 에피소드 보상
    losses: deque[float] = deque(maxlen=200)
    q_means: deque[float] = deque(maxlen=200)  # 최근 학습 배치의 평균 Q값(발산 여부 확인용)
    window: deque = deque()  # N-step 계산용 최근 (s, a, r)
    best_eval = (-float("inf"), -float("inf"))  # (평가 생존 시간, 평가 보상): 생존이 목적이라 생존 시간이 우선
    steps, episode, started = 0, 0, time.time()

    try:
        # --episodes를 주면 에피소드 수로, 아니면 --steps로 학습 길이를 정한다.
        while episode < args.episodes if args.episodes else steps < args.steps:
            state, info = env.reset(seed=args.seed + episode)
            total, terminated, truncated = 0.0, False, False
            ep_start_steps = steps
            window.clear()

            while not (terminated or truncated):
                # Epsilon-Greedy: 초반에는 탐험, 이후 점차 활용 비율 증가
                epsilon = max(args.eps_end, 1.0 - steps / args.eps_decay_steps)
                if random.random() < epsilon:
                    action = random.randrange(n_actions)
                else:
                    action = greedy_action(online, state)

                # 리플레이에는 DQN이 고른 번호를 저장하고, 환경에는 성장 선택을 끼운 번호를 보낸다.
                next_state, reward, terminated, truncated, info = env.step(to_env_action(action, env))
                window.append((state, action, reward * args.reward_scale))
                store_nstep(replay, window, next_state, terminated, terminated or truncated, args.n_step, args.gamma)
                state = next_state
                total += reward
                steps += 1

                if len(replay) >= args.learn_start and steps % args.train_every == 0:
                    loss_value, q_value = dqn_update(online, target, replay, optimizer, args.batch, args.double)
                    losses.append(loss_value)
                    q_means.append(q_value)
                if steps % args.target_every == 0:
                    target.load_state_dict(online.state_dict())
                if not args.episodes and steps >= args.steps:
                    break

            episode += 1
            returns.append(total)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            mean_q = float(np.mean(q_means)) if q_means else float("nan")
            survival = info["frames"] / env.game_config.fps  # 이번 판의 생존 시간(초)
            log.writerow([episode, steps, steps - ep_start_steps, f"{survival:.1f}", f"{total:.4f}", info["score"],
                          info["kills"], f"{epsilon:.3f}", f"{mean_loss:.5f}", f"{mean_q:.4f}"])
            log_file.flush()

            if episode % args.log_every == 0:
                speed = steps / max(time.time() - started, 1e-9)
                print(f"에피소드 {episode:4d} | step {steps:7d} | 최근 20회 평균 보상 {np.mean(returns):8.2f}"
                      f" | 생존 {survival:.0f}초 | 점수 {info['score']} | 처치 {info['kills']} | ε {epsilon:.2f}"
                      f" | loss {mean_loss:.4f} | Q {mean_q:.2f} | {speed:.0f} step/s")

            if eval_env is not None and episode % args.eval_every == 0:
                result = evaluate(online, eval_env, [10_000 + i for i in range(args.eval_episodes)])
                mean_reward = result["summary"]["total_reward"]["mean"]
                print(f"  [평가] 보상 {mean_reward:.2f} | 점수 {result['summary']['score']['mean']:.0f}"
                      f" | 처치 {result['summary']['kills']['mean']:.1f}"
                      f" | 생존 {result['summary']['survival_seconds']['mean']:.0f}초")
                score_key = (result["summary"]["survival_seconds"]["mean"], mean_reward)
                if score_key > best_eval:
                    best_eval = score_key
                    save_checkpoint(out_dir / "dqn_best.pt", online, obs_dim, n_actions)
                save_checkpoint(out_dir / "dqn_last.pt", online, obs_dim, n_actions)
    finally:
        save_checkpoint(out_dir / "dqn_last.pt", online, obs_dim, n_actions)
        log_file.close()
        env.close()
        if eval_env is not None:
            eval_env.close()
    print(f"학습 종료: 총 {episode} 에피소드, {steps} step | 저장 완료: {out_dir.resolve()}")


def play(args: argparse.Namespace) -> None:
    """저장한 모델로 평가한다. 로컬에서는 아래 주석을 풀어 화면으로 볼 수 있다."""
    model = load_checkpoint(Path(args.play))
    env = make_env(RENDER_MODE)  # 화면을 보려면 위의 RENDER_MODE = "human"을 켠다.
    try:
        result = PolicyEvaluator(env, lambda obs: to_env_action(greedy_action(model, obs), env)).evaluate(
            [10_000 + i for i in range(args.eval_episodes)])
    finally:
        env.close()
    for episode in result["episodes"]:
        print(episode)
    print(result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Space Z DQN 학습")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES,
                        help="학습할 에피소드 수. 0보다 크면 --steps 대신 이 값으로 끝낸다")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="학습할 환경 step 수(--episodes가 0일 때만 사용)")
    parser.add_argument("--seed", type=int, default=SEED, help="난수 시드")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam 학습률")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="할인율")
    parser.add_argument("--n-step", type=int, default=N_STEP, help="N-step 반환의 N(1이면 일반 DQN)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="배치 크기")
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE, help="리플레이 버퍼 크기(50k ≈ 1GB RAM)")
    parser.add_argument("--learn-start", type=int, default=LEARN_START, help="이 개수 이상 쌓이면 학습 시작")
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY, help="몇 step마다 한 번 학습할지")
    parser.add_argument("--target-every", type=int, default=TARGET_EVERY, help="몇 step마다 Target을 갱신할지")
    parser.add_argument("--eps-end", type=float, default=EPS_END, help="ε 최솟값")
    parser.add_argument("--eps-decay-steps", type=int, default=EPS_DECAY_STEPS, help="ε이 1.0에서 최솟값까지 줄어드는 step 수")
    parser.add_argument("--reward-scale", type=float, default=REWARD_SCALE, help="학습용 보상 배율(1.0=원본). 출력되는 보상은 원본 그대로")
    parser.add_argument("--double", action=argparse.BooleanOptionalAction, default=DOUBLE_DQN,
                        help="Double DQN 사용(끄려면 --no-double)")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY, help="몇 에피소드마다 출력할지")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="몇 에피소드마다 평가할지(0이면 끔)")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="평가 에피소드 수")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="체크포인트·로그 저장 폴더(코랩은 Drive 경로 권장)")
    parser.add_argument("--play", type=str, default=None, help="학습 대신 저장한 .pt 모델을 평가한다")
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0 or args.episodes < 0:
        parser.error("--steps는 양수, --episodes와 --seed는 0 이상이어야 합니다.")
    if args.n_step < 1:
        parser.error("--n-step은 1 이상이어야 합니다.")
    if args.reward_scale <= 0:
        parser.error("--reward-scale은 양수여야 합니다.")
    play(args) if args.play else train(args)


if __name__ == "__main__":
    main()
