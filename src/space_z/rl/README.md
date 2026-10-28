# RL 사용 안내

학습 코드는 기존 환경을 생성해서 사용합니다. 에이전트 작업 규칙은 [AGENTS.md](AGENTS.md)를 먼저 확인하세요.

## 폴더 구조

```text
rl/
├── train.py                         # PPO 학습 실행·모델 저장
├── environment.py                   # TankEnv: 관찰·행동·보상과 게임 코어 연결
├── rewards.py                       # RewardCalculator: 전투 이벤트 → 보상
├── evaluate.py                      # PolicyEvaluator: 고정 시드로 모델 평가
├── resource_monitor.py              # ResourceMonitor: CPU·RAM·GPU 사용량 측정
├── actions/
│   ├── action_decoder.py            # 행동 해석기의 공통 인터페이스
│   ├── discrete_action_decoder.py   # 정수 행동 → PlayerControl
│   ├── continuous_action_decoder.py # 실수 배열 → PlayerControl
│   └── player_control.py            # 이동·회전·발사·강화·전직 요청값
├── observations/
│   ├── observation_builder.py       # 관찰 생성기의 공통 인터페이스
│   ├── vector_observation.py        # 개체·성장·지형의 숫자 벡터
│   ├── pixel_observation.py         # 이미지 스택 + 성장 벡터
│   └── progression_observation.py   # 두 관찰 방식에서 공유하는 성장 인코딩
├── __init__.py                      # 주요 클래스의 공개 import 경로
├── README.md                        # 이 안내
└── AGENTS.md                        # 코드 수정 범위와 작업 규칙
```

## 현재 PPO 실행

가상환경을 활성화하고 **프로젝트 루트**에서 실행합니다.

```powershell
python -m pip install -e ".[rl]" stable-baselines3
python -m space_z.rl.train --steps 100000
python -m space_z.rl.train --steps 100000 --pixels --seed 42
```

- 기본은 벡터 관찰, `--pixels`는 이미지·성장 벡터 관찰입니다. 두 경우 모두 연속 행동과 강화·전직 선택을 사용합니다.
- 현재 `train.py`는 `render_mode="human"`으로 게임 창을 띄우고 CPU에서 학습합니다. `--pixels`는 모델 입력을 고르는 옵션입니다.
- 모델은 **실행한 폴더 기준** `trained_models/ppo_vector.zip` 또는 `ppo_pixel.zip`에 저장됩니다. 같은 경로의 파일은 덮어씁니다.
- 256 환경 step을 모아 정책을 갱신하므로 실제 학습량은 256의 배수로 올림됩니다.

## 새 학습 코드 작성

사용자가 지정한 학습 파일에서 환경과 모델을 조립합니다. 예를 들어 이미지 입력의 PPO는 다음처럼 작성합니다.

```python
from stable_baselines3 import PPO
from space_z.config import GameConfig, RewardConfig
from space_z.rl import (
    TankEnv, PixelObservation, ContinuousActionDecoder, RewardCalculator,
)

env = TankEnv(
    game_config=GameConfig(),
    observation_builder=PixelObservation(),
    action_decoder=ContinuousActionDecoder(
        enable_stat_upgrades=True, enable_evolution=True,
    ),
    reward_calculator=RewardCalculator(RewardConfig()),
    render_mode=None,  # 창 없이 학습; 사람이 보려면 "human"
)
try:
    model = PPO("MultiInputPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("ppo_example")
finally:
    env.close()
```

| 바꾸려는 항목 | 학습 코드에서 선택할 값 |
|---|---|
| 벡터 관찰 | `VectorObservation()` + `MlpPolicy` |
| 이미지·성장 관찰 | `PixelObservation()` + `MultiInputPolicy` |
| DQN 행동 | `DiscreteActionDecoder(...)` |
| PPO 행동 | `ContinuousActionDecoder(...)` 또는 `DiscreteActionDecoder(...)` |
| 게임 규칙·적 수 | `GameConfig(...)` 객체 |
| 관찰 범위·이미지 크기 | 관찰 생성기에 전달할 `ObsConfig(...)` 객체 |
| 행동 반복·에피소드 제한 | `TankEnv`의 `env_config=EnvConfig(...)` |
| 보상 계수 | `RewardCalculator(RewardConfig(...))` |

두 행동 해석기의 이동은 정지·8방향의 고정 세기이며, 연속 해석기는 회전을 실수 세기로 받습니다. 강화·전직 옵션을 켜면 해당 선택도 행동에 포함됩니다. 불가능한 선택은 환경에서 적용하지 않습니다.

학습 라이브러리가 `env.reset()/step()`을 호출합니다. 직접 제어할 때는 `reset()`의 `(observation, info)`와 `step()`의 `(observation, reward, terminated, truncated, info)`를 사용합니다. 종료 후에는 `reset()`이 필요하며 코어의 `step()`이나 `pop_events()`를 학습 코드에서 별도로 호출하지 않습니다.

픽셀 관찰은 `{"image": 이미지 스택, "progression": 성장 벡터}`입니다. 실제 크기와 범위는 `env.observation_space`, 행동 형식은 `env.action_space`에서 확인하세요. 관찰 형식이나 행동 설정이 다른 모델을 그대로 불러오지 않습니다.

## 평가와 자원 측정

- `PolicyEvaluator(eval_env, action_selector).evaluate([42, 43])`: 학습과 분리한 환경에서 평가합니다. SB3 모델의 선택 함수는 `lambda obs: model.predict(obs, deterministic=True)[0]` 형태입니다.
- `ResourceMonitor(gpu_index=0)`: `start()`로 측정을 시작하고 `stop()`으로 요약을 받습니다. 평가기의 `resource_monitor=`로 연결할 수도 있습니다.
- 평가 환경과 측정기는 사용 후 각각 `close()`합니다. GPU 지표는 장치 전체 기준이며 지원되지 않으면 `None`입니다.

설정은 학습 코드에서 객체로 전달하세요. 기존 환경·게임·설정 파일을 변경하는 작업은 사용자가 직접 요청한 경우에만 수행합니다.
