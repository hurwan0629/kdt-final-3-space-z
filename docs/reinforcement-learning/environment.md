# 강화학습 환경

`TankEnv`는 공통 `GameCore`에 관찰 생성기·행동 해석기·보상 계산기를 연결하는 Gymnasium 환경입니다. 각 객체는 호출자가 만들어 전달합니다. `GameConfig()`는 성장 활성화·무제한 탄약이며, 환경에서 이 설정을 강제로 바꾸지 않습니다. `train.py`의 `ModelTrainer`는 아직 골격입니다.

## 환경 만들기

먼저 `python -m pip install -e ".[rl]"`로 선택 의존성을 설치합니다.

```python
from space_z.config import GameConfig, RewardConfig
from space_z.rl import TankEnv, VectorObservation, DiscreteActionDecoder, RewardCalculator

env = TankEnv(
    GameConfig(),
    VectorObservation(),
    DiscreteActionDecoder(enable_stat_upgrades=True, enable_evolution=True),
    RewardCalculator(RewardConfig()),
)
try:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(0)
finally:
    env.close()
```

`VectorObservation()`을 `PixelObservation()`으로 바꾸거나 `DiscreteActionDecoder()`를 `ContinuousActionDecoder()`로 바꿀 수 있습니다. 환경의 관찰·행동 공간도 해당 객체가 정의합니다. `reset(seed=..., options=None)`은 `(observation, info)`, `step(action)`은 `(observation, reward, terminated, truncated, info)`를 반환합니다. 현재 별도 reset 옵션은 지원하지 않습니다.

## 한 step의 순서

1. 행동을 `PlayerControl`로 해석합니다.
2. 스탯 강화 → 전직·숙련 요청을 각각 한 번만 적용합니다. 포인트·단계·선택지 검증은 `PlayerProgression`이 맡으며 불가능한 선택은 적용하지 않습니다.
3. 같은 이동·회전·발사 요청으로 `GameCore.step()`을 기본 4회 진행합니다. 중간에 사망하면 멈춥니다.
4. 코어가 누적한 전투 이벤트를 `pop_events()`로 한 번 꺼내 보상을 계산하고 같은 이벤트의 복사본을 info에 넣습니다. 보상 계산기는 이벤트를 소비하거나 수정하지 않습니다.
5. 환경 step 수를 올리고 다음 관찰과 종료 여부를 반환합니다. 사망은 `terminated`, 사망하지 않고 기본 3000 step에 도달하면 `truncated`입니다.

생성 직후와 에피소드 종료 후에는 `reset()`이 필요합니다. `reset()`은 관찰 스택과 화면 연출도 비웁니다. `close()`는 관찰 생성기의 자원과 환경이 만든 창을 정리하며 반복 호출해도 안전합니다. 닫은 환경은 재사용하지 않습니다.

## 벡터 관찰

기본 **45×29타일**의 플레이어 중심 사각형에서 관찰합니다. 타일이 48px이면 2160×1392 월드 px입니다. 범위와 겹치는 살아 있는 적·천체·적 탄환만 가까운 순서로 넣고 나머지 슬롯은 0으로 채웁니다. 각 개체 슬롯의 첫 값은 존재 표시입니다. 벽에 가려진 개체도 범위 안이면 포함합니다.

| 순서 | 내용 | 기본 길이 |
|---|---|---:|
| 자기 | 속도2·체력비율·sin/cos 방향2·발사 대기·반지름·탄약 제한 여부·탄약비율·재장전 | 10 |
| 성장 | 기체 원핫21·레벨/경험치/포인트/특수보상 수4·스탯 단계5·강화 가능5·전직 슬롯4×22 | 123 |
| 적 | 존재·상대위치2·속도2·체력·방향2·반지름·기체 원핫21·엘리트 | 9×31 |
| 천체 | 존재·상대위치2·속도2·체력·반지름·종류 원핫3 | 30×10 |
| 적 탄환 | 존재·상대위치2·속도2·반지름·종류 원핫2 | 64×8 |
| 레이 | 지정한 방향에서 벽까지의 거리 | 기본 0 |
| 지형 | 벽·중력장·파괴 가능한 벽의 남은 체력 비율 | 3×29×45 |
| 합계 | float32 1차원 배열 | **5139** |

위치에는 절대 월드 좌표 대신 플레이어와의 상대 좌표를 사용합니다. 위치는 관찰 반폭·반높이, 선체·천체 속도는 기본 최대 속도, 탄환 속도는 기본 탄속, 반지름은 타일 길이로 나눕니다. 최종 벡터는 [-1, 1]로 제한합니다. 체력·발사 대기·탄약·재장전은 비율이며, 성장의 레벨·포인트·특수보상 개수는 `x/(x+1)`로 정규화합니다.

지형은 플레이어가 속한 타일을 가운데 놓으며 맵 밖은 벽입니다. 픽셀 카메라 중심과는 최대 반 타일 차이가 있습니다. `ObsConfig`로 슬롯 수·범위·추가 지형 채널·레이 수를 선택합니다. 채널은 실제 맵의 지형 기능이 꺼져도 관찰 설정에서 켜져 있으면 자리를 유지하고 0을 기록합니다. `VectorObservation.slices`에서 각 구간을 찾을 수 있습니다. 이전 241차원 모델과 호환되지 않습니다.

## 픽셀 관찰

`PixelObservation`은 같은 월드 범위를 창 없는 렌더러로 그려 다음 딕셔너리를 반환합니다.

- `image`: 기본 `(4, 116, 180)`, uint8 [0, 255]. 최근 네 관찰의 흑백 이미지이며 첫 관찰은 같은 이미지로 네 장을 채웁니다.
- `progression`: 벡터 관찰과 같은 `(123,)` float32 성장 벡터.

이미지에는 월드·개체·체력바·전투 연출이 들어가며 미니맵·메뉴·HUD는 제외합니다. 이미지 스택은 환경 관찰마다 갱신되며, 같은 코어 프레임을 다시 읽어도 밀리지 않습니다. 이미지 해상도는 `ObsConfig.image_width/image_height`, 누적 장수는 `frame_stack`으로 정합니다. 사람 창 크기와 줌은 관찰에 영향을 주지 않습니다. 모델에서 이미지 CNN 특징과 성장 벡터를 합쳐 사용합니다.

## 행동

두 해석기 모두 **정지·8방향 이동의 고정 가속 입력**을 사용합니다. 감속·가속·이동 속도는 코어의 마찰과 현재 능력치가 결정합니다.

| 구현체 | 기본 행동 | 성장 선택을 모두 켰을 때 |
|---|---|---|
| `DiscreteActionDecoder` | 이동9×회전3×발사2 = 54개 | 스탯6×전직5를 더해 1620개 |
| `ContinuousActionDecoder` | [-1, 1] 실수 배열 `[이동선택, 회전, 발사]` | `[이동선택, 회전, 발사, 스탯선택, 전직선택]` |

연속 해석기의 이동 선택은 9개 같은 폭 구간으로 나눕니다. 회전은 `입력×turn_speed` rad/frame이며 발사는 0 초과일 때 요청합니다. 스탯·전직도 같은 폭 구간으로 나누고 첫 구간은 요청 없음입니다. 배열 크기·유한성을 검증하고 [-1, 1]로 clipping합니다. 두 성장 옵션 중 하나만 켜면 그 항목만 붙습니다. 스탯 순서는 `health, damage, range, rapid, speed`, 전직은 현재 열린 선택지의 순서입니다.

코어의 정수 행동은 기존 27개 형식을 유지하고, 환경이 공통 제어값을 코어 입력으로 바꿉니다. 소수 회전은 코어의 선택 인자 `rotation_input`으로 전달하며 마우스 `aim_position`이 있으면 좌표 조준을 우선합니다.

## 보상과 info

`RewardCalculator.calculate(events)`는 피해·파괴·처치·피격·사망 보상을 합하고 실제 플레이어 발사 탄환 수×`RewardConfig.shot_cost`를 차감합니다. 현재 기본 발사 비용은 발당 0.01입니다. 발사 요청만 하고 쿨다운 등으로 발사하지 못하면 비용이 없습니다. 점사·복수 포신도 실제 탄환 생성 프레임에 계산합니다. `time_step`은 코어 프레임마다가 아니라 환경 step마다 한 번 적용합니다.

`info`는 관찰 벡터가 아닙니다. 점수·처치·맵·frames·steps·events·성장·무장 상태를 기록용으로 제공합니다. reset 직후 events는 None입니다. 기존 키를 유지하며 현재 전직 선택지를 `evolution_choices`로 제공합니다. 정확한 키는 `TankEnv._info()`를 확인하세요.

`render_mode="human"`은 사람용 창을 따로 출력하고 속도를 제한합니다. 기본 None은 속도 제한과 사람 창이 없습니다. 점수 팝업용 이벤트는 전투 이벤트와 분리하며 환경 step 끝에 화면으로 전달하거나 버립니다.

## 평가와 자원 측정

```python
from space_z.rl import PolicyEvaluator, ResourceMonitor

monitor = ResourceMonitor(sample_interval_seconds=0.5, gpu_index=0)
try:
    # env는 위 예시처럼 새로 생성한 평가 전용 환경이다.
    evaluator = PolicyEvaluator(env, lambda obs: env.action_space.sample(), monitor)
    result = evaluator.evaluate([10, 20, 30])
finally:
    monitor.close()
    env.close()
```

실제 비교에서는 무작위 함수 대신 학습을 하지 않고 행동을 고르는 모델 함수를 전달합니다. 같은 시드 목록·게임 설정·관찰 범위·보상 조건을 사용합니다. `episodes`에 에피소드별 보상 합계·점수·처치·생존 초·환경 step·게임 프레임·종료 사유를, `summary`에 평균·모집단 표준편차를 반환합니다. 생존 초는 게임 프레임/fps이며 실제 소요 시간과 구분합니다. 평가기는 호출자가 전달한 환경을 자동으로 닫지 않습니다.

`resources`에는 실제 경과 초, 샘플 수, 프로세스 평균 CPU 사용률, 최고 RAM(RSS 바이트), GPU 장치 전체 평균 사용률·최고 사용 메모리(바이트)가 들어갑니다. CPU는 여러 코어를 쓰면 100%를 넘을 수 있습니다. GPU는 NVIDIA NVML을 사용하며 장치가 없거나 지원하지 않는 지표는 None입니다. `gpu_index=None`이면 GPU 측정을 생략합니다. 메모리 최댓값은 샘플에서 관측한 최댓값이며 샘플 사이의 짧은 피크를 보장하지 않습니다. GPU 값은 다른 프로그램의 사용량도 포함하며 FLOPs는 측정하지 않습니다.

구현: [RL 패키지](../../src/space_z/rl/) · 설정: [[development/configuration|관찰·보상·환경 설정]] · [[reinforcement-learning/README|강화학습 목차]]
