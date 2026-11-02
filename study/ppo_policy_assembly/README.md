# 실제 SB3 PPO 조립 연습

이 폴더는 Space Z의 `RotationContinuousPolicy` / `RotationHybridDistribution`과 **같은 실제 라이브러리 API**를 사용한다. 기존 게임 코드나 학습 체크포인트는 변경하지 않는다.

가장 복잡한 게임 관측(5,139차원)과 5종 행동을 대신해, 학습 흐름을 보기 쉬운 관측 4차원과 행동 3종(이동·연속 회전·발사)으로만 줄였다. 의사코드가 아니라 PPO.learn()까지 실행하는 실제 구현이다. 게임 실험의 성능을 재현하려는 코드는 아니다.

## 실행

프로젝트 루트에서, 기존 GPT 강화학습 가상환경을 재사용:

```powershell
$python = "src/space_z/rl/gpt_training/.venv/Scripts/python.exe"
& $python study/ppo_policy_assembly/train.py
```

가상환경이 없다면 `torch`, `gymnasium`, `stable-baselines3`, `numpy`가 설치된 Python 환경에서 실행한다. 실습에는 원본 Space Z 자산이 필요하지 않다.

## 파일을 읽는 순서

1. `train.py`: `DummyVecEnv` + `Monitor` 생성, `PPO(SmallHybridPolicy, env, ...)`, `learn()`.
2. `policy.py`의 `SmallHybridPolicy.__init__` 및 `_build`: SB3 부모 클래스 초기화 중에 네트워크·분포 객체·Optimizer 생성.
3. `policy.py`의 `forward`: 관측을 처리하고 3종 행동, 가치, 로그 확률을 반환.
4. `policy.py`의 `SmallHybridDistribution`: Categorical / Normal / Bernoulli를 결합하고 tanh의 확률밀도를 보정.
5. `policy.py`의 `evaluate_actions`: 같은 관측과 **이미 실행한 행동**에 대해 현재 정책의 `log_prob`과 가치를 계산.
6. `environment.py`: `step(action)`에서 유효 범위를 확인하고, 실제 상태 전이·보상·종료 여부 반환.

## 객체 소유 구조

```text
model: PPO
 ├─ env: DummyVecEnv
 │    ├─ Monitor(TargetEnv)
 │    └─ Monitor(TargetEnv)
 └─ model.policy: SmallHybridPolicy(ActorCriticPolicy)
      ├─ features_extractor: SmallFeaturesExtractor(BaseFeaturesExtractor)
      │    └─ Linear(4,16) → Linear(16,8)
      ├─ mlp_extractor: Actor MLP(8→16), Critic MLP(8→16)
      ├─ action_net: Linear(16,5)
      ├─ aim_net: Linear(4,16) → Linear(16,2)
      ├─ value_net: Linear(16,1)
      ├─ log_std: nn.Parameter(shape=(1,))
      ├─ action_dist: SmallHybridDistribution(Distribution)
      └─ optimizer
```

`action_net`의 다섯 수치는 [이동 Logits 3, 회전 평균 1, 발사 Logit 1]이다. `aim_net`의 두 수치가 기본 회전·발사를 **교체**하며 Residual 방식은 아니다.

## Schedule은 PyTorch와 어떤 관계인가?

- **SB3**의 `learning_rate`는 숫자 또는 `progress_remaining -> lr` 함수다. 실습은 `learning_rate()` 함수를 전달한다.
- `SmallHybridPolicy._build(lr_schedule)`에서 Optimizer의 **초기 학습률**을 `lr_schedule(1)`로 정한다.
- `PPO.train()`은 각 학습 단계에서 SB3가 스케줄을 평가해 Optimizer의 파라미터 그룹 LR을 갱신한다.
- 이것은 `torch.optim.lr_scheduler.StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` 객체를 따로 만들어 `scheduler.step()` 하는 방식과 **다르다**. 둘을 아무 생각 없이 함께 적용하면 학습률 갱신이 충돌할 수 있다.
- `learning_rate=3e-4`처럼 숫자를 전달하면 고정 학습률이다.

## 차원과 호출 위치

```text
PPO(...) → Policy.__init__ / _build (초기화 한 번)
PPO.learn() → collect_rollouts()
  obs (2,4) → features (2,8)
  → actor_latent (2,16) / critic_latent (2,16)
  → action_net (2,5), aim_net (2,2)
  → distribution.sample() → actions (2,3)
  → value_net → values (2,1)
  → distribution.log_prob(actions) → log_probs (2,)
  → env.step(actions)
  → RolloutBuffer 저장
PPO.train() → policy.evaluate_actions(stored_obs, stored_actions)
  → 현 정책의 log_prob 재계산
  → PPO loss / backward / optimizer.step()
```

## 이 예제에서 생략한 요소

실제 Space Z는 5,139차원 개체·CNN 특징 추출, 22개 행동분포 파라미터, 강화·전직 마스킹, 4개 서브프로세스 환경, 상세 JSONL 기록과 체크포인트를 사용한다. 이 실습은 **그 프레임워크 연결을 이해하는 최소 부분**에만 집중한다.

원본 소스: `src/space_z/rl/gpt_training_rotation_continuous/policy.py`, `train.py`, `src/space_z/rl/actions/rotation_continuous_action_decoder.py`.
