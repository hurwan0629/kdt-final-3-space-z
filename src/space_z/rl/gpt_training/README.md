# GPT 학습 실험

공유 게임·RL 코드는 수정하지 않는다. `TankEnv`, `ObservationBuilder`, `ActionDecoder`,
`RewardCalculator`를 사용하고 이 폴더에서만 관찰·행동·보상을 조립한다.

현재 기본 모델은 기존 018을 유지한다. 016의 추가 학습 가중치에 2프레임 행동 주기를 적용한 순수 PPO다.
이번에는 PPO 464,896 step을 더 실행하고 체크포인트 평균까지 비교했다.
고득점 실험 후보는 새 20시드에서 최종 평균 1,591.25→2,057.75, 생존 185.72→235.43초였지만,
120초 점수 p25가 865→650으로 낮아져 기본 모델을 교체하지 않았다.
[최근 결과·실험 후보 실행 방법](RESULTS_PHASE3.md), [전체 추가 실험 기록](RESEARCH_PHASE3.md)을 참고한다.
실험 후보는 PPO로 학습한 두 부모의 50% 가중치 평균이며, 평균 과정의 추가 PPO 학습·모방학습은 0이다.
기존 018의 선택 근거는 [이전 결과](RESULTS_PHASE2.md), [이전 기록](RESEARCH_PHASE2.md)에 있다.
그 이전 개선 단계는 [최초 결과](RESULTS.md), [최초 기록](RESEARCH.md)에 보존했다.

## 점수 개선 실험 진입점

현재 개선 실험은 `improve.py`를 사용한다. 원본 5,139개 입력을 개체별 신경망과 지형 CNN으로
처리하고, 가능한 강화·진화 선택만 남긴다. 시연 데이터·모방 손실은 사용하지 않는다.
최근 실험 목적·조건·실패·다음 결정은 `RESEARCH_PHASE3.md`에 기록한다.

- `experiment.json`: 가설, 전체 설정, 보상 계수, 버전, 부모 모델·소스 해시.
- `episodes.jsonl`: 실제 점수·처치·생존과 보상 항목별 누적값.
- `evaluations.jsonl`: 같은 기본 게임에서 일정 학습량마다 측정한 시드별 점수·생존·처치와 평가 모델 해시.
- `tensorboard/`: PPO 지표, 게임 지표, 보상 항목, 기본 게임 평가 곡선.
- `checkpoints/best.zip`: 실험의 선택 목표와 중간 검증 시드 기준 후보. 넓은 검증의 최종 선택과 다를 수 있다. `latest.zip`은 최근 학습 결과다.
- `checkpoints/evaluated_*.zip`: 014부터 각 평가에 사용한 정확한 가중치. `best.zip`과 해시로 대조할 수 있다.

`score_reward.py`가 개선 실험의 보상을, `networks.py`가 특징 추출과 성장 마스크를 담당한다.
현재 정책은 개체 인코더·지형 CNN의 이동/성장 출력과 별도의 조준 신경망을 사용한다.
조준 입력에는 관찰 격자로 가시성을 근사한 가까운 개체의 상대 위치·속도 등을 정렬한다.
이는 입력 전처리이며, 실제 회전 세기와 발사 여부도 PPO로 학습한 신경망이 결정한다.
`train.py`는 아래 기존 실험을 재현하기 위해 보존한 진입점이다.
새 실험 이름과 가설을 지정하면 기존 결과를 덮어쓰지 않는다.

```powershell
& $py -B -m space_z.rl.gpt_training.improve --name my_full_experiment --stage full --reward-profile score_v6 --steps 262144 --num-envs 6 --hypothesis "기본 게임 점수 개선 가설"
```

`$py`와 환경 변수 설정은 아래 실행 절을 참고한다. 학습 종료 후 TensorBoard를 열 수 있다.

```powershell
& src/space_z/rl/gpt_training/.venv/Scripts/tensorboard.exe --logdir src/space_z/rl/gpt_training/experiments
```

새 실험 디렉터리에 `stop.request` 파일과 중단 이유를 쓰면 다음 검증 경계에서 저장 후 종료한다.
기본 게임 장면은 `capture.py`로 PNG와 `frames.json`에 기록할 수 있다.

## 기존 train 진입점: 순수 PPO

`train`의 기본 `--training-method pure_ppo`는 무작위 초기화한 정책을 PPO 보상으로만 학습한다.
시연 데이터·모방 손실·교사 평가를 사용하지 않는다. 이어 학습도 순수 PPO 이력이 있는 모델만 허용한다.

- 공유 `VectorObservation` 5,139개를 그대로 사용한다. 추천 이동·예측 조준점은 계산하지 않는다.
- `MultiDiscrete([9, 9, 2, 6, 5])`: 이동·회전 세기·발사·능력치 강화·진화를 정책이 선택한다.
- 성장 포인트가 없거나 현재 불가능한 선택은 기존 게임 규칙대로 적용되지 않는다. 자동 성장 규칙은 호출하지 않는다.
- 보상은 아래 기존 학습 보상을 유지한다. 보상 설계와 쉬운 환경에서 시작하는 커리큘럼은 모방학습이 아니다.
- 기존 혼합 모델과 관찰·행동 공간이 달라 가중치를 재사용하지 않는다.

## 이전 혼합 학습 설계 (001–003, 비교용 보존)

- 현재 상태에서 계산한 전술 관찰 164개: 자기 상태·성장·가시 목표의 예상 조준 오차·추천 이동 점수·벽 레이·탄환 위협.
- 이동 9개, 회전 세기 9개, 발사 여부를 각각 고르는 `MultiDiscrete`. 포신 정밀 조준이 가능하다.
- 강화·진화는 관찰의 선택 가능 정보에 기반한 고정 규칙으로 자동 처리한다. 평가에서도 같은 규칙을 사용한다.
- 규칙 정책의 행동을 모방한 뒤 PPO로 보정한다. 따라서 순수 PPO로 처음부터 발견한 전략과는 구분한다.
- 학습 보상은 게임 점수 증가/100, 피해 중간 보상, 체력 비율 피격 페널티, 사망 페널티, 작은 생존 보상이다.
- 원래 게임 점수·처치·생존 시간은 학습 보상과 별도로 기록한다.
- `harvest`: 적·내부 벽 없는 천체 수집, `combat`: 적 3기·내부 벽 없음, `full`: 수정 없는 기본 GameConfig.
- 자동 평가는 항상 `full`에서 수행한다. 시간 제한 생존은 완전 생존으로 단정하지 않는다.
- 현재 두 학습 진입점 모두 환경 worker를 최대 6개로 제한한다. Windows 실행기·조정 프로세스까지 총 8개다.
  평가는 조정 프로세스에서 순차 실행한다. TensorBoard 서버는 학습 종료 후 실행한다.

## 실행

프로젝트 루트의 PowerShell에서 다음 환경 변수를 먼저 설정한다.

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = "src/space_z/rl/gpt_training/.venv/Scripts/python.exe"
```

이 폴더 안의 전용 가상 환경을 사용한다. 다른 가상 환경이나 의존성 파일은 변경하지 않는다.
정확한 설치 버전은 `environment.lock.txt`를 참고한다.

```powershell
& $py -B -m space_z.rl.gpt_training.check
& $py -B -m space_z.rl.gpt_training.train --name pure_ppo_new --stage harvest --steps 131072 --num-envs 6
```

다음 단계로 이어 학습할 때는 새 실험 이름과 체크포인트를 명시한다.

```powershell
& $py -B -m space_z.rl.gpt_training.train --name pure_ppo_full_next --stage full --steps 131072 --num-envs 6 --resume src/space_z/rl/gpt_training/experiments/004_pure_ppo_harvest/checkpoints/final.zip
```

```powershell
& $py -B -m space_z.rl.gpt_training.play src/space_z/rl/gpt_training/experiments/004_pure_ppo_harvest/checkpoints/final.zip --seed 404
& src/space_z/rl/gpt_training/.venv/Scripts/tensorboard.exe --logdir src/space_z/rl/gpt_training/experiments
```

순수 PPO의 재생 기본값은 `pure_ppo_model.zip`이다. 이전 혼합 모델 `best_model.zip`과
그 선택 기록은 보존한다. 체크포인트의 학습 방식에 맞춰 재생·평가 환경을 선택한다.
`pure_ppo_model.zip`은 `018_reaction_calibration/checkpoints/candidate.zip`의 정확한 사본이다.
신경망 가중치는 `016_full_score_focus/checkpoints/evaluated_001474560.zip`과 같고,
재생·평가의 행동 주기 2프레임을 모델에 보존했다. 선택 모델은 1,474,560 step이다.
018 선택 단계에서 530,432 step, 이번 단계에서 464,896 step을 추가 실험했다.
006 이후 점수 개선 연구의 전체 PPO 실행량은 3,188,736 step이다.
새 분기의 부모 모델은 1,808,384 step까지 학습했으나 초반 안정성 기준에 미달해 기본 모델로 채택하지 않았다.
고득점 실험 후보는 `022_checkpoint_blend/checkpoints/blend_050.zip`이며, 저장된 부모 학습량과 평균 과정의 추가 학습량 0을 구분한다.
이전 014 모델은 원래 체크포인트에, 이전 공개 메타데이터는 `pure_ppo_model_before_phase2.json`에 보존했다.
이전 132,608 step 모델은 `005_pure_ppo_full/checkpoints/best.zip`에,
이전 기본 모델의 메타데이터는 `pure_ppo_model_before_improvement.json`에 보존했다.
현재 출처·해시·비교 조건과 한계는 `pure_ppo_model.json` 및 `phase2_candidate_selection.json`을 참고한다.

```powershell
& $py -B -m space_z.rl.gpt_training.play --seed 505 --max-seconds 400
```

`--max-seconds`는 게임 시간이다. `--max-steps`도 계속 지원하며 두 옵션은 함께 지정하지 않는다.
현재 모델에서 6,000 step은 200초다. 400초 재생에는 `--max-seconds 400`을 권장한다.
모델별 행동 주기를 재생·평가·화면 저장에서 자동으로 읽는다. 기본 물리 FPS는 60으로 유지한다.

현재 모델에서 이동·성장 학습을 이어가면서 학습된 조준 신경망을 보존하는 예시다.
새 실험 이름을 쓰며, `--steps`는 추가 학습량이다.

```powershell
& $py -B -m space_z.rl.gpt_training.improve --name full_body_next --stage full --reward-profile score_v6 --selection-objective early_stability --frame-skip 2 --aim-residual --independent-aim --ignore-aim-health --freeze-aim --learning-rate 0.0001 --entropy 0.004 --steps 262144 --num-envs 6 --episode-steps 6000 --eval-steps 6000 --resume src/space_z/rl/gpt_training/pure_ppo_model.zip --hypothesis "새 가설을 여기에 기록"
```

이는 새로운 학습 실험이다. 선택 모델은 4프레임 환경에서 학습한 뒤 재생 주기를 2프레임으로 조정했다.
추가 학습의 `--frame-skip`은 명시적으로 기록하며 그 값으로 새 체크포인트의 재생 주기도 저장한다.
할인율은 행동 step 기준이므로 학습 주기를 바꾸면 물리 시간에 대한 할인 범위도 달라진다.

학습·선택에 쓰지 않은 시드로 긴 평가를 별도 실행할 수 있다. 출력 파일은 새 경로여야 한다.

```powershell
& $py -B -m space_z.rl.gpt_training.evaluate src/space_z/rl/gpt_training/experiments/004_pure_ppo_harvest/checkpoints/final.zip --seeds 404 505 606 --max-steps 6000 --output src/space_z/rl/gpt_training/pure_ppo_heldout_new.json
```

## 기록과 복원

각 `experiments/<name>/`에는 전체 조건·버전·상태인 `experiment.json`, 학습 에피소드
`episodes.jsonl`, 고정 시드 평가 `evaluations.jsonl`, TensorBoard, 소스 사본,
체크포인트가 남는다. 모방 데이터는 이전 혼합 방식에서만 생성된다. 비교 결론과 다음 실험의 이유는 `EXPERIMENTS.md`에 기록한다.

과거 실험을 재현할 때만 `--training-method imitation_ppo`를 명시한다.

실험 폴더는 덮어쓰지 않는다. Ctrl+C 시 `interrupted.zip`을 저장하지만 게임 중간 상태와
진행 중 rollout은 복원하지 않는다. 새 실행의 `--steps`는 추가 학습량이며 rollout 단위로 초과할 수 있다.
기존 `training_hurwan` 체크포인트와는 관찰·행동 형식이 달라 호환되지 않는다.

`improve`의 `best.zip`은 평균 게임 점수, 생존 시간, 제한 시간 도달 비율 순서로 비교해 보존한다.
이전 `train` 진입점은 제한 시간 도달 비율을 먼저 비교한다. 선택 기준은 각 실행의 기록에서 확인한다.
평가 시드에 대한 선택이므로 별도 시드 평가를 일반화 확인에 사용한다.
공유 소스와 맵은 `shared_source.zip`에 읽기 전용 사본을 보관했고 파일별 해시는
`shared_source_manifest.json`에 기록했다. 의존성·캐시도 이 폴더 안에 둔다.
