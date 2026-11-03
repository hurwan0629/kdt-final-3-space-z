# 연속 회전 PPO

200만 스텝 실험과 수정 후 독립 평가를 완료했다. 새 시드 20개에서 기존 모델 대비 평균 점수가 1,508 → 2,902점으로 올랐다. 상세 결과와 제약은 [RESULTS.md](RESULTS.md), 의사결정 과정은 [RESEARCH.md](RESEARCH.md)에 기록했다.

학습 중 진화 요청이 NumPy 정수라 무시되던 오류를 최종 화면 검증에서 발견했다. 디코더를 고친 뒤 가중치를 고정하고 별도의 새 시드 20개로 다시 평가했다. 학습 자체는 오류 수정 전 200만 스텝이며, 수정 후 추가 연구 학습은 없다. `initial_decoder_results/`와 접두사 없는 `heldout_*`는 이전 구현의 결과다. 최종 판정에는 **`corrected_*`** 파일을 사용한다.

원본 5139차원 관찰과 순수 PPO를 유지한다. 이동은 정지·8방향, 발사는 Bernoulli, 강화·진화는 마스킹된 범주 선택이다. 회전만 tanh Gaussian으로 바꾸며 -1~1은 최대 회전 속도에 대한 비율이다. 이동 세기는 연속이 아니다.

SB3가 지원하는 5칸 Box에 행동을 담지만, 이산 선택은 실제 Categorical/Bernoulli에서 뽑는다. 실수를 이산 구간으로 잘라 선택하지 않는다. `policy.py`가 결합 log probability와 tanh Jacobian을 계산하고, 공용 `actions/rotation_continuous_action_decoder.py`가 값을 검증해 PlayerControl로 변환한다.

## 실행 환경

프로젝트 루트의 PowerShell에서 기존 실험의 가상환경을 재사용한다. 기존 `.venv`와 공용 소스·기본 모델을 수정하지 않는다.

```powershell
$env:PYTHONPATH="$PWD/src"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYGAME_HIDE_SUPPORT_PROMPT='1'
$env:MPLCONFIGDIR="$PWD/src/space_z/rl/gpt_training_rotation_continuous/.cache/matplotlib"
$python = 'src/space_z/rl/gpt_training/.venv/Scripts/python.exe'
```

검증:

```powershell
& $python -B -m space_z.rl.gpt_training_rotation_continuous.check
```

모델 화면 재생:

```powershell
& $python -B -m space_z.rl.gpt_training_rotation_continuous.play --seed 505 --zoom 0.35
```

`selected_model.zip`이 확정된 모델이다. 기본 재생은 greedy이며 성장 행동도 정책이 선택한다. 다른 체크포인트는 `play` 뒤에 경로를 명시한다. 기존 `gpt_training.play`는 행동 형식이 달라 새 모델에 사용할 수 없다.

TensorBoard:

```powershell
& src/space_z/rl/gpt_training/.venv/Scripts/tensorboard.exe --logdir src/space_z/rl/gpt_training_rotation_continuous
```

## 기록과 재현

- `protocol.json`: 학습 예산·구간·seed·선발 조건을 사전에 고정한다.
- `experiments/*/experiment.json`: 가설, 실제 설정, 부모·소스 해시, 완료 상태, 추가 학습 스텝을 기록한다.
- `episodes.jsonl`: 학습 중 에피소드별 점수·생존·보상을 기록한다.
- `evaluations.jsonl`: 기본 게임에서 10만 스텝마다 측정한 실제 점수·생존·성장 선택을 기록한다.
- `tensorboard/`: 학습 손실·보상과 평가 평균·하위 25% 성적을 기록한다.
- `source_code.zip`: 해당 실험이 사용한 구현과 재사용 코드의 사본이다.
- `RESEARCH.md`: 실험 의도와 결과에 따른 결정을 기록한다.
- `decoder_correction_protocol.json`, `corrected_candidate_selection.json`: 수정 후 새 검증 시드와 모델을 평가 전에 고정한 기록이다.
- `corrected_heldout_comparison.json`, `.csv`, `.png`: 같은 시드로 비교한 최종 수치·에피소드·그림이다.
- `visual_corrected_seed*/`: 실제 렌더러로 재생한 화면과 행동·점수·기체 기록이다.
- `final_source_code.zip`, `checks/final_validation.json`: 최종 코드와 검증 근거다.

각 구간은 `--steps`를 2,000의 배수로 받아 4개 환경 × 500스텝 rollout을 완성한다. 50,000스텝마다 복구 모델을 저장한다. 구간 폴더에 `stop.request`를 만들면 진행 중인 100,000스텝 블록의 학습·평가·저장 후 종료한다. Ctrl+C는 현재 가중치를 `interrupted.zip`으로 보존한다. 프로세스를 강제 종료하면 마지막 저장 이후의 진행은 사라진다.

고정 seed, CPU 1 thread, 소스·버전·체크포인트 해시를 기록한다. 다른 기기·버전에서 학습 결과의 비트 단위 동일성은 보장하지 않는다. 평가에는 학습 당시와 같은 코드·정책 클래스가 필요하다.

체크포인트는 모델·optimizer를 보존하지만 진행 중인 게임 환경 상태까지 보존하지 않는다. 재개 학습은 지정한 seed의 새 에피소드로 시작한다. 재생 방식은 기본값과 `--stochastic` / `--no-stochastic`으로 구분하며, sampled 재생은 Torch seed를 `1000003 + game_seed`로 고정한다.
