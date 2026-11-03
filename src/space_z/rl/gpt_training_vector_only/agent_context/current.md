# 현재 작업

## 완료 상태

004_parallel_continuation을 완료했다. 사용자가 직접 실행을 맡겼고 추가 약 30만 스텝과 프로세스 상한 12개를 승인했다. 추가 307,200스텝, 누적 634,880스텝. 학습 PID 446188과 환경 워커 10개는 종료됐으며 활성 자체 학습은 없다.

## 범위와 유지 사항

기존 5139차원 VectorObservation만 정책에 입력한다. 모방학습·목표 선택·조준 보조 정보는 없다. policy.py와 보상·게임·PPO 설정을 유지했다. 4환경 DummyVecEnv에서 10환경 SubprocVecEnv로 바꿨고 rollout은 1,024 → 2,560이다. 환경·본체·실행기를 합쳐 실측 12 Python 프로세스. 기존 소스 30개와 모든 부모 체크포인트 해시가 일치한다. 변경은 gpt_training_vector_only/와 필수 graphify 출력 안에서 이뤄졌다.

## 사용자 선호

세부 수치는 TensorBoard에 충분히 기록하고 HTML에는 새 세부 지표를 넣지 않는다. 새 지표는 46개이며 개별 에피소드와 최근 20평균, 개발 평가, 새 시드 확인을 구분한다. 정확한 명중 횟수는 기존 이벤트에 없어 피해량과 피해 발생 스텝 수로 구분한다. TensorBoard notes/에 단위·집계 방식과 실험 목적을 적었다. 진행 중 작업을 지표 추가 때문에 급히 바꾸지 않는다.

## 실행 순서와 결과

1. 비교 evaluations/004_pretraining_comparison: 같은 새 20시드에서 65,536/196,608/327,680스텝 모델을 각각 평가했다. 평균 점수 55.5/5.75/13.25, 평균 생존 54.7425/69.095/56.960833초. 학습 없이 순차 실행했다.
2. 비교 결과를 보고 재선택하지 않고 사전에 고정한 003 final(327,680)에서 이어 학습했다. 가중치 153개·Adam 56상태의 정확한 복원을 확인했다.
3. 개발 20시드 평가 누적 스텝: 327,680/389,120/450,560/512,000/573,440/634,880.
4. 개발 평균 점수: 13.25/55/65.75/35/114.25/87.5. 선택 기준은 기존처럼 평균 점수 우선, 동점이면 생존이다.
5. 선택 모델은 573,440스텝: 개발 평균 114.25점, 생존 70.220833초. 마지막 모델 634,880스텝은 87.5점, 생존 78.793333초로 별도 보존했다.
6. 후보 고정 뒤 사전 선언한 740xxx 새 6시드에서 부모와 비교했다. 평균 점수 5 → 103.333333, 중앙값 5 → 105, 평균 생존 48.722222 → 55.033333초. 점수는 3승 1패 2무, 선택 모델 2판은 무득점. 이 결과로 재선택·추가 학습하지 않았다.

## 파일 위치

- 학습: ../experiments/004_parallel_continuation/
- 선택 모델: checkpoints/best.zip, 누적 573,440스텝
- 마지막 모델: checkpoints/final.zip, 누적 634,880스텝
- 세부 지표: logs/tensorboard/ 와 logs/episodes.jsonl, logs/training.jsonl
- 개별/최근20 지표: logs/tensorboard/episodes/
- 확인 평가: confirmation.json, logs/tensorboard/confirmation/{parent,selected}/
- 3모델 비교: ../evaluations/004_pretraining_comparison/
- 사전 계획: 004_plan.json. 사전 검사: 004_preflight.json.
- 검증: 004_comparison_verification.json, 004_verification.json.
- 전체 터미널 캡처: 004_console_capture.log.
- 기존 갤러리는 ../reports/index.html, 아키텍처는 ../reports/architecture.html. 새 세부 telemetry는 HTML 데이터에서 제외했다.

## 검증과 제약

학습 172에피소드, 개발 평가 120판, 확인 12판을 검증했다. 보상 원본×계수=기여량, 기여량 합=총보상이다. TensorBoard main 6,450값과 개별·최근20 15,824값이 원본과 일치했다. 최종 가중치가 마지막 평가 ZIP의 153텐서와 일치한다. PNG 6개가 정상이다.

PowerShell 실행 세션은 PNG 색상 프로필 경고를 NativeCommandError로 처리해 종료코드 1을 반환했다. Python을 명시적으로 0으로 종료하면서 stderr만 출력한 진단에서도 같은 셸 동작을 재현했다. 학습 상태·평가·저장물은 모두 완료됐고 Python traceback은 없다. 기존 이미지 파일은 수정하지 않았다.

평가는 최대 120 게임 초이고 학습 초기화는 1개다. 병렬화에 따른 rollout 변경도 함께 있었으므로 학습량 하나의 효과라고 분리해 단정하지 않는다. 전용 벽 회피나 장시간 생존 검증은 하지 않았다. 기본 play.py는 기존 002 경로를 유지한다. 모델이나 원본 기록을 덮어쓰지 않는다. 다음 학습은 별도 실험으로 기록한다.

## 명령

프로젝트 루트 PowerShell에서 PYTHONPATH를 src로 지정하고 gpt_training/.venv의 Python을 사용한다. graphify 갱신은 GRAPHIFY_MAX_WORKERS=1. compare 모듈은 --name으로 새 이름을 지정해야 기존 평가를 덮어쓰지 않는다.

## 005 진행 중 · 사용자 후속 승인

사용자는 결과가 괜찮으면 더 학습하고, 나쁘면 정리 후 많이 바꾸지 않은 분기를 진행하라고 지시했다. 004 best(573,440)에서 같은 설정으로 005_same_settings_continuation을 시작했다. 추가 307,200, 목표 880,640, PID 447492, exec session 84335. 환경 10 + 본체 + 실행기 실측 12개. 개발 730xxx 20시드, 새 확인 750xxx 6시드. 계획·다음 분기 기준은 005_plan.json(UTF-8 BOM 가능, Python에서는 utf-8-sig로 읽기)에 사전 기록했다. 현재 실험은 변경 없이 끝낸다. 확인 결과가 좋으면 다음 307,200스텝 구간도 같은 설정으로 이어가고, 나쁘거나 정체하면 004 best에서 학습률 하나만 낮춘 비교 분기를 진행한다. 이번 작업은 먼저 최대 두 구간을 확인한다. 학습 중 추가 Python 프로세스·graphify 실행 금지. stdout/stderr는 005_stdout.log / 005_stderr.log, Start-Process -Wait로 실제 exit code를 받도록 실행했다. 소스 코드는 아직 변경하지 않았다.

### 005 이후 작은 수정이 필요할 경우

005_plan.json의 사전 기준(개발 평균 125.675점 이상·생존 63.19875초 이상, 새 750xxx에서 부모 대비 평균 점수 유지 이상·생존 90% 이상·점수 승수>=패수)을 완료 후 확인한다. 통계적 유의성 기준이 아니라 다음 탐색 선택 기준이다. 통과하면 006은 005 best에서 같은 설정으로 추가 307,200. 실패/정체하면 006은 비교를 위해 004 best(573,440)에서 학습률만 0.0003→0.0001로 낮춰 추가 307,200. 보상·네트워크·게임·환경10·PPO 다른 항목은 유지. 개발730xxx20, 확인760xxx6. 학습률 변경이 필요하면 기존 experiment.py에 선택적 --learning-rate만 추가하고 부모 PPO 설정의 학습률 상속·검증을 함께 맞춘다. 부모 가중치/Adam 정확한 복원을 검증한 뒤 lr_schedule와 optimizer.param_groups의 lr만 명시적으로 바꾼다. 부모 config나 체크포인트는 변경하지 않는다. 현재 005가 실행 중일 때는 코드나 graphify를 건드리지 않는다. 005 프로세스/워커 종료 확인 후 작은 수정·검증·graphify(워커1)를 마치고 006을 시작한다.


## 009 실행 계획 · 2026-11-03

현재 인계의 최신 기준은 009_plan.json, 009_preflight.json 및 experiments/009_combined_settings_continuation/의 원본 기록이다. 위 005 진행 중 표기는 과거 기록이며, 006~008은 state.json 기준 완료됐다.

사용자는 007의 보상 강화와 008의 낮은 학습률을 유지하며 전체 프로세스 상한을 12개로 맞춘 다음 실험을 요청했고, 추가로 사망 보상을 줄여 학습하도록 승인했다. 사망 계수는 -3 → -1.5로 정했다. 출발은 008 best 828,416스텝(SHA-256 1e9a791b8141d345fb63ce88d0a742258a208395dcdbee781c855128acd9fb03), 추가 512,000, 목표 1,340,416이다.

관찰·정책·게임과 그 외 PPO 설정은 유지한다. 점수/천체 피해/적 피해 계수 0.03/0.015/0.015, 피격 -0.02, 학습률 0.0001이다. 환경 10개, rollout 2,560, 저장 간격 12,800, 개발 평가 간격 51,200이다. 기존 730xxx 개발 20시드와 새 790xxx 확인 20시드는 모두 최대 120초다. 기존 평균 점수 우선·동점 시 생존 선택 규칙을 유지하고 final을 별도로 남긴다. 확인 결과로 재선택하거나 추가 실험을 자동으로 시작하지 않는다.

실행기는 --max-owned-processes 12로 사전 검증하고 각 rollout 끝에 실제 자체 Python 프로세스 수를 다시 확인한다. TensorBoard resources/owned_python_processes와 state.json에도 기록한다. 실행 중 추가 Python 작업과 graphify를 동시에 돌리지 않는다. 기존 실험과 부모 체크포인트를 보존한다.

주의: 현재 저장 형식의 PPO.load는 learning_rate 속성 0.0003으로 lr_schedule을 다시 구성하지만 Adam에 저장된 lr은 0.0001이다. 009에서도 --learning-rate 0.0001을 명시해 복원 검증 후 덮어쓴다. 부모 정책 153텐서와 Adam 56상태가 정확히 복원됨을 사전 확인했다. 현재 environment.py/policy.py와 008 소스 스냅샷의 차이는 AST가 같은 서식 변경뿐이다.

실행 명령의 정확한 인자는 009_plan.json의 arguments에 있다. stdout/stderr는 009_stdout.log/009_stderr.log로 보관한다. 진행 및 종료 여부는 터미널 경고가 아닌 실험 state.json과 실제 프로세스 종료를 함께 확인한다.
