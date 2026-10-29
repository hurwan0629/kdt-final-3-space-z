# 전체 성장 강화학습과 검증

main의 `GameConfig`, 관찰 생성기, 행동 해석기, 성장 규칙을 그대로 사용한다.
레벨 1에서 시작하며 스탯 5종, 10/20/30레벨 전직, 이후 숙련을 포함한다.
기존 전투 전용 실행과 체크포인트는 `full_growth=False` 기본값으로 계속 읽을 수 있다.

## 실행

현재 PC에서는 `C:\wooki\game\강화학습_전체성장.pyw`를 더블 클릭하거나,
저장소의 `scripts/train_full_growth.pyw`를 실행할 수 있다.
기존 전체 성장 실행이 있으면 이어 학습하고, 없으면 전투 모델을 옮기거나 새로 시작한다.

저장소 폴더에서 실행한다. `requirements-training.txt`의 별도 학습 환경을 사용한다.

```powershell
$env:PYTHONPATH = 'src'
$env:SDL_VIDEODRIVER = 'windows'
..\.venv-rl\Scripts\python.exe -m space_z.rl.training_wooki.launch --init-combat runs/ppo-score100-recovery --run-dir runs/my-full-growth --workers 4 --steps 2000000
```

기존 평가 최고 전투 모델은 변경하지 않는다. 이동·회전·발사와 벽 우회·모서리 탈출
정책을 보존해 옮기고, 새 성장 학습 카운터는 0부터 시작한다.
새 정책은 전투 잔차와 스탯·전직 선택을 학습한다. 선택 불가능한 성장 행동은 마스크한다.
스탯 포인트를 기다리는 선택은 가능하지만 초기에는 사용할 수 있는 스탯을 선호한다.
전직 보상이 열려 있으면 유효한 후보 중 하나를 고른다.

원래 게임의 1,620개 행동을 PPO의 `[전투 54, 스탯 6, 전직 5]` 선택으로 분해해 연결한다.
스탯·전직은 main `TankEnv.step()` 안에서 적용되며 별도로 게임을 진행하거나 보상을 중복 소비하지 않는다.
성장 상태의 마스크는 원래 관찰값을 사용하므로 관찰 정규화는 끈다.

학습 판은 최대 **600초 또는 사망까지**, 표준 평가 판은 **100초 또는 사망까지**다.
훈련 시작 레벨·점수·포인트를 인위적으로 올리지 않는다.
게임의 실제 득점 증가를 학습에 사용하고, 표준 평가 평균 실제 점수로 최고 모델을 보존한다.
600초 안에 후반 레벨에 도달하지 못하면 그 전직의 성능을 배웠다고 간주하지 않는다.

이어 학습:

```powershell
..\.venv-rl\Scripts\python.exe -m space_z.rl.training_wooki.launch --resume --run-dir runs/my-full-growth --steps 2000000
```

재개하는 것은 정책·옵티마이저·학습 스텝이며, 진행 중이던 판 자체는 새로 시작한다.
전투 모델이 없는 새 PC에서는 `--init-combat` 대신 `--full-growth`로 처음부터 학습할 수 있다.
`runs/`의 모델과 통계는 `.gitignore`에 따라 Git에 포함되지 않는다.

## 검증

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
..\.venv-rl\Scripts\python.exe -m unittest discover -s tests -p test_full_growth.py -v
..\.venv-rl\Scripts\python.exe -m space_z.rl.training_wooki.verify_growth --run-dir runs/my-full-growth --episodes 6 --steps 1500
..\.venv-rl\Scripts\python.exe -m space_z.rl.training_wooki.verify_growth --run-dir runs/my-full-growth --episodes 3 --steps 9000 --output runs/my-full-growth/verification-long.json
```

검증 보고서는 다음을 구분한다.

- **기능 검증:** 모든 행동 조합의 연결, 스탯 포인트 소모와 최대 단계, 모든 전직 경로와 숙련,
  불가능한 선택 마스크, 실제 PPO 갱신·저장·재개. 테스트에만 명시적인 경험치 fixture를 사용한다.
- **자연 플레이 평가:** 레벨 1부터 실제로 획득한 점수·최고 레벨·강화 적용 횟수·전직·숙련·기체 이력.
  여기에는 fixture 경험치나 보너스 포인트를 주지 않는다.

`verification.json`에 아직 관찰하지 못한 기체도 기록한다. 기능 테스트를 통과해도
모든 기체를 잘 다루는 정책이 학습됐다는 뜻은 아니다. `--require-all-ships`를 지정하면
모든 기체가 자연 평가에 등장하지 않은 경우 종료 코드 2를 반환한다.
이는 선택 경로의 관찰 범위를 확인하는 엄격한 옵션이며, 최적 정책이 모든 전직을 골라야 한다는 뜻은 아니다.

모니터에는 현재 레벨·기체·스탯 단계·남은 포인트·전직/숙련 적용 횟수가 표시된다.
실제 학습 worker의 프레임 중계가 아니라 평가 최고 정책의 별도 시연이다.
