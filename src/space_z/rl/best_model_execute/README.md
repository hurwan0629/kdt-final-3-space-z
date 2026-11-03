# 선정 모델 재생 묶음

이 폴더를 다른 컴퓨터의 같은 게임 프로젝트 `src/space_z/rl/` 아래에 복사하면, 기존 `gpt_training*/` 없이 두 모델을 재생할 수 있다. 모델·정책·관찰·행동 해석기·환경 연결 코드는 모두 **이 폴더 바로 아래**에 있다. 학습·모방·자동 조준 스크립트는 실행하지 않는다.

공유할 때는 `best_model_execute.zip` 하나를 전달해도 된다. 받는 사람이 게임 프로젝트의 `src/space_z/rl/` 아래에 압축을 풀면 `best_model_execute/`가 생긴다. ZIP 자체는 실행에 사용하지 않는다.

## 무엇이 들어 있는가

| 실행 이름 | 원래 선정 모델 | 내용 |
|---|---|---|
| `discrete` | D018, 기존 `pure_ppo_model.zip` | 이동·회전·발사·강화·진화 모두 이산. 초반 하위 성적까지 고려해 유지한 기본 모델 |
| `rotation` | R005의 계보 1,200,000스텝 선정 후보 | 회전만 연속 -1~1, 나머지는 이산. 진화 int 오류를 고친 실행 방식 |

‘최고’는 각 연구에서 최종 선정한 모델을 뜻한다. 평균 점수는 더 높았지만 초반 p25 기준을 통과하지 못한 D022와 초기 모방학습 모델은 포함하지 않았다.

`models.json`에 원본·배포 ZIP 해시, 클래스 경로, 설정과 코드 출처를 기록했다. 다른 폴더를 참조하던 직렬화 경로만 배포용으로 바꾸었으며 정책의 모든 가중치는 원본과 비트 단위로 같다. ZIP 해시는 다시 저장했으므로 원본과 다르다. 기본 재생은 두 모델 모두 CPU·greedy·2게임프레임마다 결정이다.

## 다른 컴퓨터에 필요한 범위

기존 게임 프로젝트의 아래 항목은 있어야 한다. **게임 전체를 대체하는 독립 게임 EXE는 아니다.**

```text
프로젝트/
  assets/                        게임 이미지
  maps/                          게임 맵
  src/space_z/
    __init__.py
    config/                      게임 설정·기체·성장 정의
    game/                        게임 코어
    rendering/                   화면 출력
    resource_paths.py
    rl/
      best_model_execute/        이 폴더 전체 복사
```

공용 `rl/` 바로 아래 파일이 있어도 무방하며, 이 묶음은 `rl/__init__.py`나 `actions/`, `observations/`, 다른 학습 폴더를 import하지 않는다. 초기화 시 다른 RL 폴더를 불러오는 일을 피하려고 실행은 아래의 **직접 파일 경로**를 사용한다. 게임 코어·맵·이미지는 같은 프로젝트 버전을 사용해야 성능과 재현 조건이 유지된다. 필요한 공유 Python 소스의 해시는 `models.json/shared_game_source_sha256`에 있다.

## Windows에서 설치·실행

1. **64비트 Python 3.12**를 설치한다. 이미 설치되어 있다면 생략한다.
2. 이 폴더의 `setup.bat`을 한 번 실행한다. 인터넷에서 패키지를 설치하므로 처음에는 시간이 걸린다.
3. `play_discrete.bat` 또는 `play_rotation.bat`을 실행한다.

가상환경 `.venv`는 해당 컴퓨터에서 설치 과정이 만든다. 기존 학습용 가상환경을 복사할 필요가 없다. CUDA/GPU도 필요 없다. 창을 닫거나 Ctrl+C로 종료할 수 있으며 모델 파일은 수정하지 않는다.

PowerShell에서 옵션을 지정하려면 다음처럼 실행한다.

```powershell
cd src/space_z/rl/best_model_execute
.\.venv\Scripts\python.exe -B play.py --model discrete --seed 61001 --zoom 0.35
.\.venv\Scripts\python.exe -B play.py --model rotation --seed 61001 --zoom 0.35
```

기본은 최대 **400게임초**다. 시간을 바꾸려면 `--max-seconds 600`, 화면 없이 확인하려면 `--headless --max-seconds 4`를 붙인다. 시작 시 모델 이름, 종료 시 점수·처치·게임 시간·최종 기체를 출력한다. 행동은 모델이 선택하며 스탯/진화를 별도 규칙으로 강제하지 않는다.

## 터미널 설치와 검증

Windows의 배치 파일 대신 직접 설치할 수도 있다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B check.py
```

Linux/macOS에서는 Python 3.12로 `.venv`를 만들고 `.venv/bin/python`을 사용한다. 화면 재생에는 데스크톱 디스플레이가 필요하다. 검증은 Windows/Python 3.12/CPU에서 수행했으며 다른 OS의 실제 화면까지 시험하지는 않았다.

`check.py`는 원래 RL 패키지 import를 차단한 채 두 모델을 로드하고, 실제 예측·환경 진행·성장 요청의 적용을 확인한다. 검사에 사용하는 점수 부여는 진화 검증용 fixture이며 일반 재생이나 학습에는 적용하지 않는다. 배포 제작 시의 격리·동일성 검사 결과는 `verification.json`에 보존한다.

설치 버전은 실제 검증 환경에 맞췄다. NumPy 2.5.4의 Python 3.12 이상 요구와 패키지 존재 여부는 [PyPI 메타데이터](https://pypi.org/pypi/numpy/2.5.4/json)로 확인했다. CPU PyTorch wheel은 [공식 저장소](https://download.pytorch.org/whl/cpu)를 사용한다. 설치에는 Python과 네트워크가 필요하며 실행 이후에는 모델 다운로드나 외부 서비스 접속이 없다.
