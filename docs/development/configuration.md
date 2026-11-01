# 설정과 종류별 정의

설정 데이터는 `space_z.config`에 모으고 분야별 파일로 나눕니다. 게임의 현재 체력·점수는 설정이 아니라 각 런타임 객체의 상태입니다.

| 파일 | 설정·정의 | 사용하는 곳 |
|---|---|---|
| [game.py](../../src/space_z/config/game.py) | `GameConfig`, 기본 체력·탄속·FPS·월드, 천체 종류 | 코어·전투·화면·관찰 |
| [enemy_ai.py](../../src/space_z/config/enemy_ai.py) | `EnemyAIConfig`, 감지·반응·추적·성장 | 적 처리 |
| [terrain.py](../../src/space_z/config/terrain.py) | `TerrainConfig`, 타일·블록·생성·지형 기능 | 지형 생성·충돌·화면 |
| [learning.py](../../src/space_z/config/learning.py) | `ObsConfig`, `RewardConfig`, `EnvConfig` | 관찰·학습 보상·환경 step |
| [progression_definitions.py](../../src/space_z/config/progression_definitions.py) | `RewardCard`, `ShotSpec`, `ShipClass`, 강화·우주선 목록 | 실제 성장·전투·기체 표시 |

```python
from space_z.config import GameConfig
from space_z.game.game_core import GameCore

game_config = GameConfig(player_hp=150)
game = GameCore(game_config, seed=42)
```

기본 탄속과 종류별 탄속 배율은 모두 설정 데이터입니다. 같은 폴더 안에서 공통 기준과 종류별 목록을 나눠 찾기 쉽게 합니다. `config/progression_definitions.py`는 수치·종류를 정의하고 `game/player_progression.py`는 포인트 소비·강화·진화를 적용합니다. `rendering/progression_panel.py`는 패널 영역과 표시 문구를 계산합니다.

`GameConfig.enemy_ai`와 `.terrain`은 설정 객체이며 각 인스턴스마다 새로 만듭니다. 공유 설정을 받은 코어·앱·렌더러는 같은 인스턴스를 참조할 수 있습니다. `GameConfig.player_hp`와 `Tank.max_hp`의 차이, 계산된 property와 저장 필드의 차이는 클래스 문서에서 구분합니다.

## 설정 변경 주체

| 시점·종류 | 변경 주체 | 적용 방식 |
|---|---|---|
| 실행 인자와 초기 맵·규칙 선택 | `main()` 또는 환경 생성 코드 | 객체 생성 전에 설정 인스턴스 구성 |
| 메뉴의 난이도·새 맵 선택 | `GameApplication` | 메뉴 선택을 반영하고 필요한 새 판 초기화 수행 |
| 실행 중 카메라 줌 | `Renderer.set_zoom()` | 공유 `camera_zoom`과 렌더러 뷰를 함께 갱신 |
| 현재 체력·위치·점수 | 각 게임 담당 객체 | 설정을 덮어쓰지 않고 런타임 상태 변경 |

기존 `GameConfig(...)` 생성자·필드명·기본값은 유지합니다. 동일한 설정을 받은 앱·코어·렌더러는 변경을 공유하므로, 독립된 게임 환경에는 별도 설정 인스턴스를 생성합니다. 실행 중 지형 설정을 변경한 경우 실제 맵 재생성은 별도 처리입니다.

[[development/classes/PlayerProgression|실제 강화 처리]] · [[development/classes/Combat#발사 상태|실제 탄속 계산]] · [[development/README|개발 안내]]
