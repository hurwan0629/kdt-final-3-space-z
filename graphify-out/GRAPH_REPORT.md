# Graph Report - 2026-10-12_game  (2026-11-01)

## Corpus Check
- 427 files · ~515,425 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 368 file(s) not represented in the graph (top: .zip 281, .jsonl 55, .csv 9)

## Summary
- 2397 nodes · 6287 edges · 138 communities (69 shown, 69 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 1035 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6d17fcdb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- project-structure.md
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- renderer.py
- numpy
- Tank
- .reset
- ResourceMonitor
- Combat
- WeaponsSkillsTests
- GameApplication
- GameConfig
- SpriteAssets
- PlayerProgression
- TileMap
- TerrainConfig
- RewardCalculator
- PixelObservation
- can_show_stat_upgrades
- empty_game
- _muzzle
- wooki02.py
- VectorObservation
- open_game
- ObsConfig
- wooki01.py
- ._tick_effects
- TankEnv
- ActionDecoder
- json
- .step
- environments.py
- TrainSettings
- SmarterEnemyAITests
- dashboard.py
- gpt_training/train.py
- GrowthMaskedPolicy
- 프로젝트 구조
- TileMap
- docs/README.md
- MenusReloadTests
- encode_action
- .pop_score_award_events
- RotationHybridDistribution
- .draw
- gpt_training/environment.py
- dataclasses
- GameCore
- tmp.md
- 점수 개선 실험
- 관련 함수
- LiveCallback
- SpaceFeatures
- train_v2.py
- DifficultyMenuTests
- MinimapTests
- FullGrowthTests
- DisplayAssetsTests
- feature_extractors.py
- PolicyEvaluator
- 실험 기록
- VectorObservation
- StatsPanelTests
- TrainingLog
- Space Z 실시간 강화학습
- random_agent_test.py
- DiscreteActionDecoder
- LearnedGrowthPolicy
- 게임 이미지 에셋
- runtime.py
- GrowthMaskedPolicy
- .application
- ObservationBuilder
- TrainingEnvironmentTests
- make_environment
- torch
- hashlib
- select_model.py
- 연속 회전 PPO 실험 결과
- Code Architecture Guidelines
- architecture/README.md
- new-images/README.md
- 2026-10-29-full-growth-rl.md
- StateOwnershipTests
- space-z
- 재생 주기에 맞춘 추가 순수 PPO 학습
- 관련 함수
- 점수 획득 속도와 안정성 개선
- 연속 회전 PPO 실험 기록
- development/README.md
- RotationContinuousPolicy
- RotationHybridDistribution
- RecordedState
- .test_every_zoom_and_corner_keeps_fractional_player_centered
- Space Z
- VectorObservationTests
- REPORT.md
- RotationContinuousPolicy
- 015: 같은 보상으로 추가 학습
- CombatEvents
- 초기 점수와 하위 성적 개선 결과
- rotation_policy.py
- SpatialFeaturesExtractor
- FullGrowthPolicy
- .image
- TrainingLog
- RendererTests
- Space Z 프로젝트 구조
- time
- 선정 모델 재생 묶음
- PlayerShotEventTests
- Windows 실행 파일 배포
- 처음 실행하기
- 전체 성장 강화학습과 검증

## God Nodes (most connected - your core abstractions)
1. `GameCore` - 152 edges
2. `Renderer` - 109 edges
3. `GameConfig` - 82 edges
4. `TileMap` - 74 edges
5. `ObsConfig` - 63 edges
6. `Tank` - 55 edges
7. `GameApplication` - 51 edges
8. `RewardConfig` - 46 edges
9. `Combat` - 46 edges
10. `VectorObservation` - 44 edges

## Surprising Connections (you probably didn't know these)
- `폴더별 책임` --references--> `GameApplication`  [INFERRED]
  docs/development/project-structure.md → src/space_z/app.py
- `담당 객체와 설정` --references--> `GameConfig`  [INFERRED]
  docs/development/classes/GameCore.md → src/space_z/config/game.py
- `벡터 관찰` --references--> `ObsConfig`  [INFERRED]
  docs/reinforcement-learning/environment.md → src/space_z/config/learning.py
- `학습 환경 경로` --references--> `GameCore`  [INFERRED]
  docs/development/project-structure.md → src/space_z/game/game_core.py
- `관련 함수와 읽는 객체` --references--> `Renderer`  [INFERRED]
  docs/development/classes/Tank.md → src/space_z/rendering/renderer.py

## Import Cycles
- None detected.

## Communities (138 total, 69 thin omitted)

### Community 1 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.07
Nodes (29): 1. 전체 흐름과 현재 모델, 2.1 처음에는 모방학습을 사용했고, 요청에 따라 폐기하지 않고 분리 보존했다, 2.2 순수 PPO를 실행하는 것과 게임을 잘하는 것은 달랐다, 2.3 CNN과 보상을 추가했지만, 쉬운 연습 성적은 실전에 그대로 옮겨지지 않았다, 2.4 조준 입력과 이동 학습을 분리하면서 실전 점수가 올라가기 시작했다, 2.5 더 학습했더니 초반 평균만 좋아져, 보상 비중과 행동 주기를 다시 봤다, 2.6 정상 게임에서도 추가 학습이 항상 좋은 것은 아니었다, 2.7 연속 회전은 200만 스텝을 5개 구간으로 나누고, 조준·몸체를 번갈아 적응시켰다 (+21 more)

### Community 3 - "numpy"
Cohesion: 0.06
Nodes (8): ActionDecoder, TankEnv, ObservationBuilder, build_progression(), ship_one_hot(), demonstration_action(), evasive_action(), route_move()

### Community 4 - "Tank"
Cohesion: 0.05
Nodes (26): EnemyController, 공통 성장 상태, 관련 함수, 역할과 접근 경로, 재등장, 적별 AI 상태, 찾아보기, 월드와 개체 목록 (+18 more)

### Community 5 - ".reset"
Cohesion: 0.08
Nodes (13): GameCore, 관련 동작과 객체, 관련 함수, 담당 객체와 설정, 새 판 초기화, 시간과 전투 이벤트, 역할과 접근 경로, 찾아보기 (+5 more)

### Community 6 - "ResourceMonitor"
Cohesion: 0.09
Nodes (8): RL 사용 안내, 새 학습 코드 작성, 평가와 자원 측정, 폴더 구조, 현재 PPO 실행, ResourceMonitor, EvaluationTests, ResourceMonitorTests

### Community 7 - "Combat"
Cohesion: 0.05
Nodes (30): Bullet, 공격과 충돌, 관련 함수, 역할과 접근 경로, 이동과 수명, 찾아보기, Combat, 관련 동작과 객체 (+22 more)

### Community 9 - "GameApplication"
Cohesion: 0.15
Nodes (3): 메뉴와 표시 상태, GameApplication, MouseControlTests

### Community 10 - "GameConfig"
Cohesion: 0.09
Nodes (14): 설정 변경 주체, 설정과 종류별 정의, 바꾸려는 기능으로 찾기, 적 AI와 월드, 맵과 지형, EnemyAIConfig, GameConfig, ShapeStats (+6 more)

### Community 11 - "SpriteAssets"
Cohesion: 0.08
Nodes (5): _health_mask(), _recolor_health(), _redraw_barrel(), SpriteAssets, resource_path()

### Community 12 - "PlayerProgression"
Cohesion: 0.05
Nodes (22): 역할과 접근 경로, 적 갱신, 점수 획득 기록, PlayerProgression, 관련 동작과 객체, 관련 함수, 역할과 접근 경로, 일반 강화 (+14 more)

### Community 15 - "TileMap"
Cohesion: 0.08
Nodes (4): 관련 동작과 객체, TileMap, ShapeSpawnAvoidsTightCornersTests, SweptCircleTests

### Community 16 - "TerrainConfig"
Cohesion: 0.11
Nodes (12): map_tests(), _clear_outer_wall_band(), expand_map(), _generate_maze(), generate_random_map(), grid_channels(), list_maps(), load_map() (+4 more)

### Community 17 - "RewardCalculator"
Cohesion: 0.26
Nodes (4): RewardConfig, empty_events(), RewardCalculator, RewardCalculatorTests

### Community 18 - "PixelObservation"
Cohesion: 0.07
Nodes (14): 강화학습 환경, 벡터 관찰, 보상과 info, 평가와 자원 측정, 픽셀 관찰, 행동, 환경 만들기, ContinuousActionDecoder (+6 more)

### Community 19 - "can_show_stat_upgrades"
Cohesion: 0.16
Nodes (3): can_show_stat_upgrades(), get_collapsed_stat_panel_rect(), StatPanelCollapseTests

### Community 20 - "empty_game"
Cohesion: 0.15
Nodes (3): empty_game(), OwnershipTests, ProgressionTests

### Community 22 - "wooki02.py"
Cohesion: 0.15
Nodes (13): checkpoint_path(), learning_rate(), evaluate(), load_model(), make_env(), make_model(), one_env(), penalty_components() (+5 more)

### Community 23 - "VectorObservation"
Cohesion: 0.09
Nodes (3): build_progression(), ship_one_hot(), VectorObservation

### Community 24 - "open_game"
Cohesion: 0.13
Nodes (6): EliteEnemyTests, EnemyGrowthTests, evolve(), open_game(), ShipTweakTests, StatLimitTests

### Community 25 - "ObsConfig"
Cohesion: 0.09
Nodes (10): 강화학습 안내, EnvConfig, ObsConfig, 기존 인터페이스로 해결, CornerRecovery, escape_direction(), game_config(), TrainingEnv (+2 more)

### Community 26 - "wooki01.py"
Cohesion: 0.32
Nodes (11): migrate_combat(), main(), atomic_json(), make_model(), publish_checkpoint(), read_json(), main(), parser() (+3 more)

### Community 27 - "._tick_effects"
Cohesion: 0.18
Nodes (10): Beam, 관련 함수, 역할과 접근 경로, 연출 속성, 찾아보기, Explosion, 관련 함수, 역할과 접근 경로 (+2 more)

### Community 29 - "ActionDecoder"
Cohesion: 0.17
Nodes (3): ActionDecoder, FactorizedActionDecoder, PlayerControl

### Community 30 - "json"
Cohesion: 0.09
Nodes (11): copy_dependency_licenses(), main(), verify_bundle(), main(), RejectOtherRLFolders, load_model(), main(), read_manifest() (+3 more)

### Community 31 - ".step"
Cohesion: 0.14
Nodes (9): 시간과 발사 입력, 한 step의 순서, evolve(), gain_level(), main(), capture(), read_player_input(), decode_action() (+1 more)

### Community 32 - "environments.py"
Cohesion: 0.13
Nodes (8): ControlRecord, EpisodeStepRecord, PlayerSnapshot, json_default(), append_record(), reward_components(), RewardBreakdown, StepLogRecord

### Community 33 - "TrainSettings"
Cohesion: 0.19
Nodes (5): TrainSettings, main(), DashboardTests, TrainingJobTests, RuntimeTests

### Community 35 - "dashboard.py"
Cohesion: 0.15
Nodes (6): Demo, draw_panel(), duration(), main(), run_dashboard(), UI

### Community 36 - "gpt_training/train.py"
Cohesion: 0.16
Nodes (14): screenshot(), main(), checkpoint_ranking(), main(), assess(), training_environment(), main(), assess() (+6 more)

### Community 38 - "프로젝트 구조"
Cohesion: 0.25
Nodes (8): 사람이 플레이하는 경로, 소유와 협력, 실제 소스 배치, 처음 읽는 순서, 폴더별 책임, 프로젝트 구조, 학습 환경 경로, 한 프레임의 개요

### Community 40 - "docs/README.md"
Cohesion: 0.22
Nodes (8): 전투와 재장전, 조작과 게임 흐름, 기존 테스트 불일치와 캐시 검토, 현재 제한과 검토 사항, 게임 안내, 성장과 21종 진화, Space Z 문서, 추천 탐색 경로

### Community 43 - ".pop_score_award_events"
Cohesion: 0.14
Nodes (7): 점수 팝업, 바뀌는 상태와 소비자, 시작 조건과 참여 객체, 예시, 플레이어 사망과 천체 파괴의 차이, 피해와 적 처치, ScoreAward

### Community 45 - ".draw"
Cohesion: 0.15
Nodes (6): 게임 상태와 화면 연출의 차이, _hp_color(), _mix(), Renderer, Ghost, RGB

### Community 46 - "gpt_training/environment.py"
Cohesion: 0.09
Nodes (9): main(), measure(), EnvironmentSettings, TacticalObservation, TrainingEnvironment, main(), reward_settings(), ScoreReward (+1 more)

### Community 47 - "dataclasses"
Cohesion: 0.05
Nodes (30): ActionSettings, configuration_to_dict(), evaluation_to_dict(), EvaluationConfig, ExperimentConfig, FeatureExtractorSettings, merge_settings(), ObservationSettings (+22 more)

### Community 49 - "tmp.md"
Cohesion: 0.14
Nodes (12): RL 작업 규칙, 기본 수정 범위, 작업 환경 코드는 직접 요청이 있을 때만 수정, 현재 범위에서 해결할 수 없는 경우, 검증과 보고, 데이터 구조와 객체 구성, 모듈 경계와 변경, 목적 (+4 more)

### Community 50 - "점수 개선 실험"
Cohesion: 0.07
Nodes (30): 006: 구조화 관찰과 가까운 명중 연습, 006 완료와 영상 확인, 007 완료: 연습의 개선이 실전에 전이되지 않음, 007: 움직이는 비발사 적을 상대하는 명중 연습, 008: 기본 게임 직접 학습과 발사·위험 보상, 008 중단과 계산 검증, 009: 발사하는 적과 조준 전용 잔차 신경망, 009 완료 (+22 more)

### Community 54 - "train_v2.py"
Cohesion: 0.05
Nodes (34): ActionSettings, append_record(), configuration_path(), configuration_to_dict(), ControlRecord, create_model(), EpisodeStepRecord, evaluate_checkpoint() (+26 more)

### Community 58 - "DisplayAssetsTests"
Cohesion: 0.08
Nodes (7): build_stat_upgrade_rows(), get_evolution_card_rects(), get_expanded_stat_panel_rect(), Palette, RectBounds, ScorePopup, DisplayAssetsTests

### Community 61 - "실험 기록"
Cohesion: 0.12
Nodes (16): 001_imitation_combat, 002_full_ppo, 003_dagger_full, 004 결과와 다음 단계, 005 결과와 최종 확인, 가설과 조건, 관측 결과, 관측 결과 (+8 more)

### Community 62 - "VectorObservation"
Cohesion: 0.05
Nodes (11): VectorObservation, GrowthFeatures, CombatFeatures, CrowdPolicy, ExploringPolicy, GuardedPolicy, KitingPolicy, PathfindingPolicy (+3 more)

### Community 64 - "TrainingLog"
Cohesion: 0.18
Nodes (3): reward_components(), RewardBreakdown, TrainingLog

### Community 66 - "Space Z 실시간 강화학습"
Cohesion: 0.25
Nodes (7): Space Z 실시간 강화학습, 난이도와 평가, 명령줄과 DQN 비교, 보상, 선택한 모델, 실행, 화면과 파일

### Community 67 - "random_agent_test.py"
Cohesion: 0.39
Nodes (6): check_not_stuck(), determinism_test(), main(), make_env(), rollout(), run_random()

### Community 69 - "DiscreteActionDecoder"
Cohesion: 0.17
Nodes (3): DiscreteActionDecoder, PlayerControl, DiscreteActionDecoderTests

### Community 71 - "게임 이미지 에셋"
Cohesion: 0.33
Nodes (4): v2 이미지 보관본, 게임 이미지 에셋, 그 외 에셋, 진화 트리

### Community 72 - "runtime.py"
Cohesion: 0.12
Nodes (14): bootstrap(), refine(), base_vec(), _clean(), evaluate(), legacy_checkpoint_modules(), load_checkpoint(), make_vec_env() (+6 more)

### Community 77 - "make_environment"
Cohesion: 0.23
Nodes (3): EnvironmentTests, make_environment(), PixelObservationTests

### Community 79 - "torch"
Cohesion: 0.14
Nodes (16): main(), main(), FactorizedActions, game_configuration(), growth_choices(), make_environment(), teacher_action(), main() (+8 more)

### Community 81 - "hashlib"
Cohesion: 0.27
Nodes (5): configure_worker(), evaluate_checkpoint(), main(), main(), evaluate()

### Community 82 - "select_model.py"
Cohesion: 0.27
Nodes (5): compare(), main(), merge_development_evaluation(), summarize(), main()

### Community 83 - "연속 회전 PPO 실험 결과"
Cohesion: 0.18
Nodes (8): 기록과 재현, 실행 환경, 연속 회전 PPO, 실행 범위와 자원 제한 위반, 연속 회전 PPO 실험 결과, 진화 오류와 결과의 한계, 최종 독립 평가, 학습 과정과 최종 모델

### Community 84 - "Code Architecture Guidelines"
Cohesion: 0.15
Nodes (12): 1. Deep Modules & Information Hiding, 2. High Cohesion & Low Coupling, 3. Meaningful Abstraction, 4. Cognitive Load & Readability, 5. Explicit Data & Control Flow, 6. Change Locality, 7. Design & Implementation Decisions, 8. Code Review & Verification (+4 more)

### Community 96 - "재생 주기에 맞춘 추가 순수 PPO 학습"
Cohesion: 0.13
Nodes (15): 019 종료와 추가 검증, 019 중간 판단과 020 계획, 019: 학습과 재생의 행동 주기 맞추기, 020 종료 · 넓은 검증 · 021 계획, 022: 가중치 변화 폭을 줄이는 보정 실험, 022의 개발 40시드 확인과 두 번째 후보 고정, 022의 사전 8시드 선별, 기록 (+7 more)

### Community 97 - "관련 함수"
Cohesion: 0.13
Nodes (10): GameApplication, 관련 동작과 객체, 관련 함수, 스탯 창 상태, 역할과 접근 경로, 찾아보기, 이름과 경로 변경, debug_lines() (+2 more)

### Community 98 - "점수 획득 속도와 안정성 개선"
Cohesion: 0.12
Nodes (17): 016 두 번째 구간과 종료, 016: 보조 보상 축소와 실제 점수·생존 비중 확대, 016 시작의 성장 선택 확인, 016 첫 구간, 017: 이동·성장을 보존하고 실제 전투의 조준만 미세 조정, 017 종료와 개발 16시드 종합 선택, 2프레임 주기 개발 결과, 개발 10시드의 초기 결과 (+9 more)

### Community 100 - "연속 회전 PPO 실험 기록"
Cohesion: 0.17
Nodes (12): 001: 조준 적응, 첫 100,000스텝, 002: 기본 게임 적응과 계획 수정, 003: 이동 적응과 재생 방식 진단, 004: 좋은 조준을 보존한 몸체 학습 연장, 005: 몸체를 보존한 조준 정밀화, 구현 경계, 목적과 조건, 사전 검증 (+4 more)

### Community 101 - "development/README.md"
Cohesion: 0.26
Nodes (4): 개발 안내, 수정 위치와 검증, 이번 구조 정리의 검증 결과, 점수 획득 기록과 팝업 분리 검증

### Community 102 - "RotationContinuousPolicy"
Cohesion: 0.08
Nodes (8): RotationContinuousActionDecoder, EntityEncoder, SpatialFeaturesExtractor, check_evolution_application(), main(), RotationTrainingEnvironment, main(), RotationContinuousPolicy

### Community 105 - ".test_every_zoom_and_corner_keeps_fractional_player_centered"
Cohesion: 0.23
Nodes (7): Renderer, 관련 동작과 객체, 역할과 접근 경로, 찾아보기, 화면과 카메라, 입력과 화면, 설정의 공유, CenteredCameraTests

### Community 106 - "Space Z"
Cohesion: 0.18
Nodes (11): Space Z, 강화학습, 기체 크기와 전체 미니맵, 맵, 스탯과 성장, 실행 파일 만들기, 실행과 검증, 우주선 진화: 21종 (+3 more)

### Community 108 - "REPORT.md"
Cohesion: 0.13
Nodes (12): GPT 학습 실험, 기록과 복원, 기존 train 진입점: 순수 PPO, 실행, 이전 혼합 학습 설계 (001–003, 비교용 보존), 점수 개선 실험 진입점, 고득점 실험 후보의 새 20시드 비교, 기록과 검증 (+4 more)

### Community 110 - "015: 같은 보상으로 추가 학습"
Cohesion: 0.40
Nodes (5): 015: 같은 보상으로 추가 학습, 015 시작 평가, 015 완료: 채택하지 않음, 015 첫 추가 구간: 빠르지만 불안정, 보상 구성 확인

### Community 112 - "초기 점수와 하위 성적 개선 결과"
Cohesion: 0.19
Nodes (9): 독립 20시드 · 기본 게임 · 최대 400초, 로그·검사·복원, 실제로 바꾼 것, 재생, 초기 점수와 하위 성적 개선 결과, 모델 선택 후 새 시드 10개에서 비교, 순수 PPO 점수 개선 결과, 실험과 검증 기록 (+1 more)

### Community 126 - "Space Z 프로젝트 구조"
Cohesion: 0.33
Nodes (5): Space Z 프로젝트 구조, 검증 기록, 근거와 갱신, 읽는 방법과 범위, 파일

### Community 128 - "선정 모델 재생 묶음"
Cohesion: 0.33
Nodes (5): Windows에서 설치·실행, 다른 컴퓨터에 필요한 범위, 무엇이 들어 있는가, 선정 모델 재생 묶음, 터미널 설치와 검증

### Community 130 - "Windows 실행 파일 배포"
Cohesion: 0.50
Nodes (4): Windows 실행 파일 배포, 개발자가 다시 만드는 방법, 받는 사람이 실행하는 방법, 이번 배포본 검증

### Community 131 - "처음 실행하기"
Cohesion: 0.50
Nodes (4): 개발 환경 설치, 실행 옵션, 처음 실행하기, 화면 없이 환경 사용

### Community 132 - "전체 성장 강화학습과 검증"
Cohesion: 0.50
Nodes (3): 검증, 실행, 전체 성장 강화학습과 검증

## Knowledge Gaps
- **1 isolated node(s):** `space-z`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 967 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **69 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GameConfig` connect `GameConfig` to `2. 어떤 문제를 보고 다음 실험을 선택했는가`, `renderer.py`, `numpy`, `전체 성장 강화학습과 검증`, `.reset`, `PlayerShotEventTests`, `WeaponsSkillsTests`, `TileMap`, `PixelObservation`, `empty_game`, `open_game`, `ObsConfig`, `TankEnv`, `json`, `.step`, `SmarterEnemyAITests`, `TileMap`, `MenusReloadTests`, `encode_action`, `gpt_training/environment.py`, `dataclasses`, `GameCore`, `점수 개선 실험`, `관련 함수`, `train_v2.py`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `VectorObservation`, `StatsPanelTests`, `random_agent_test.py`, `DiscreteActionDecoder`, `runtime.py`, `.application`, `make_environment`, `torch`, `StateOwnershipTests`, `관련 함수`, `.test_every_zoom_and_corner_keeps_fractional_player_centered`, `RendererTests`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Are the 85 inferred relationships involving `GameCore` (e.g. with `스탯 창 상태` and `바뀌는 상태와 소비자`) actually correct?**
  _`GameCore` has 85 INFERRED edges - model-reasoned connections that need verification._
- **What connects `space-z` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `2. 어떤 문제를 보고 다음 실험을 선택했는가` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Why does `GameCore` connect `GameCore` to `PlayerShotEventTests`, `renderer.py`, `numpy`, `Tank`, `.reset`, `Combat`, `WeaponsSkillsTests`, `GameApplication`, `GameConfig`, `PlayerProgression`, `TileMap`, `PixelObservation`, `can_show_stat_upgrades`, `empty_game`, `VectorObservation`, `open_game`, `ObsConfig`, `TankEnv`, `.step`, `SmarterEnemyAITests`, `프로젝트 구조`, `TileMap`, `MenusReloadTests`, `encode_action`, `.pop_score_award_events`, `.draw`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `VectorObservation`, `StatsPanelTests`, `.application`, `ObservationBuilder`, `make_environment`, `StateOwnershipTests`, `관련 함수`, `.test_every_zoom_and_corner_keeps_fractional_player_centered`, `CombatEvents`, `.build`, `RendererTests`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `Renderer` (e.g. with `공격과 충돌` and `스탯 창 상태`) actually correct?**
  _`Renderer` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Should `renderer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09456740442655935 - nodes in this community are weakly interconnected._