# Graph Report - 2026-10-12_game  (2026-11-02)

## Corpus Check
- 459 files · ~566,979 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 101 file(s) not represented in the graph (top: .jsonl 66, .csv 9, .request 7)

## Summary
- 2690 nodes · 6848 edges · 164 communities (95 shown, 69 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 1039 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ed934be4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- project-structure.md
- experiment.py
- renderer.py
- ActionDecoder
- Tank
- GameCore
- ResourceMonitor
- Combat
- WeaponsSkillsTests
- shared_source_sha256
- GameConfig
- SpriteAssets
- PlayerProgression
- diagnose_walls.py
- TileMap
- _generate_maze
- RewardCalculator
- TankEnv
- can_show_stat_upgrades
- empty_game
- TerrainConfig
- wooki02.py
- VectorObservation
- open_game
- TrainingEnv
- shared_source_sha256
- ._tick_effects
- TankEnv
- ActionDecoder
- run_session.py
- .update
- environments.py
- TrainSettings
- SmarterEnemyAITests
- dashboard.py
- gpt_training/environment.py
- GrowthMaskedPolicy
- 프로젝트 구조
- TileMap
- docs/README.md
- MenusReloadTests
- shared_source_sha256
- Renderer
- progression_panel.py
- .draw
- ScoreReward
- environments.py
- .size
- tmp.md
- 점수 개선 실험
- 관련 함수
- LiveCallback
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- evaluate_checkpoint
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
- gpt_training_rotation_continuous/policy.py
- Space Z 실시간 강화학습
- random_agent_test.py
- ObservationBuilder
- DiscreteActionDecoder
- build_progression
- 게임 이미지 에셋
- wooki01.py
- GrowthMaskedPolicy
- .application
- ObservationBuilder
- gpt_training_rotation_continuous/environment.py
- make_environment
- gpt_training/check.py
- write_json
- json
- 연속 회전 PPO 실험 결과
- Code Architecture Guidelines
- architecture/README.md
- new-images/README.md
- 2026-10-29-full-growth-rl.md
- StateOwnershipTests
- space-z
- 재생 주기에 맞춘 추가 순수 PPO 학습
- .activate_menu_action
- 점수 획득 속도와 안정성 개선
- 연속 회전 PPO 실험 기록
- 수정 위치와 검증
- RotationContinuousPolicy
- RotationHybridDistribution
- RecordedState
- .test_every_zoom_and_corner_keeps_fractional_player_centered
- Space Z
- runtime.py
- gpt_report/README.md
- RotationContinuousPolicy
- 003_training_duration/manifest.json
- 001_contract_checks/manifest.json
- 초기 점수와 하위 성적 개선 결과
- ContinuousActionDecoder
- read_configuration
- SpatialFeaturesExtractor
- PixelObservation
- architecture
- .image
- 002_full_game_pilot/manifest.json
- Space Z 프로젝트 구조
- parent
- 선정 모델 재생 묶음
- torch
- Windows 실행 파일 배포
- 처음 실행하기
- TankEnv
- ObsConfig
- architecture
- architecture
- growth.py
- _health_mask
- GPT 학습 실험
- generate_random_map
- SpaceFeatures
- versions
- versions
- LearnedGrowthPolicy
- TilePosition
- project-structure.md
- ShapeStats
- 현재 작업
- .pop_events
- versions
- KitingPolicy
- RuntimeTests
- .__init__
- encode_action
- _muzzle
- build_progression
- 016: 보조 보상 축소와 실제 점수·생존 비중 확대
- 추가 학습 결과: 고득점 후보 보존, 기본 모델 유지
- configuration_path
- escape_direction

## God Nodes (most connected - your core abstractions)
1. `GameCore` - 152 edges
2. `Renderer` - 114 edges
3. `GameConfig` - 85 edges
4. `TileMap` - 74 edges
5. `ObsConfig` - 66 edges
6. `Tank` - 55 edges
7. `GameApplication` - 51 edges
8. `RewardConfig` - 48 edges
9. `Combat` - 46 edges
10. `VectorObservation` - 46 edges

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

## Communities (164 total, 69 thin omitted)

### Community 1 - "experiment.py"
Cohesion: 0.05
Nodes (31): RotationHybridDistribution, main(), main(), environment_configuration(), make_environment(), OutcomeRewardEnvironment, EpisodeRecorder, evaluate() (+23 more)

### Community 2 - "renderer.py"
Cohesion: 0.05
Nodes (3): move_from_vector(), build_menu_buttons(), ship_one_hot()

### Community 4 - "Tank"
Cohesion: 0.04
Nodes (33): EnemyController, 공통 성장 상태, 관련 함수, 역할과 접근 경로, 재등장, 적별 AI 상태, 찾아보기, 월드와 개체 목록 (+25 more)

### Community 5 - "GameCore"
Cohesion: 0.08
Nodes (11): 관련 함수, Shape, 역할과 접근 경로, 위치와 표류, 종류와 체력, 찾아보기, GameCore, ScoreAward (+3 more)

### Community 6 - "ResourceMonitor"
Cohesion: 0.10
Nodes (6): RL 사용 안내, 평가와 자원 측정, 폴더 구조, 현재 PPO 실행, ResourceMonitor, ResourceMonitorTests

### Community 7 - "Combat"
Cohesion: 0.06
Nodes (33): Bullet, 공격과 충돌, 관련 함수, 역할과 접근 경로, 이동과 수명, 찾아보기, Combat, 관련 동작과 객체 (+25 more)

### Community 9 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 10 - "GameConfig"
Cohesion: 0.11
Nodes (17): 설정과 종류별 정의, 바꾸려는 기능으로 찾기, 적 AI와 월드, 검증, 실행, 전체 성장 강화학습과 검증, 강화학습 안내, EnemyAIConfig (+9 more)

### Community 11 - "SpriteAssets"
Cohesion: 0.11
Nodes (3): _recolor_health(), _redraw_barrel(), SpriteAssets

### Community 12 - "PlayerProgression"
Cohesion: 0.07
Nodes (17): 적 갱신, PlayerProgression, 관련 동작과 객체, 관련 함수, 역할과 접근 경로, 일반 강화, 점수와 레벨, 진화와 숙련 (+9 more)

### Community 13 - "diagnose_walls.py"
Cohesion: 0.12
Nodes (14): BulletObserver, capture_frame(), choose_clips(), encode_json(), expected_wall_grid(), main(), write_record(), replay_and_capture() (+6 more)

### Community 16 - "_generate_maze"
Cohesion: 0.25
Nodes (3): expand_map(), _generate_maze(), MapError

### Community 17 - "RewardCalculator"
Cohesion: 0.11
Nodes (6): RewardConfig, empty_events(), CombatEvents, RewardCalculator, PlayerShotEventTests, RewardCalculatorTests

### Community 18 - "TankEnv"
Cohesion: 0.15
Nodes (3): main(), make_environment(), ModelTrainer

### Community 19 - "can_show_stat_upgrades"
Cohesion: 0.14
Nodes (5): 설정 변경 주체, GameApplication, main(), get_collapsed_stat_panel_rect(), StatPanelCollapseTests

### Community 20 - "empty_game"
Cohesion: 0.12
Nodes (4): empty_game(), OwnershipTests, ProgressionTests, RendererTests

### Community 21 - "TerrainConfig"
Cohesion: 0.22
Nodes (6): TerrainConfig, grid_channels(), load_map(), map_to_text(), parse_map(), LargeWallLayoutTests

### Community 22 - "wooki02.py"
Cohesion: 0.15
Nodes (14): checkpoint_path(), learning_rate(), evaluate(), load_model(), make_env(), make_model(), one_env(), penalty_components() (+6 more)

### Community 24 - "open_game"
Cohesion: 0.13
Nodes (6): EliteEnemyTests, EnemyGrowthTests, evolve(), open_game(), ShipTweakTests, StatLimitTests

### Community 25 - "TrainingEnv"
Cohesion: 0.24
Nodes (3): CornerRecovery, game_config(), TrainingEnv

### Community 26 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 27 - "._tick_effects"
Cohesion: 0.18
Nodes (10): Beam, 관련 함수, 역할과 접근 경로, 연출 속성, 찾아보기, Explosion, 관련 함수, 역할과 접근 경로 (+2 more)

### Community 29 - "ActionDecoder"
Cohesion: 0.12
Nodes (4): ActionDecoder, FactorizedActionDecoder, RotationContinuousActionDecoder, PlayerControl

### Community 30 - "run_session.py"
Cohesion: 0.14
Nodes (11): json_default(), ModelRegistry, prepare_session(), RunSession, save_checkpoint(), save_configuration(), session_status(), TrainingRunRecord (+3 more)

### Community 31 - ".update"
Cohesion: 0.14
Nodes (7): evolve(), gain_level(), main(), read_player_input(), step_with_score_award(), key_state(), MouseControlTests

### Community 32 - "environments.py"
Cohesion: 0.11
Nodes (8): ControlRecord, EpisodeStepRecord, PlayerSnapshot, append_record(), reward_components(), RewardBreakdown, StepLogRecord, TrainingLog

### Community 33 - "TrainSettings"
Cohesion: 0.19
Nodes (6): refine(), load_checkpoint(), TrainSettings, main(), DashboardTests, TrainingJobTests

### Community 35 - "dashboard.py"
Cohesion: 0.16
Nodes (6): Demo, draw_panel(), duration(), main(), run_dashboard(), UI

### Community 36 - "gpt_training/environment.py"
Cohesion: 0.11
Nodes (25): main(), measure(), make_environment(), teacher_action(), main(), main(), checkpoint_ranking(), main() (+17 more)

### Community 38 - "프로젝트 구조"
Cohesion: 0.29
Nodes (7): 사람이 플레이하는 경로, 실제 소스 배치, 처음 읽는 순서, 폴더별 책임, 프로젝트 구조, 학습 환경 경로, 한 프레임의 개요

### Community 40 - "docs/README.md"
Cohesion: 0.20
Nodes (9): 전투와 재장전, 조작과 게임 흐름, 기존 테스트 불일치와 캐시 검토, 현재 제한과 검토 사항, 맵과 지형, 게임 안내, 성장과 21종 진화, Space Z 문서 (+1 more)

### Community 42 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 43 - "Renderer"
Cohesion: 0.25
Nodes (6): Renderer, 게임 상태와 화면 연출의 차이, 관련 동작과 객체, 역할과 접근 경로, 점수 팝업, 찾아보기

### Community 44 - "progression_panel.py"
Cohesion: 0.16
Nodes (10): 이름과 경로 변경, RewardCard, build_player_stat_summary(), build_stat_upgrade_rows(), can_show_stat_upgrades(), get_evolution_card_rects(), get_expanded_stat_panel_rect(), Palette (+2 more)

### Community 46 - "ScoreReward"
Cohesion: 0.26
Nodes (3): main(), ScoreReward, ScoreRewardSettings

### Community 47 - "environments.py"
Cohesion: 0.12
Nodes (10): configuration_to_dict(), ExperimentConfig, PPOOverrides, make_environment(), make_training_environments(), build_policy_kwargs(), create_model(), LoggedPPO (+2 more)

### Community 48 - ".size"
Cohesion: 0.30
Nodes (3): _hp_color(), _mix(), RGB

### Community 49 - "tmp.md"
Cohesion: 0.14
Nodes (12): RL 작업 규칙, 기본 수정 범위, 작업 환경 코드는 직접 요청이 있을 때만 수정, 현재 범위에서 해결할 수 없는 경우, 검증과 보고, 데이터 구조와 객체 구성, 모듈 경계와 변경, 목적 (+4 more)

### Community 50 - "점수 개선 실험"
Cohesion: 0.07
Nodes (30): 006: 구조화 관찰과 가까운 명중 연습, 006 완료와 영상 확인, 007 완료: 연습의 개선이 실전에 전이되지 않음, 007: 움직이는 비발사 적을 상대하는 명중 연습, 008: 기본 게임 직접 학습과 발사·위험 보상, 008 중단과 계산 검증, 009: 발사하는 적과 조준 전용 잔차 신경망, 009 완료 (+22 more)

### Community 52 - "LiveCallback"
Cohesion: 0.38
Nodes (3): publish_checkpoint(), LiveCallback, _run_training()

### Community 53 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.07
Nodes (29): 1. 전체 흐름과 현재 모델, 2.1 처음에는 모방학습을 사용했고, 요청에 따라 폐기하지 않고 분리 보존했다, 2.2 순수 PPO를 실행하는 것과 게임을 잘하는 것은 달랐다, 2.3 CNN과 보상을 추가했지만, 쉬운 연습 성적은 실전에 그대로 옮겨지지 않았다, 2.4 조준 입력과 이동 학습을 분리하면서 실전 점수가 올라가기 시작했다, 2.5 더 학습했더니 초반 평균만 좋아져, 보상 비중과 행동 주기를 다시 봤다, 2.6 정상 게임에서도 추가 학습이 항상 좋은 것은 아니었다, 2.7 연속 회전은 200만 스텝을 5개 구간으로 나누고, 조준·몸체를 번갈아 적응시켰다 (+21 more)

### Community 54 - "evaluate_checkpoint"
Cohesion: 0.08
Nodes (23): configuration_path(), ControlRecord, create_model(), EpisodeStepRecord, evaluate_checkpoint(), ExperimentConfig, LoggedPPO, main() (+15 more)

### Community 57 - "FullGrowthTests"
Cohesion: 0.17
Nodes (3): legal_growth_masks(), FullGrowthTests, TrainingEnvironmentTests

### Community 60 - "PolicyEvaluator"
Cohesion: 0.13
Nodes (3): PolicyEvaluator, evaluate_checkpoint(), EvaluationTests

### Community 61 - "실험 기록"
Cohesion: 0.12
Nodes (16): 001_imitation_combat, 002_full_ppo, 003_dagger_full, 004 결과와 다음 단계, 005 결과와 최종 확인, 가설과 조건, 관측 결과, 관측 결과 (+8 more)

### Community 62 - "VectorObservation"
Cohesion: 0.11
Nodes (8): VectorObservation, route_move(), CombatFeatures, ExploringPolicy, GuardedPolicy, PathfindingPolicy, RecoveryPolicy, ThreatCombatFeatures

### Community 64 - "TrainingLog"
Cohesion: 0.20
Nodes (3): reward_components(), RewardBreakdown, TrainingLog

### Community 66 - "Space Z 실시간 강화학습"
Cohesion: 0.25
Nodes (7): Space Z 실시간 강화학습, 난이도와 평가, 명령줄과 DQN 비교, 보상, 선택한 모델, 실행, 화면과 파일

### Community 67 - "random_agent_test.py"
Cohesion: 0.33
Nodes (8): check_not_stuck(), determinism_test(), main(), make_env(), map_tests(), rollout(), run_random(), list_maps()

### Community 69 - "DiscreteActionDecoder"
Cohesion: 0.19
Nodes (3): DiscreteActionDecoder, PlayerControl, DiscreteActionDecoderTests

### Community 70 - "build_progression"
Cohesion: 0.12
Nodes (4): TacticalObservation, build_progression(), VectorObservationTests, enemy()

### Community 71 - "게임 이미지 에셋"
Cohesion: 0.33
Nodes (4): v2 이미지 보관본, 게임 이미지 에셋, 그 외 에셋, 진화 트리

### Community 72 - "wooki01.py"
Cohesion: 0.23
Nodes (11): migrate_combat(), main(), atomic_json(), read_json(), stage_for_step(), main(), verify(), main() (+3 more)

### Community 76 - "gpt_training_rotation_continuous/environment.py"
Cohesion: 0.15
Nodes (7): RotationContinuousActionDecoder, EnvironmentSettings, check_evolution_application(), main(), RotationTrainingEnvironment, main(), training_environment()

### Community 77 - "make_environment"
Cohesion: 0.22
Nodes (3): EnvironmentTests, make_environment(), PixelObservationTests

### Community 79 - "gpt_training/check.py"
Cohesion: 0.16
Nodes (4): main(), FactorizedActions, growth_choices(), TrainingEnvironment

### Community 81 - "write_json"
Cohesion: 0.11
Nodes (7): append_record(), configuration_to_dict(), evaluation_to_dict(), EvaluationConfig, json_default(), merge_settings(), RecordedState

### Community 82 - "json"
Cohesion: 0.10
Nodes (8): copy_dependency_licenses(), main(), verify_bundle(), compare(), main(), merge_development_evaluation(), summarize(), main()

### Community 83 - "연속 회전 PPO 실험 결과"
Cohesion: 0.18
Nodes (8): 기록과 재현, 실행 환경, 연속 회전 PPO, 실행 범위와 자원 제한 위반, 연속 회전 PPO 실험 결과, 진화 오류와 결과의 한계, 최종 독립 평가, 학습 과정과 최종 모델

### Community 84 - "Code Architecture Guidelines"
Cohesion: 0.15
Nodes (12): 1. Deep Modules & Information Hiding, 2. High Cohesion & Low Coupling, 3. Meaningful Abstraction, 4. Cognitive Load & Readability, 5. Explicit Data & Control Flow, 6. Change Locality, 7. Design & Implementation Decisions, 8. Code Review & Verification (+4 more)

### Community 96 - "재생 주기에 맞춘 추가 순수 PPO 학습"
Cohesion: 0.13
Nodes (15): 019 종료와 추가 검증, 019 중간 판단과 020 계획, 019: 학습과 재생의 행동 주기 맞추기, 020 종료 · 넓은 검증 · 021 계획, 022: 가중치 변화 폭을 줄이는 보정 실험, 022의 개발 40시드 확인과 두 번째 후보 고정, 022의 사전 8시드 선별, 기록 (+7 more)

### Community 97 - ".activate_menu_action"
Cohesion: 0.15
Nodes (9): GameApplication, 관련 동작과 객체, 관련 함수, 메뉴와 표시 상태, 스탯 창 상태, 시간과 발사 입력, 역할과 접근 경로, 찾아보기 (+1 more)

### Community 98 - "점수 획득 속도와 안정성 개선"
Cohesion: 0.12
Nodes (17): 015: 같은 보상으로 추가 학습, 015 시작 평가, 015 완료: 채택하지 않음, 015 첫 추가 구간: 빠르지만 불안정, 017: 이동·성장을 보존하고 실제 전투의 조준만 미세 조정, 017 종료와 개발 16시드 종합 선택, 2프레임 주기 개발 결과, 개발 10시드의 초기 결과 (+9 more)

### Community 100 - "연속 회전 PPO 실험 기록"
Cohesion: 0.15
Nodes (13): 001: 조준 적응, 첫 100,000스텝, 002: 기본 게임 적응과 계획 수정, 003: 이동 적응과 재생 방식 진단, 004: 좋은 조준을 보존한 몸체 학습 연장, 005: 몸체를 보존한 조준 정밀화, 구현 경계, 목적과 조건, 사전 검증 (+5 more)

### Community 101 - "수정 위치와 검증"
Cohesion: 0.67
Nodes (3): 수정 위치와 검증, 이번 구조 정리의 검증 결과, 점수 획득 기록과 팝업 분리 검증

### Community 102 - "RotationContinuousPolicy"
Cohesion: 0.11
Nodes (9): configure_worker(), evaluate_checkpoint(), main(), main(), digest(), main(), read(), RotationContinuousPolicy (+1 more)

### Community 106 - "Space Z"
Cohesion: 0.18
Nodes (11): Space Z, 강화학습, 기체 크기와 전체 미니맵, 맵, 스탯과 성장, 실행 파일 만들기, 실행과 검증, 우주선 진화: 21종 (+3 more)

### Community 107 - "runtime.py"
Cohesion: 0.16
Nodes (11): bootstrap(), demonstration_action(), evasive_action(), base_vec(), _clean(), evaluate(), legacy_checkpoint_modules(), make_vec_env() (+3 more)

### Community 108 - "gpt_report/README.md"
Cohesion: 0.20
Nodes (5): 모델 선택 후 새 시드 10개에서 비교, 순수 PPO 점수 개선 결과, 실험과 검증 기록, 재생과 TensorBoard, 오류가 있는 초기 실행의 보존 기록

### Community 110 - "003_training_duration/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 111 - "001_contract_checks/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 112 - "초기 점수와 하위 성적 개선 결과"
Cohesion: 0.40
Nodes (5): 독립 20시드 · 기본 게임 · 최대 400초, 로그·검사·복원, 실제로 바꾼 것, 재생, 초기 점수와 하위 성적 개선 결과

### Community 114 - "read_configuration"
Cohesion: 0.11
Nodes (9): ActionSettings, evaluation_to_dict(), EvaluationConfig, FeatureExtractorSettings, game_configuration(), merge_settings(), ObservationSettings, read_configuration() (+1 more)

### Community 116 - "PixelObservation"
Cohesion: 0.10
Nodes (9): 강화학습 환경, 벡터 관찰, 보상과 info, 평가와 자원 측정, 픽셀 관찰, 행동, 환경 만들기, PixelObservation (+1 more)

### Community 117 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 124 - "002_full_game_pilot/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 126 - "Space Z 프로젝트 구조"
Cohesion: 0.40
Nodes (5): Space Z 프로젝트 구조, 검증 기록, 근거와 갱신, 읽는 방법과 범위, 파일

### Community 127 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 128 - "선정 모델 재생 묶음"
Cohesion: 0.33
Nodes (5): Windows에서 설치·실행, 다른 컴퓨터에 필요한 범위, 무엇이 들어 있는가, 선정 모델 재생 묶음, 터미널 설치와 검증

### Community 129 - "torch"
Cohesion: 0.10
Nodes (11): display(), main(), main(), RejectOtherRLFolders, load_model(), main(), read_manifest(), main() (+3 more)

### Community 130 - "Windows 실행 파일 배포"
Cohesion: 0.50
Nodes (4): Windows 실행 파일 배포, 개발자가 다시 만드는 방법, 받는 사람이 실행하는 방법, 이번 배포본 검증

### Community 131 - "처음 실행하기"
Cohesion: 0.50
Nodes (4): 개발 환경 설치, 실행 옵션, 처음 실행하기, 화면 없이 환경 사용

### Community 134 - "ObsConfig"
Cohesion: 0.12
Nodes (3): capture(), ObsConfig, encode_growth_action()

### Community 135 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 137 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 138 - "growth.py"
Cohesion: 0.20
Nodes (4): FullGrowthPolicy, GrowthFeatures, TacticalCombatFeatures, make_model()

### Community 140 - "GPT 학습 실험"
Cohesion: 0.33
Nodes (6): GPT 학습 실험, 기록과 복원, 기존 train 진입점: 순수 PPO, 실행, 이전 혼합 학습 설계 (001–003, 비교용 보존), 점수 개선 실험 진입점

### Community 141 - "generate_random_map"
Cohesion: 0.17
Nodes (3): _clear_outer_wall_band(), empty_map(), generate_random_map()

### Community 143 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 144 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 149 - "현재 작업"
Cohesion: 0.29
Nodes (6): 003 완료 상태, 다음 판단, 목표와 경계, 사용자 진입점과 기록, 현재 작업, 확인 명령 (프로젝트 루트 PowerShell)

### Community 151 - ".pop_events"
Cohesion: 0.18
Nodes (8): GameCore, 관련 동작과 객체, 담당 객체와 설정, 새 판 초기화, 시간과 전투 이벤트, 역할과 접근 경로, 찾아보기, 새 학습 코드 작성

### Community 152 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 159 - "016: 보조 보상 축소와 실제 점수·생존 비중 확대"
Cohesion: 0.40
Nodes (5): 016 두 번째 구간과 종료, 016: 보조 보상 축소와 실제 점수·생존 비중 확대, 016 시작의 성장 선택 확인, 016 첫 구간, 추론 방식과 실패 장면 확인

### Community 160 - "추가 학습 결과: 고득점 후보 보존, 기본 모델 유지"
Cohesion: 0.40
Nodes (5): 고득점 실험 후보의 새 20시드 비교, 기록과 검증, 실제로 진행한 것, 직접 재생, 추가 학습 결과: 고득점 후보 보존, 기본 모델 유지

## Knowledge Gaps
- **1 isolated node(s):** `space-z`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1176 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **69 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GameConfig` connect `GameConfig` to `torch`, `renderer.py`, `experiment.py`, `Tank`, `GameCore`, `ObsConfig`, `WeaponsSkillsTests`, `generate_random_map`, `RewardCalculator`, `TankEnv`, `can_show_stat_upgrades`, `empty_game`, `TerrainConfig`, `.pop_events`, `open_game`, `TrainingEnv`, `.__init__`, `TankEnv`, `.update`, `SmarterEnemyAITests`, `gpt_training/environment.py`, `TileMap`, `docs/README.md`, `MenusReloadTests`, `environments.py`, `점수 개선 실험`, `관련 함수`, `2. 어떤 문제를 보고 다음 실험을 선택했는가`, `evaluate_checkpoint`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `VectorObservation`, `StatsPanelTests`, `random_agent_test.py`, `.application`, `make_environment`, `StateOwnershipTests`, `.test_every_zoom_and_corner_keeps_fractional_player_centered`, `runtime.py`, `read_configuration`, `PixelObservation`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Are the 85 inferred relationships involving `GameCore` (e.g. with `스탯 창 상태` and `바뀌는 상태와 소비자`) actually correct?**
  _`GameCore` has 85 INFERRED edges - model-reasoned connections that need verification._
- **What connects `space-z` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `experiment.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05095839177185601 - nodes in this community are weakly interconnected._
- **Why does `GameCore` connect `GameCore` to `renderer.py`, `Tank`, `TankEnv`, `Combat`, `WeaponsSkillsTests`, `GameConfig`, `PlayerProgression`, `generate_random_map`, `TileMap`, `RewardCalculator`, `TilePosition`, `can_show_stat_upgrades`, `empty_game`, `.handle_event`, `.pop_events`, `VectorObservation`, `open_game`, `.__init__`, `TankEnv`, `build_progression`, `.update`, `SmarterEnemyAITests`, `프로젝트 구조`, `TileMap`, `MenusReloadTests`, `Renderer`, `progression_panel.py`, `.draw`, `.size`, `관련 함수`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `VectorObservation`, `StatsPanelTests`, `ObservationBuilder`, `build_progression`, `.application`, `ObservationBuilder`, `make_environment`, `StateOwnershipTests`, `.activate_menu_action`, `.test_every_zoom_and_corner_keeps_fractional_player_centered`, `PixelObservation`, `.build`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `Renderer` (e.g. with `공격과 충돌` and `스탯 창 상태`) actually correct?**
  _`Renderer` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Should `renderer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.054758800521512385 - nodes in this community are weakly interconnected._