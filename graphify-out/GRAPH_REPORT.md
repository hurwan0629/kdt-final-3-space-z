# Graph Report - 2026-10-12_game  (2026-11-02)

## Corpus Check
- 487 files · ~641,774 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 117 file(s) not represented in the graph (top: .jsonl 82, .csv 9, .request 7)

## Summary
- 3003 nodes · 7189 edges · 168 communities (110 shown, 58 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 1043 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0bb06b9c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- docs/README.md
- experiment.py
- shared_source_sha256
- EnemyController
- GameCore
- ResourceMonitor
- Combat
- WeaponsSkillsTests
- shared_source_sha256
- 전체 성장 강화학습과 검증
- SpriteAssets
- PlayerProgression
- diagnose_walls.py
- TileMap
- shared_source_sha256
- RewardCalculator
- VectorObservation
- shared_source_sha256
- empty_game
- generate_random_map
- wooki02.py
- best_model_execute/environment.py
- open_game
- shared_source_sha256
- shared_source_sha256
- ._tick_effects
- TankEnv
- ActionDecoder
- run_session.py
- GameApplication
- SmallHybridDistribution
- runtime.py
- SmarterEnemyAITests
- dashboard.py
- gpt_training/environment.py
- GrowthMaskedPolicy
- 프로젝트 구조
- TileMap
- make_environment
- MenusReloadTests
- shared_source_sha256
- .step
- renderer.py
- Renderer
- ScoreReward
- environments.py
- CombatFeatures
- checks.py
- 점수 개선 실험
- 관련 함수
- 관련 함수
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- evaluate_checkpoint
- DifficultyMenuTests
- MinimapTests
- FullGrowthTests
- DisplayAssetsTests
- feature_extractors.py
- PolicyEvaluator
- 실험 기록
- training_wooki/policy.py
- StatsPanelTests
- stable_baselines3_common_vec_env
- numpy
- Space Z 실시간 강화학습
- random_agent_test.py
- ActionDecoder
- DiscreteActionDecoder
- ._enemy_ai
- 게임 이미지 에셋
- wooki01.py
- GrowthMaskedPolicy
- RotationHybridDistribution
- 004_parallel_continuation/manifest.json
- make_environment
- pathlib
- 005_same_settings_continuation/manifest.json
- 006_lower_learning_rate/manifest.json
- Code Architecture Guidelines
- architecture/README.md
- new-images/README.md
- 2026-10-29-full-growth-rl.md
- 007_higher_combat_reward/manifest.json
- space-z
- 재생 주기에 맞춘 추가 순수 PPO 학습
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- 점수 획득 속도와 안정성 개선
- 연속 회전 PPO 실험 기록
- time
- RotationContinuousPolicy
- RotationHybridDistribution
- RecordedState
- TrainingEnvironmentTests
- 강화학습 환경
- architecture
- gpt_report/README.md
- RotationContinuousPolicy
- 003_training_duration/manifest.json
- 001_contract_checks/manifest.json
- 초기 점수와 하위 성적 개선 결과
- ContinuousActionDecoder
- GameConfig
- SpatialFeaturesExtractor
- PixelObservation
- architecture
- parent
- 002_full_game_pilot/manifest.json
- architecture
- parent
- 선정 모델 재생 묶음
- torch
- parent
- architecture
- TankEnv
- PlayerControl
- ObsConfig
- architecture
- architecture
- growth.py
- parent
- GPT 학습 실험
- game_core.py
- SpaceFeatures
- versions
- versions
- architecture
- parent
- VectorFeaturesExtractor
- TrainingJsonWriter
- 현재 작업
- ._clamp_to_arena
- .record_player_kill
- versions
- TargetEnv
- RotationContinuousActionDecoder
- .__init__
- 014: 조준을 보존하고 기본 게임에서 이동·성장 학습
- progression_definitions.py
- versions
- versions
- versions
- versions
- PlayerProgression
- render_screens.py
- verify_loaded_parent

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
- `역할과 접근 경로` --references--> `Combat`  [INFERRED]
  docs/development/classes/Bullet.md → src/space_z/game/combat.py
- `역할과 접근 경로` --references--> `Tank`  [INFERRED]
  docs/development/classes/EnemyController.md → src/space_z/game/entities.py

## Import Cycles
- None detected.

## Communities (168 total, 58 thin omitted)

### Community 0 - "docs/README.md"
Cohesion: 0.05
Nodes (53): Space Z 프로젝트 구조, 검증 기록, 근거와 갱신, 읽는 방법과 범위, 파일, 클래스와 상태, 시작 조건과 참여 객체, 예시: 트윈을 한 번 클릭 (+45 more)

### Community 1 - "experiment.py"
Cohesion: 0.16
Nodes (9): main(), EpisodeRecorder, evaluate(), parent_metadata(), parse_reward_overrides(), collect_telemetry(), empty_telemetry(), episode_metrics() (+1 more)

### Community 3 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 4 - "EnemyController"
Cohesion: 0.10
Nodes (10): EnemyController, 공통 성장 상태, 관련 함수, 역할과 접근 경로, 재등장, 적 갱신, 찾아보기, 관련 함수 (+2 more)

### Community 5 - "GameCore"
Cohesion: 0.07
Nodes (14): 관련 함수, Shape, 관련 함수, 역할과 접근 경로, 위치와 표류, 종류와 체력, 찾아보기, 바뀌는 상태와 소비자 (+6 more)

### Community 6 - "ResourceMonitor"
Cohesion: 0.09
Nodes (7): RL 사용 안내, 새 학습 코드 작성, 평가와 자원 측정, 폴더 구조, 현재 PPO 실행, ResourceMonitor, ResourceMonitorTests

### Community 7 - "Combat"
Cohesion: 0.12
Nodes (8): 피해 적용, 체력 상태, 시작 조건과 참여 객체, 예시, 플레이어 사망과 천체 파괴의 차이, 피해와 적 처치, Combat, Explosion

### Community 9 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 10 - "전체 성장 강화학습과 검증"
Cohesion: 0.50
Nodes (3): 검증, 실행, 전체 성장 강화학습과 검증

### Community 11 - "SpriteAssets"
Cohesion: 0.09
Nodes (4): _health_mask(), _recolor_health(), _redraw_barrel(), SpriteAssets

### Community 12 - "PlayerProgression"
Cohesion: 0.08
Nodes (11): 일반 강화, 진화와 숙련, 강화와 진화, 시작 조건과 참여 객체, 예시와 주의할 구분, 일반 강화 순서, 진화와 숙련 순서, 상태의 저장 위치와 변경 책임 (+3 more)

### Community 13 - "diagnose_walls.py"
Cohesion: 0.11
Nodes (15): BulletObserver, capture_frame(), choose_clips(), encode_json(), expected_wall_grid(), main(), write_record(), replay_and_capture() (+7 more)

### Community 15 - "TileMap"
Cohesion: 0.08
Nodes (4): 관련 동작과 객체, TileMap, ShapeSpawnAvoidsTightCornersTests, SweptCircleTests

### Community 16 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 17 - "RewardCalculator"
Cohesion: 0.14
Nodes (6): RewardConfig, empty_events(), RewardCalculator, make_environment(), PlayerShotEventTests, RewardCalculatorTests

### Community 18 - "VectorObservation"
Cohesion: 0.06
Nodes (10): EnvConfig, TankEnv, VectorObservation, main(), ModelTrainer, CornerRecovery, escape_direction(), game_config() (+2 more)

### Community 19 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 20 - "empty_game"
Cohesion: 0.15
Nodes (3): empty_game(), OwnershipTests, ProgressionTests

### Community 21 - "generate_random_map"
Cohesion: 0.10
Nodes (12): map_tests(), _clear_outer_wall_band(), expand_map(), _generate_maze(), generate_random_map(), grid_channels(), list_maps(), load_map() (+4 more)

### Community 22 - "wooki02.py"
Cohesion: 0.14
Nodes (15): checkpoint_path(), learning_rate(), evaluate(), load_model(), main(), make_env(), make_model(), one_env() (+7 more)

### Community 23 - "best_model_execute/environment.py"
Cohesion: 0.06
Nodes (4): ObservationBuilder, build_progression(), ship_one_hot(), VectorObservation

### Community 24 - "open_game"
Cohesion: 0.13
Nodes (6): EliteEnemyTests, EnemyGrowthTests, evolve(), open_game(), ShipTweakTests, StatLimitTests

### Community 25 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 26 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 27 - "._tick_effects"
Cohesion: 0.18
Nodes (10): Beam, 관련 함수, 역할과 접근 경로, 연출 속성, 찾아보기, Explosion, 관련 함수, 역할과 접근 경로 (+2 more)

### Community 29 - "ActionDecoder"
Cohesion: 0.17
Nodes (3): ActionDecoder, FactorizedActionDecoder, PlayerControl

### Community 30 - "run_session.py"
Cohesion: 0.11
Nodes (13): make_training_environments(), json_default(), ModelRegistry, prepare_session(), RunSession, save_checkpoint(), save_configuration(), session_status() (+5 more)

### Community 31 - "GameApplication"
Cohesion: 0.06
Nodes (22): GameApplication, 관련 동작과 객체, 관련 함수, 메뉴와 표시 상태, 스탯 창 상태, 시간과 발사 입력, 역할과 접근 경로, 찾아보기 (+14 more)

### Community 32 - "SmallHybridDistribution"
Cohesion: 0.10
Nodes (9): SmallHybridDistribution, SmallHybridPolicy, Schedule은 PyTorch와 어떤 관계인가?, 객체 소유 구조, 실제 SB3 PPO 조립 연습, 실행, 이 예제에서 생략한 요소, 차원과 호출 위치 (+1 more)

### Community 33 - "runtime.py"
Cohesion: 0.10
Nodes (14): demonstration_action(), evasive_action(), refine(), base_vec(), _clean(), legacy_checkpoint_modules(), load_checkpoint(), _one_env() (+6 more)

### Community 35 - "dashboard.py"
Cohesion: 0.16
Nodes (6): Demo, draw_panel(), duration(), main(), run_dashboard(), UI

### Community 36 - "gpt_training/environment.py"
Cohesion: 0.13
Nodes (25): main(), measure(), screenshot(), make_environment(), teacher_action(), main(), checkpoint_ranking(), main() (+17 more)

### Community 38 - "프로젝트 구조"
Cohesion: 0.29
Nodes (7): 사람이 플레이하는 경로, 실제 소스 배치, 처음 읽는 순서, 폴더별 책임, 프로젝트 구조, 학습 환경 경로, 한 프레임의 개요

### Community 39 - "TileMap"
Cohesion: 0.10
Nodes (3): resource_path(), AssetRefreshTests, step_with_score_award()

### Community 40 - "make_environment"
Cohesion: 0.14
Nodes (4): make_environment(), OutcomeRewardEnvironment, main(), VectorOnlyPolicy

### Community 42 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 43 - ".step"
Cohesion: 0.18
Nodes (8): Bullet, 공격과 충돌, 관련 함수, 역할과 접근 경로, 이동과 수명, 찾아보기, 발사 상태, 처리 순서

### Community 44 - "renderer.py"
Cohesion: 0.10
Nodes (5): _ShapeSnap, Palette, ScorePopup, build_progression(), ship_one_hot()

### Community 45 - "Renderer"
Cohesion: 0.09
Nodes (6): 화면과 카메라, _hp_color(), _mix(), Renderer, Ghost, CenteredCameraTests

### Community 46 - "ScoreReward"
Cohesion: 0.15
Nodes (4): TrainingEnvironment, main(), ScoreReward, ScoreRewardSettings

### Community 47 - "environments.py"
Cohesion: 0.07
Nodes (17): ExperimentConfig, PPOOverrides, ControlRecord, EpisodeStepRecord, make_environment(), PlayerSnapshot, build_policy_kwargs(), create_model() (+9 more)

### Community 48 - "CombatFeatures"
Cohesion: 0.17
Nodes (4): GrowthFeatures, CombatFeatures, TacticalCombatFeatures, ThreatCombatFeatures

### Community 49 - "checks.py"
Cohesion: 0.40
Nodes (11): main(), environment_configuration(), main(), assess(), new_experiment(), save_model(), update_report(), event() (+3 more)

### Community 50 - "점수 개선 실험"
Cohesion: 0.08
Nodes (25): 006: 구조화 관찰과 가까운 명중 연습, 006 완료와 영상 확인, 007 완료: 연습의 개선이 실전에 전이되지 않음, 007: 움직이는 비발사 적을 상대하는 명중 연습, 008: 기본 게임 직접 학습과 발사·위험 보상, 008 중단과 계산 검증, 009: 발사하는 적과 조준 전용 잔차 신경망, 009 완료 (+17 more)

### Community 51 - "관련 함수"
Cohesion: 0.10
Nodes (10): 새 판 초기화, Renderer, 게임 상태와 화면 연출의 차이, 관련 동작과 객체, 관련 함수, 역할과 접근 경로, 점수 팝업, 지형과 이미지 (+2 more)

### Community 52 - "관련 함수"
Cohesion: 0.18
Nodes (7): Combat, 관련 동작과 객체, 관련 함수, 재장전, 찾아보기, 한 프레임의 순서, _segment_circle_hit()

### Community 53 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.10
Nodes (20): 1. 전체 흐름과 현재 모델, 3. 무엇을 어디에서 바꾸었는가, 4. 실제로 어떤 순서로 작업하고 검증했는가, 5. 이 보고서는 어떤 전문가 방식을 참고해 구성했는가, 6. 직접 확인하는 가장 짧은 경로, 7. 상세 설정 원장, PPO와 실행 주기, 관찰과 신경망 (+12 more)

### Community 54 - "evaluate_checkpoint"
Cohesion: 0.04
Nodes (37): ActionSettings, append_record(), configuration_path(), configuration_to_dict(), ControlRecord, create_model(), EpisodeStepRecord, evaluate_checkpoint() (+29 more)

### Community 57 - "FullGrowthTests"
Cohesion: 0.17
Nodes (3): legal_growth_masks(), LearnedGrowthPolicy, FullGrowthTests

### Community 58 - "DisplayAssetsTests"
Cohesion: 0.08
Nodes (6): build_stat_upgrade_rows(), get_evolution_card_rects(), get_expanded_stat_panel_rect(), RectBounds, DisplayAssetsTests, RendererTests

### Community 61 - "실험 기록"
Cohesion: 0.12
Nodes (16): 001_imitation_combat, 002_full_ppo, 003_dagger_full, 004 결과와 다음 단계, 005 결과와 최종 확인, 가설과 조건, 관측 결과, 관측 결과 (+8 more)

### Community 62 - "training_wooki/policy.py"
Cohesion: 0.11
Nodes (7): route_move(), CrowdPolicy, ExploringPolicy, GuardedPolicy, KitingPolicy, PathfindingPolicy, RecoveryPolicy

### Community 63 - "StatsPanelTests"
Cohesion: 0.15
Nodes (3): build_player_stat_summary(), StateOwnershipTests, StatsPanelTests

### Community 64 - "stable_baselines3_common_vec_env"
Cohesion: 0.18
Nodes (3): learning_rate(), main(), ProgressCallback

### Community 65 - "numpy"
Cohesion: 0.07
Nodes (8): RotationContinuousActionDecoder, EnvironmentSettings, EntityEncoder, SpatialFeaturesExtractor, check_evolution_application(), main(), RotationTrainingEnvironment, SmallFeaturesExtractor

### Community 66 - "Space Z 실시간 강화학습"
Cohesion: 0.25
Nodes (7): Space Z 실시간 강화학습, 난이도와 평가, 명령줄과 DQN 비교, 보상, 선택한 모델, 실행, 화면과 파일

### Community 67 - "random_agent_test.py"
Cohesion: 0.39
Nodes (6): check_not_stuck(), determinism_test(), main(), make_env(), rollout(), run_random()

### Community 69 - "DiscreteActionDecoder"
Cohesion: 0.20
Nodes (3): DiscreteActionDecoder, encode_growth_action(), DiscreteActionDecoderTests

### Community 70 - "._enemy_ai"
Cohesion: 0.08
Nodes (10): 적별 AI 상태, AI 기억과 이동 목표, Tank, 관련 함수와 읽는 객체, 무장과 탄약, 역할과 접근 경로, 위치와 이동, 찾아보기 (+2 more)

### Community 71 - "게임 이미지 에셋"
Cohesion: 0.33
Nodes (4): v2 이미지 보관본, 게임 이미지 에셋, 그 외 에셋, 진화 트리

### Community 72 - "wooki01.py"
Cohesion: 0.17
Nodes (16): bootstrap(), migrate_combat(), main(), atomic_json(), evaluate(), make_model(), make_vec_env(), normalizer() (+8 more)

### Community 76 - "004_parallel_continuation/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 77 - "make_environment"
Cohesion: 0.22
Nodes (3): EnvironmentTests, make_environment(), PixelObservationTests

### Community 79 - "pathlib"
Cohesion: 0.16
Nodes (5): display(), main(), main(), FactorizedActions, growth_choices()

### Community 81 - "005_same_settings_continuation/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 83 - "006_lower_learning_rate/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 84 - "Code Architecture Guidelines"
Cohesion: 0.15
Nodes (12): 1. Deep Modules & Information Hiding, 2. High Cohesion & Low Coupling, 3. Meaningful Abstraction, 4. Cognitive Load & Readability, 5. Explicit Data & Control Flow, 6. Change Locality, 7. Design & Implementation Decisions, 8. Code Review & Verification (+4 more)

### Community 89 - "007_higher_combat_reward/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 96 - "재생 주기에 맞춘 추가 순수 PPO 학습"
Cohesion: 0.13
Nodes (15): 019 종료와 추가 검증, 019 중간 판단과 020 계획, 019: 학습과 재생의 행동 주기 맞추기, 020 종료 · 넓은 검증 · 021 계획, 022: 가중치 변화 폭을 줄이는 보정 실험, 022의 개발 40시드 확인과 두 번째 후보 고정, 022의 사전 8시드 선별, 기록 (+7 more)

### Community 97 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.22
Nodes (9): 2.1 처음에는 모방학습을 사용했고, 요청에 따라 폐기하지 않고 분리 보존했다, 2.2 순수 PPO를 실행하는 것과 게임을 잘하는 것은 달랐다, 2.3 CNN과 보상을 추가했지만, 쉬운 연습 성적은 실전에 그대로 옮겨지지 않았다, 2.4 조준 입력과 이동 학습을 분리하면서 실전 점수가 올라가기 시작했다, 2.5 더 학습했더니 초반 평균만 좋아져, 보상 비중과 행동 주기를 다시 봤다, 2.6 정상 게임에서도 추가 학습이 항상 좋은 것은 아니었다, 2.7 연속 회전은 200만 스텝을 5개 구간으로 나누고, 조준·몸체를 번갈아 적응시켰다, 2.8 실제 화면에서 발견한 진화 오류로 최종 평가를 다시 했다 (+1 more)

### Community 98 - "점수 획득 속도와 안정성 개선"
Cohesion: 0.09
Nodes (22): 015: 같은 보상으로 추가 학습, 015 시작 평가, 015 완료: 채택하지 않음, 015 첫 추가 구간: 빠르지만 불안정, 016 두 번째 구간과 종료, 016: 보조 보상 축소와 실제 점수·생존 비중 확대, 016 시작의 성장 선택 확인, 016 첫 구간 (+14 more)

### Community 100 - "연속 회전 PPO 실험 기록"
Cohesion: 0.08
Nodes (21): 기록과 재현, 실행 환경, 연속 회전 PPO, 001: 조준 적응, 첫 100,000스텝, 002: 기본 게임 적응과 계획 수정, 003: 이동 적응과 재생 방식 진단, 004: 좋은 조준을 보존한 몸체 학습 연장, 005: 몸체를 보존한 조준 정밀화 (+13 more)

### Community 101 - "time"
Cohesion: 0.36
Nodes (3): build_report(), atomic_replace(), read_jsonl()

### Community 102 - "RotationContinuousPolicy"
Cohesion: 0.07
Nodes (18): copy_dependency_licenses(), main(), verify_bundle(), configure_worker(), evaluate_checkpoint(), main(), main(), digest() (+10 more)

### Community 106 - "강화학습 환경"
Cohesion: 0.25
Nodes (8): 강화학습 환경, 벡터 관찰, 보상과 info, 평가와 자원 측정, 픽셀 관찰, 한 step의 순서, 행동, 환경 만들기

### Community 107 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 108 - "gpt_report/README.md"
Cohesion: 0.15
Nodes (10): 고득점 실험 후보의 새 20시드 비교, 기록과 검증, 실제로 진행한 것, 직접 재생, 추가 학습 결과: 고득점 후보 보존, 기본 모델 유지, 모델 선택 후 새 시드 10개에서 비교, 순수 PPO 점수 개선 결과, 실험과 검증 기록 (+2 more)

### Community 110 - "003_training_duration/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 111 - "001_contract_checks/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 112 - "초기 점수와 하위 성적 개선 결과"
Cohesion: 0.40
Nodes (5): 독립 20시드 · 기본 게임 · 최대 400초, 로그·검사·복원, 실제로 바꾼 것, 재생, 초기 점수와 하위 성적 개선 결과

### Community 114 - "GameConfig"
Cohesion: 0.08
Nodes (22): 설정 변경 주체, 설정과 종류별 정의, 바꾸려는 기능으로 찾기, 적 AI와 월드, 맵과 지형, EnemyAIConfig, GameConfig, ShapeStats (+14 more)

### Community 117 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 118 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 124 - "002_full_game_pilot/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 126 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 127 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 128 - "선정 모델 재생 묶음"
Cohesion: 0.33
Nodes (5): Windows에서 설치·실행, 다른 컴퓨터에 필요한 범위, 무엇이 들어 있는가, 선정 모델 재생 묶음, 터미널 설치와 검증

### Community 129 - "torch"
Cohesion: 0.12
Nodes (10): main(), RejectOtherRLFolders, load_model(), main(), read_manifest(), main(), game_configuration(), main() (+2 more)

### Community 130 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 131 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 134 - "ObsConfig"
Cohesion: 0.08
Nodes (7): 강화학습 안내, capture(), ObsConfig, stage_for_step(), VectorObservationTests, enemy(), RuntimeTests

### Community 135 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 137 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 139 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 140 - "GPT 학습 실험"
Cohesion: 0.33
Nodes (6): GPT 학습 실험, 기록과 복원, 기존 train 진입점: 순수 PPO, 실행, 이전 혼합 학습 설계 (001–003, 비교용 보존), 점수 개선 실험 진입점

### Community 141 - "game_core.py"
Cohesion: 0.12
Nodes (14): 월드와 개체 목록, 바뀌는 상태, move_from_vector(), _overlap(), _push_apart(), Beam, Bullet, Shape (+6 more)

### Community 143 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 144 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 145 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 146 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 149 - "현재 작업"
Cohesion: 0.18
Nodes (10): 005 이후 작은 수정이 필요할 경우, 005 진행 중 · 사용자 후속 승인, 검증과 제약, 명령, 범위와 유지 사항, 사용자 선호, 실행 순서와 결과, 완료 상태 (+2 more)

### Community 151 - ".record_player_kill"
Cohesion: 0.13
Nodes (10): 역할과 접근 경로, GameCore, 관련 동작과 객체, 담당 객체와 설정, 시간과 전투 이벤트, 역할과 접근 경로, 점수 획득 기록, 찾아보기 (+2 more)

### Community 152 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 156 - "014: 조준을 보존하고 기본 게임에서 이동·성장 학습"
Cohesion: 0.40
Nodes (5): 014 두 번째 구간, 014 완료: 검증 목표 달성, 014: 조준을 보존하고 기본 게임에서 이동·성장 학습, 014 첫 구간과 긴 플레이 확인 계획, 실제 학습 가중치 검사

### Community 157 - "progression_definitions.py"
Cohesion: 0.18
Nodes (3): _muzzle(), ShotOverrides, ShotSpec

### Community 158 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 159 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 160 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 161 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 162 - "PlayerProgression"
Cohesion: 0.50
Nodes (4): PlayerProgression, 관련 동작과 객체, 역할과 접근 경로, 찾아보기

## Knowledge Gaps
- **1 isolated node(s):** `space-z`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1441 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **58 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GameConfig` connect `GameConfig` to `torch`, `config/__init__.py`, `GameCore`, `WeaponsSkillsTests`, `전체 성장 강화학습과 검증`, `game_core.py`, `TileMap`, `RewardCalculator`, `VectorObservation`, `empty_game`, `generate_random_map`, `._clamp_to_arena`, `best_model_execute/environment.py`, `.record_player_kill`, `open_game`, `.__init__`, `014: 조준을 보존하고 기본 게임에서 이동·성장 학습`, `TankEnv`, `GameApplication`, `runtime.py`, `SmarterEnemyAITests`, `TileMap`, `make_environment`, `MenusReloadTests`, `renderer.py`, `Renderer`, `CombatFeatures`, `checks.py`, `관련 함수`, `2. 어떤 문제를 보고 다음 실험을 선택했는가`, `evaluate_checkpoint`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `training_wooki/policy.py`, `StatsPanelTests`, `random_agent_test.py`, `ActionDecoder`, `make_environment`, `강화학습 환경`, `PixelObservation`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Are the 85 inferred relationships involving `GameCore` (e.g. with `스탯 창 상태` and `바뀌는 상태와 소비자`) actually correct?**
  _`GameCore` has 85 INFERRED edges - model-reasoned connections that need verification._
- **What connects `space-z` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.05030525030525031 - nodes in this community are weakly interconnected._
- **Why does `GameCore` connect `GameCore` to `config/__init__.py`, `EnemyController`, `TankEnv`, `ObsConfig`, `Combat`, `WeaponsSkillsTests`, `PlayerProgression`, `game_core.py`, `TileMap`, `RewardCalculator`, `VectorObservation`, `empty_game`, `generate_random_map`, `._clamp_to_arena`, `.record_player_kill`, `best_model_execute/environment.py`, `open_game`, `.__init__`, `TankEnv`, `GameApplication`, `SmarterEnemyAITests`, `프로젝트 구조`, `TileMap`, `MenusReloadTests`, `.step`, `renderer.py`, `Renderer`, `관련 함수`, `DifficultyMenuTests`, `MinimapTests`, `DisplayAssetsTests`, `StatsPanelTests`, `ActionDecoder`, `._enemy_ai`, `make_environment`, `강화학습 환경`, `GameConfig`, `PixelObservation`, `.build`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `Renderer` (e.g. with `공격과 충돌` and `스탯 창 상태`) actually correct?**
  _`Renderer` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Should `config/__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1036734693877551 - nodes in this community are weakly interconnected._