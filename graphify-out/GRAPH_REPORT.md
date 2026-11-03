# Graph Report - 2026-10-12_game  (2026-11-03)

## Corpus Check
- 671 files · ~965,573 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 154 file(s) not represented in the graph (top: .jsonl 119, .csv 9, .request 7)

## Summary
- 3565 nodes · 7834 edges · 202 communities (152 shown, 50 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 1041 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `550cf9bd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- docs/README.md
- experiment.py
- config/__init__.py
- shared_source_sha256
- EnemyController
- Shape
- ResourceMonitor
- Tank
- WeaponsSkillsTests
- shared_source_sha256
- combat_bootstrap.py
- SpriteAssets
- PlayerProgression
- diagnose_walls.py
- hashlib
- TileMap
- shared_source_sha256
- RewardConfig
- shared_source_sha256
- empty_game
- shared_source_sha256
- wooki02.py
- open_game
- shared_source_sha256
- shared_source_sha256
- ._tick_effects
- game/terrain.py
- ActionDecoder
- run_session.py
- GameApplication
- SmallHybridDistribution
- .get_menu_buttons
- shared_source_sha256
- dashboard.py
- gpt_training/environment.py
- GrowthMaskedPolicy
- 프로젝트 구조
- TileMap
- random_agent_test.py
- MenusReloadTests
- shared_source_sha256
- shared_source_sha256
- numpy
- Renderer
- ScoreReward
- environments.py
- shared_source_sha256
- shared_source_sha256
- 점수 개선 실험
- 관련 함수
- shared_source_sha256
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- VectorObservation
- DifficultyMenuTests
- MinimapTests
- FullGrowthTests
- _muzzle
- feature_extractors.py
- record_milestones.py
- 실험 기록
- FullGrowthPolicy
- StatsPanelTests
- stable_baselines3_common_vec_env
- numpy
- Space Z 실시간 강화학습
- TrainingEnv
- .record_player_kill
- DiscreteActionDecoder
- ObsConfig
- 게임 이미지 에셋
- runtime.py
- GrowthMaskedPolicy
- shared_source_sha256
- 004_parallel_continuation/manifest.json
- training_hurwan/train.py
- protocol
- shared_source_sha256
- 005_same_settings_continuation/manifest.json
- SpatialFeaturesExtractor
- 006_lower_learning_rate/manifest.json
- Code Architecture Guidelines
- architecture/README.md
- new-images/README.md
- 2026-10-29-full-growth-rl.md
- 007_higher_combat_reward/manifest.json
- space-z
- 재생 주기에 맞춘 추가 순수 PPO 학습
- combat.py
- 점수 획득 속도와 안정성 개선
- 연속 회전 PPO 실험 기록
- trainer.py
- pathlib
- SpaceFeatures
- RecordedState
- TrainingEnvironmentTests
- SmarterEnemyAITests
- architecture
- gpt_report/README.md
- gpt_training_rotation_continuous/report.py
- 003_training_duration/manifest.json
- 001_contract_checks/manifest.json
- 초기 점수와 하위 성적 개선 결과
- SweptCircleTests
- GameConfig
- LiveCallback
- architecture
- parent
- 연속 회전 PPO 실험 결과
- 002_full_game_pilot/manifest.json
- configuration.py
- architecture
- parent
- 선정 모델 재생 묶음
- raw/gpt_training_vector_only/experiments/002_full_game_pilot/manifest.json
- parent
- architecture
- 008_combined_reward_and_lr/manifest.json
- TrainSettings
- architecture
- architecture
- ShapeStats
- parent
- GPT 학습 실험
- WorldPoint
- raw/gpt_training_vector_only/experiments/003_training_duration/manifest.json
- versions
- versions
- architecture
- parent
- .test_mostly_avoids_a_tight_corner_that_leaves_no_room_to_orbit
- 현재 작업
- GameCore
- raw/gpt_training_vector_only/experiments/004_parallel_continuation/manifest.json
- versions
- TargetEnv
- raw/gpt_training_vector_only/experiments/005_same_settings_continuation/manifest.json
- raw/gpt_training_vector_only/experiments/006_lower_learning_rate/manifest.json
- architecture
- versions
- versions
- versions
- versions
- parent
- raw/gpt_training_vector_only/experiments/007_higher_combat_reward/manifest.json
- raw/gpt_training_vector_only/experiments/008_combined_reward_and_lr/manifest.json
- .step
- train_v2.py
- .pop_score_award_events
- versions
- TrainingEnvironment
- 영상 체크포인트 20개 시드 검증
- argparse
- 2. 어떤 문제를 보고 다음 실험을 선택했는가
- 016: 보조 보상 축소와 실제 점수·생존 비중 확대
- 평가 결과 모음
- architecture
- architecture
- parent
- architecture
- parent
- architecture
- parent
- architecture
- parent
- architecture
- parent
- architecture
- parent
- 추가 학습 결과: 고득점 후보 보존, 기본 모델 유지
- versions
- versions
- versions
- versions
- versions
- versions
- versions
- 순수 PPO 점수 개선 결과
- 전체 성장 강화학습과 검증
- ObservationBuilder

## God Nodes (most connected - your core abstractions)
1. `GameCore` - 152 edges
2. `Renderer` - 116 edges
3. `GameConfig` - 87 edges
4. `TileMap` - 74 edges
5. `ObsConfig` - 66 edges
6. `Tank` - 55 edges
7. `GameApplication` - 51 edges
8. `RewardConfig` - 50 edges
9. `VectorObservation` - 48 edges
10. `TankEnv` - 47 edges

## Surprising Connections (you probably didn't know these)
- `폴더별 책임` --references--> `GameApplication`  [INFERRED]
  docs/development/project-structure.md → src/space_z/app.py
- `담당 객체와 설정` --references--> `GameConfig`  [INFERRED]
  docs/development/classes/GameCore.md → src/space_z/config/game.py
- `벡터 관찰` --references--> `ObsConfig`  [INFERRED]
  docs/reinforcement-learning/environment.md → src/space_z/config/learning.py
- `역할과 접근 경로` --references--> `Tank`  [INFERRED]
  docs/development/classes/EnemyController.md → src/space_z/game/entities.py
- `역할과 접근 경로` --references--> `Tank`  [INFERRED]
  docs/development/classes/Tank.md → src/space_z/game/entities.py

## Import Cycles
- None detected.

## Communities (202 total, 50 thin omitted)

### Community 0 - "docs/README.md"
Cohesion: 0.05
Nodes (50): Space Z 프로젝트 구조, 검증 기록, 근거와 갱신, 읽는 방법과 범위, 파일, 클래스와 상태, 객체 간 동작 흐름, 개발 안내 (+42 more)

### Community 1 - "experiment.py"
Cohesion: 0.06
Nodes (37): display(), main(), collect(), evaluation_nodes(), main(), normalize(), plot(), read_json() (+29 more)

### Community 2 - "config/__init__.py"
Cohesion: 0.09
Nodes (3): empty_map(), Palette, ScorePopup

### Community 3 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 4 - "EnemyController"
Cohesion: 0.08
Nodes (12): EnemyController, 공통 성장 상태, 관련 함수, 역할과 접근 경로, 재등장, 적 갱신, 적별 AI 상태, 찾아보기 (+4 more)

### Community 5 - "Shape"
Cohesion: 0.07
Nodes (10): 관련 함수, 새 판 초기화, Shape, 관련 함수, 역할과 접근 경로, 위치와 표류, 종류와 체력, 찾아보기 (+2 more)

### Community 6 - "ResourceMonitor"
Cohesion: 0.07
Nodes (9): PolicyEvaluator, RL 사용 안내, 새 학습 코드 작성, 평가와 자원 측정, 폴더 구조, 현재 PPO 실행, ResourceMonitor, EvaluationTests (+1 more)

### Community 7 - "Tank"
Cohesion: 0.08
Nodes (22): Bullet, 공격과 충돌, 관련 함수, 역할과 접근 경로, 이동과 수명, 찾아보기, 발사 상태, 월드와 개체 목록 (+14 more)

### Community 9 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 10 - "combat_bootstrap.py"
Cohesion: 0.12
Nodes (10): bootstrap(), demonstration_action(), evasive_action(), refine(), GrowthFeatures, CombatFeatures, TacticalCombatFeatures, ThreatCombatFeatures (+2 more)

### Community 11 - "SpriteAssets"
Cohesion: 0.09
Nodes (4): _health_mask(), _recolor_health(), _redraw_barrel(), SpriteAssets

### Community 12 - "PlayerProgression"
Cohesion: 0.07
Nodes (15): PlayerProgression, 관련 동작과 객체, 역할과 접근 경로, 일반 강화, 점수와 레벨, 진화와 숙련, 찾아보기, 강화와 진화 (+7 more)

### Community 13 - "diagnose_walls.py"
Cohesion: 0.11
Nodes (15): BulletObserver, capture_frame(), choose_clips(), encode_json(), expected_wall_grid(), main(), write_record(), replay_and_capture() (+7 more)

### Community 14 - "hashlib"
Cohesion: 0.07
Nodes (12): RotationContinuousActionDecoder, check_evolution_application(), main(), RotationTrainingEnvironment, configure_worker(), evaluate_checkpoint(), main(), main() (+4 more)

### Community 15 - "TileMap"
Cohesion: 0.11
Nodes (3): 관련 동작과 객체, TileMap, TilePosition

### Community 16 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 17 - "RewardConfig"
Cohesion: 0.05
Nodes (16): RewardConfig, empty_events(), CombatEvents, TankEnv, make_environment(), OutcomeRewardEnvironment, PixelObservation, VectorObservation (+8 more)

### Community 19 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 20 - "empty_game"
Cohesion: 0.18
Nodes (3): empty_game(), OwnershipTests, ProgressionTests

### Community 21 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 22 - "wooki02.py"
Cohesion: 0.16
Nodes (13): checkpoint_path(), learning_rate(), evaluate(), load_model(), make_env(), make_model(), one_env(), penalty_components() (+5 more)

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

### Community 28 - "game/terrain.py"
Cohesion: 0.11
Nodes (12): map_tests(), _clear_outer_wall_band(), expand_map(), _generate_maze(), generate_random_map(), grid_channels(), list_maps(), load_map() (+4 more)

### Community 29 - "ActionDecoder"
Cohesion: 0.05
Nodes (6): ActionDecoder, FactorizedActionDecoder, RotationContinuousActionDecoder, PlayerControl, RotationContinuousPolicy, RotationHybridDistribution

### Community 30 - "run_session.py"
Cohesion: 0.16
Nodes (9): json_default(), ModelRegistry, prepare_session(), RunSession, save_checkpoint(), save_configuration(), TrainingRunRecord, TrainingRunRecordUpdates (+1 more)

### Community 31 - "GameApplication"
Cohesion: 0.06
Nodes (15): 메뉴와 표시 상태, 시간과 발사 입력, 설정 변경 주체, evolve(), gain_level(), main(), capture(), GameApplication (+7 more)

### Community 32 - "SmallHybridDistribution"
Cohesion: 0.10
Nodes (9): SmallHybridDistribution, SmallHybridPolicy, Schedule은 PyTorch와 어떤 관계인가?, 객체 소유 구조, 실제 SB3 PPO 조립 연습, 실행, 이 예제에서 생략한 요소, 차원과 호출 위치 (+1 more)

### Community 33 - ".get_menu_buttons"
Cohesion: 0.16
Nodes (10): GameApplication, 관련 동작과 객체, 관련 함수, 스탯 창 상태, 역할과 접근 경로, 찾아보기, 이름과 경로 변경, debug_lines() (+2 more)

### Community 34 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 35 - "dashboard.py"
Cohesion: 0.14
Nodes (6): Demo, draw_panel(), duration(), main(), run_dashboard(), UI

### Community 36 - "gpt_training/environment.py"
Cohesion: 0.11
Nodes (29): main(), screenshot(), main(), EnvironmentSettings, game_configuration(), growth_choices(), make_environment(), teacher_action() (+21 more)

### Community 38 - "프로젝트 구조"
Cohesion: 0.29
Nodes (7): 사람이 플레이하는 경로, 실제 소스 배치, 처음 읽는 순서, 폴더별 책임, 프로젝트 구조, 학습 환경 경로, 한 프레임의 개요

### Community 39 - "TileMap"
Cohesion: 0.10
Nodes (3): resource_path(), AssetRefreshTests, step_with_score_award()

### Community 40 - "random_agent_test.py"
Cohesion: 0.39
Nodes (6): check_not_stuck(), determinism_test(), main(), make_env(), rollout(), run_random()

### Community 42 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 43 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 44 - "numpy"
Cohesion: 0.05
Nodes (12): encode_action(), move_from_vector(), wrap_angle(), AITarget, PendingShot, ActionDecoder, build_progression(), ship_one_hot() (+4 more)

### Community 45 - "Renderer"
Cohesion: 0.10
Nodes (6): 관련 함수, 화면과 카메라, 관련 함수와 읽는 객체, Renderer, Ghost, CenteredCameraTests

### Community 46 - "ScoreReward"
Cohesion: 0.17
Nodes (6): main(), measure(), main(), reward_settings(), ScoreReward, ScoreRewardSettings

### Community 47 - "environments.py"
Cohesion: 0.12
Nodes (8): ControlRecord, EpisodeStepRecord, PlayerSnapshot, append_record(), reward_components(), RewardBreakdown, StepLogRecord, TrainingLog

### Community 48 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 49 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 50 - "점수 개선 실험"
Cohesion: 0.07
Nodes (30): 006: 구조화 관찰과 가까운 명중 연습, 006 완료와 영상 확인, 007 완료: 연습의 개선이 실전에 전이되지 않음, 007: 움직이는 비발사 적을 상대하는 명중 연습, 008: 기본 게임 직접 학습과 발사·위험 보상, 008 중단과 계산 검증, 009: 발사하는 적과 조준 전용 잔차 신경망, 009 완료 (+22 more)

### Community 51 - "관련 함수"
Cohesion: 0.14
Nodes (4): 지형과 이미지, _hp_color(), _mix(), RGB

### Community 52 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 53 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.10
Nodes (20): 1. 전체 흐름과 현재 모델, 3. 무엇을 어디에서 바꾸었는가, 4. 실제로 어떤 순서로 작업하고 검증했는가, 5. 이 보고서는 어떤 전문가 방식을 참고해 구성했는가, 6. 직접 확인하는 가장 짧은 경로, 7. 상세 설정 원장, PPO와 실행 주기, 관찰과 신경망 (+12 more)

### Community 57 - "FullGrowthTests"
Cohesion: 0.17
Nodes (3): legal_growth_masks(), LearnedGrowthPolicy, FullGrowthTests

### Community 60 - "record_milestones.py"
Cohesion: 0.17
Nodes (10): edit(), main(), main(), prepare(), publish(), read_json(), record_clip(), save_manifest() (+2 more)

### Community 61 - "실험 기록"
Cohesion: 0.12
Nodes (16): 001_imitation_combat, 002_full_ppo, 003_dagger_full, 004 결과와 다음 단계, 005 결과와 최종 확인, 가설과 조건, 관측 결과, 관측 결과 (+8 more)

### Community 62 - "FullGrowthPolicy"
Cohesion: 0.08
Nodes (7): FullGrowthPolicy, CrowdPolicy, ExploringPolicy, GuardedPolicy, KitingPolicy, PathfindingPolicy, RecoveryPolicy

### Community 63 - "StatsPanelTests"
Cohesion: 0.15
Nodes (3): build_player_stat_summary(), StateOwnershipTests, StatsPanelTests

### Community 64 - "stable_baselines3_common_vec_env"
Cohesion: 0.18
Nodes (3): learning_rate(), main(), ProgressCallback

### Community 65 - "numpy"
Cohesion: 0.07
Nodes (6): EntityEncoder, SpatialFeaturesExtractor, main(), entity_encoder(), VectorFeaturesExtractor, SmallFeaturesExtractor

### Community 66 - "Space Z 실시간 강화학습"
Cohesion: 0.25
Nodes (7): Space Z 실시간 강화학습, 난이도와 평가, 명령줄과 DQN 비교, 보상, 선택한 모델, 실행, 화면과 파일

### Community 67 - "TrainingEnv"
Cohesion: 0.14
Nodes (5): CornerRecovery, escape_direction(), TrainingEnv, encode_growth_action(), stage_for_step()

### Community 68 - ".record_player_kill"
Cohesion: 0.20
Nodes (8): 역할과 접근 경로, GameCore, 관련 동작과 객체, 담당 객체와 설정, 시간과 전투 이벤트, 역할과 접근 경로, 점수 획득 기록, 찾아보기

### Community 69 - "DiscreteActionDecoder"
Cohesion: 0.05
Nodes (14): 강화학습 환경, 벡터 관찰, 보상과 info, 평가와 자원 측정, 픽셀 관찰, 행동, 환경 만들기, ContinuousActionDecoder (+6 more)

### Community 70 - "ObsConfig"
Cohesion: 0.10
Nodes (5): build_stat_upgrade_rows(), get_evolution_card_rects(), get_expanded_stat_panel_rect(), RectBounds, DisplayAssetsTests

### Community 71 - "게임 이미지 에셋"
Cohesion: 0.33
Nodes (4): v2 이미지 보관본, 게임 이미지 에셋, 그 외 에셋, 진화 트리

### Community 72 - "runtime.py"
Cohesion: 0.16
Nodes (20): migrate_combat(), main(), atomic_json(), base_vec(), _clean(), legacy_checkpoint_modules(), load_checkpoint(), make_model() (+12 more)

### Community 74 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

### Community 76 - "004_parallel_continuation/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 77 - "training_hurwan/train.py"
Cohesion: 0.24
Nodes (4): make_training_environments(), session_status(), configuration_path(), main()

### Community 78 - "protocol"
Cohesion: 0.13
Nodes (14): clips, created, protocol, camera_zoom, caveats, decoder, deterministic, encoded_size (+6 more)

### Community 79 - "shared_source_sha256"
Cohesion: 0.06
Nodes (31): shared_source_sha256, src\\space_z\\config\\enemy_ai.py, src\\space_z\\config\\game.py, src\\space_z\\config\\__init__.py, src\\space_z\\config\\learning.py, src\\space_z\\config\\progression_definitions.py, src\\space_z\\config\\terrain.py, src\\space_z\\game\\actions.py (+23 more)

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

### Community 97 - "combat.py"
Cohesion: 0.08
Nodes (17): Combat, 관련 동작과 객체, 관련 함수, 재장전, 찾아보기, 피해 적용, 한 프레임의 순서, 시작 조건과 참여 객체 (+9 more)

### Community 98 - "점수 획득 속도와 안정성 개선"
Cohesion: 0.12
Nodes (17): 015: 같은 보상으로 추가 학습, 015 시작 평가, 015 완료: 채택하지 않음, 015 첫 추가 구간: 빠르지만 불안정, 017: 이동·성장을 보존하고 실제 전투의 조준만 미세 조정, 017 종료와 개발 16시드 종합 선택, 2프레임 주기 개발 결과, 개발 10시드의 초기 결과 (+9 more)

### Community 100 - "연속 회전 PPO 실험 기록"
Cohesion: 0.15
Nodes (13): 001: 조준 적응, 첫 100,000스텝, 002: 기본 게임 적응과 계획 수정, 003: 이동 적응과 재생 방식 진단, 004: 좋은 조준을 보존한 몸체 학습 연장, 005: 몸체를 보존한 조준 정밀화, 구현 경계, 목적과 조건, 사전 검증 (+5 more)

### Community 101 - "trainer.py"
Cohesion: 0.15
Nodes (8): ExperimentConfig, make_environment(), build_policy_kwargs(), create_model(), evaluate_checkpoint(), LoggedPPO, train_with_checkpoints(), validate_policy_configuration()

### Community 102 - "pathlib"
Cohesion: 0.13
Nodes (5): main(), verify_bundle(), digest(), main(), read()

### Community 107 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 109 - "gpt_training_rotation_continuous/report.py"
Cohesion: 0.20
Nodes (6): copy_dependency_licenses(), compare(), main(), merge_development_evaluation(), summarize(), main()

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
Cohesion: 0.06
Nodes (23): 설정과 종류별 정의, 바꾸려는 기능으로 찾기, 적 AI와 월드, 맵과 지형, 강화학습 안내, EnemyAIConfig, GameConfig, EnvConfig (+15 more)

### Community 117 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 118 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 119 - "연속 회전 PPO 실험 결과"
Cohesion: 0.18
Nodes (8): 기록과 재현, 실행 환경, 연속 회전 PPO, 실행 범위와 자원 제한 위반, 연속 회전 PPO 실험 결과, 진화 오류와 결과의 한계, 최종 독립 평가, 학습 과정과 최종 모델

### Community 124 - "002_full_game_pilot/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 125 - "configuration.py"
Cohesion: 0.11
Nodes (11): ActionSettings, configuration_to_dict(), evaluation_to_dict(), EvaluationConfig, FeatureExtractorSettings, game_configuration(), merge_settings(), ObservationSettings (+3 more)

### Community 126 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 127 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 128 - "선정 모델 재생 묶음"
Cohesion: 0.33
Nodes (5): Windows에서 설치·실행, 다른 컴퓨터에 필요한 범위, 무엇이 들어 있는가, 선정 모델 재생 묶음, 터미널 설치와 검증

### Community 129 - "raw/gpt_training_vector_only/experiments/002_full_game_pilot/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 130 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 131 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 133 - "008_combined_reward_and_lr/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 134 - "TrainSettings"
Cohesion: 0.19
Nodes (5): TrainSettings, main(), DashboardTests, TrainingJobTests, RuntimeTests

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

### Community 142 - "raw/gpt_training_vector_only/experiments/003_training_duration/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

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
Cohesion: 0.17
Nodes (11): 005 이후 작은 수정이 필요할 경우, 005 진행 중 · 사용자 후속 승인, 009 실행 계획 · 2026-11-03, 검증과 제약, 명령, 범위와 유지 사항, 사용자 선호, 실행 순서와 결과 (+3 more)

### Community 150 - "GameCore"
Cohesion: 0.08
Nodes (6): GameCore, ExpandedWorldTests, ShapeSpawnTests, StatPanelCollapseTests, MinimapInputTests, RendererTests

### Community 151 - "raw/gpt_training_vector_only/experiments/004_parallel_continuation/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 152 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 154 - "raw/gpt_training_vector_only/experiments/005_same_settings_continuation/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 156 - "raw/gpt_training_vector_only/experiments/006_lower_learning_rate/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 157 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

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

### Community 163 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 164 - "raw/gpt_training_vector_only/experiments/007_higher_combat_reward/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 165 - "raw/gpt_training_vector_only/experiments/008_combined_reward_and_lr/manifest.json"
Cohesion: 0.15
Nodes (12): architecture_sha256, config_sha256, created_at, distribution_dependency, goal, id, initialization, input_contract (+4 more)

### Community 168 - ".step"
Cohesion: 0.13
Nodes (7): Tank, 무장과 탄약, 역할과 접근 경로, 위치와 이동, 찾아보기, 체력 상태, 한 step의 순서

### Community 169 - "train_v2.py"
Cohesion: 0.05
Nodes (33): append_record(), configuration_path(), configuration_to_dict(), ControlRecord, create_model(), EpisodeStepRecord, evaluate_checkpoint(), evaluation_to_dict() (+25 more)

### Community 170 - ".pop_score_award_events"
Cohesion: 0.10
Nodes (8): Renderer, 게임 상태와 화면 연출의 차이, 관련 동작과 객체, 역할과 접근 경로, 점수 팝업, 찾아보기, 바뀌는 상태와 소비자, ScoreAward

### Community 173 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 175 - "영상 체크포인트 20개 시드 검증"
Cohesion: 0.40
Nodes (4): 생존 시간 (초), 영상 체크포인트 20개 시드 검증, 파일, 획득 점수

### Community 177 - "argparse"
Cohesion: 0.17
Nodes (3): main(), RejectOtherRLFolders, main()

### Community 178 - "2. 어떤 문제를 보고 다음 실험을 선택했는가"
Cohesion: 0.22
Nodes (9): 2.1 처음에는 모방학습을 사용했고, 요청에 따라 폐기하지 않고 분리 보존했다, 2.2 순수 PPO를 실행하는 것과 게임을 잘하는 것은 달랐다, 2.3 CNN과 보상을 추가했지만, 쉬운 연습 성적은 실전에 그대로 옮겨지지 않았다, 2.4 조준 입력과 이동 학습을 분리하면서 실전 점수가 올라가기 시작했다, 2.5 더 학습했더니 초반 평균만 좋아져, 보상 비중과 행동 주기를 다시 봤다, 2.6 정상 게임에서도 추가 학습이 항상 좋은 것은 아니었다, 2.7 연속 회전은 200만 스텝을 5개 구간으로 나누고, 조준·몸체를 번갈아 적응시켰다, 2.8 실제 화면에서 발견한 진화 오류로 최종 평가를 다시 했다 (+1 more)

### Community 179 - "016: 보조 보상 축소와 실제 점수·생존 비중 확대"
Cohesion: 0.40
Nodes (5): 016 두 번째 구간과 종료, 016: 보조 보상 축소와 실제 점수·생존 비중 확대, 016 시작의 성장 선택 확인, 016 첫 구간, 추론 방식과 실패 장면 확인

### Community 180 - "평가 결과 모음"
Cohesion: 0.25
Nodes (7): 수집 범위, 이미지 다시 생성, 이미지 읽는 방법, 통합 JSON, 평가 결과 모음, 해석에 필요한 예외, 확인할 파일

### Community 182 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 183 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 184 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 185 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 186 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 187 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 188 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 189 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 190 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 191 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 192 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 193 - "architecture"
Cohesion: 0.29
Nodes (7): architecture, action_parameters, blocks, context_tokens, heads, name, width

### Community 194 - "parent"
Cohesion: 0.29
Nodes (7): parent, checkpoint, environment_reset_on_resume, experiment_id, optimizer_preserved, sha256, timesteps

### Community 195 - "추가 학습 결과: 고득점 후보 보존, 기본 모델 유지"
Cohesion: 0.40
Nodes (5): 고득점 실험 후보의 새 20시드 비교, 기록과 검증, 실제로 진행한 것, 직접 재생, 추가 학습 결과: 고득점 후보 보존, 기본 모델 유지

### Community 197 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 198 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 199 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 200 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 201 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 202 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 203 - "versions"
Cohesion: 0.40
Nodes (5): versions, gymnasium, python, stable_baselines3, torch

### Community 204 - "순수 PPO 점수 개선 결과"
Cohesion: 0.50
Nodes (4): 모델 선택 후 새 시드 10개에서 비교, 순수 PPO 점수 개선 결과, 실험과 검증 기록, 재생과 TensorBoard

### Community 206 - "전체 성장 강화학습과 검증"
Cohesion: 0.50
Nodes (3): 검증, 실행, 전체 성장 강화학습과 검증

## Knowledge Gaps
- **1 isolated node(s):** `space-z`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1937 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **50 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GameConfig` connect `GameConfig` to `experiment.py`, `config/__init__.py`, `Shape`, `Tank`, `WeaponsSkillsTests`, `combat_bootstrap.py`, `RewardConfig`, `.test_mostly_avoids_a_tight_corner_that_leaves_no_room_to_orbit`, `empty_game`, `GameCore`, `open_game`, `GameApplication`, `.get_menu_buttons`, `gpt_training/environment.py`, `TileMap`, `random_agent_test.py`, `train_v2.py`, `MenusReloadTests`, `.step`, `numpy`, `Renderer`, `점수 개선 실험`, `관련 함수`, `2. 어떤 문제를 보고 다음 실험을 선택했는가`, `DifficultyMenuTests`, `MinimapTests`, `record_milestones.py`, `StatsPanelTests`, `.record_player_kill`, `DiscreteActionDecoder`, `ObsConfig`, `전체 성장 강화학습과 검증`, `SmarterEnemyAITests`, `configuration.py`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Are the 85 inferred relationships involving `GameCore` (e.g. with `스탯 창 상태` and `바뀌는 상태와 소비자`) actually correct?**
  _`GameCore` has 85 INFERRED edges - model-reasoned connections that need verification._
- **What connects `space-z` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.05303030303030303 - nodes in this community are weakly interconnected._
- **Why does `GameCore` connect `GameCore` to `config/__init__.py`, `EnemyController`, `Shape`, `Tank`, `WeaponsSkillsTests`, `PlayerProgression`, `WorldPoint`, `TileMap`, `RewardConfig`, `.test_mostly_avoids_a_tight_corner_that_leaves_no_room_to_orbit`, `empty_game`, `VectorObservation`, `open_game`, `GameApplication`, `.get_menu_buttons`, `프로젝트 구조`, `TileMap`, `.step`, `MenusReloadTests`, `.pop_score_award_events`, `numpy`, `Renderer`, `관련 함수`, `VectorObservation`, `DifficultyMenuTests`, `MinimapTests`, `StatsPanelTests`, `.record_player_kill`, `DiscreteActionDecoder`, `ObsConfig`, `.build`, `ObservationBuilder`, `combat.py`, `SmarterEnemyAITests`, `GameConfig`, `.build`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `Renderer` (e.g. with `공격과 충돌` and `스탯 창 상태`) actually correct?**
  _`Renderer` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Should `experiment.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05956112852664577 - nodes in this community are weakly interconnected._