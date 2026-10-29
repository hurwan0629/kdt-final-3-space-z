"""SB3 adapter for main observations, actions and rewards."""

# 게임과 SB3 사이의 연결: 관찰 분리, 성장 행동 변환, 에피소드 지표 집계.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from space_z.config import GameConfig, ObsConfig, EnvConfig, RewardConfig
from space_z.rl.environment import TankEnv
from .settings import stage_for_step
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.actions.discrete_action_decoder import DiscreteActionDecoder
from space_z.rl.rewards import RewardCalculator
from .corner_recovery import CornerRecovery


def game_config(stage):
    # 단계별 적 수·지형·AI 설정을 적용한 GameConfig를 반환한다.
    cfg = GameConfig(
        progression_enabled=False,
        limited_ammo=False,
        window_width=960,
        window_height=640,
    )
    if stage == 0:
        cfg.world_scale, cfg.world_area_scale = 3, 1.
        cfg.num_enemies, cfg.num_shapes = 3, 20
        cfg.terrain.enabled = False
        cfg.enemy_ai.aim_noise_deg = 24.
        cfg.enemy_ai.move_speed_scale = .65
    elif stage == 1:
        cfg.num_enemies = 6
    return cfg


class TrainingEnv(gym.Wrapper):
    """Adapt main observation arrays for SB3 without changing game interfaces."""

    def __init__(
        self,
        max_steps=3000,
        curriculum=True,
        evaluation=False,
        gamma=.995,
        score_objective=False,
        corner_recovery=False,
        full_growth=False,
    ):
        # 실행 옵션과 센서를 준비하고, 기본 TankEnv를 SB3용 관찰·행동 공간으로 감싼다.
        self.full_growth = full_growth
        self.score_objective = score_objective
        self.corner_recovery = corner_recovery
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.curriculum, self.evaluation, self.gamma = curriculum, evaluation, gamma
        self.training_step = 0
        self.stage = 2 if evaluation else stage_for_step(0, curriculum)
        self.sensor_cfg = ObsConfig()
        self.seed_rng = np.random.default_rng(42)
        super().__init__(self._new_env())

        # 성장을 켜면 전투·스탯·전직을 독립적인 세 선택으로 모델에 노출한다.
        if self.full_growth:
            from .growth import STAT_COUNT, EVOLUTION_COUNT

            self.action_space = spaces.MultiDiscrete([54, STAT_COUNT, EVOLUTION_COUNT])
        self.observation_space = spaces.Dict(
            {
                'vector': spaces.Box(
                    -1., 1., (self.sensor_cfg.vector_size,), np.float32
                ),
                'grid': spaces.Box(-1., 1., self.sensor_cfg.grid_shape(), np.float32),
            }
        )
        if self.corner_recovery:
            self.observation_space.spaces['recovery'] = spaces.Box(
                0., 1., (20,), np.float32
            )
        self._components = {}

    def _new_env(self):
        # 현재 단계와 성장 옵션으로 관찰·행동·보상 객체를 연결한 TankEnv를 만든다.
        self.builder = VectorObservation(self.sensor_cfg)
        self.reward_config = RewardConfig()
        cfg = game_config(self.stage)
        cfg.progression_enabled = self.full_growth
        return TankEnv(
            cfg,
            self.builder,
            DiscreteActionDecoder(
                enable_stat_upgrades=self.full_growth, enable_evolution=self.full_growth
            ),
            RewardCalculator(self.reward_config),
            env_config=EnvConfig(frame_skip=4, max_steps=self.max_steps),
        )

    def set_training_step(self, step):
        # 다음 reset에서 난이도를 선택할 때 사용할 전체 학습 진행량을 갱신한다.
        self.training_step = int(step)

    def _observation(self, obs):
        # 기본 관찰을 vector와 grid로 나누고, 옵션에 따라 탈출 상태를 붙인다.
        split = self.sensor_cfg.vector_size
        result = {
            'vector': obs[:split],
            'grid': obs[split:].reshape(self.sensor_cfg.grid_shape()),
        }
        if self.corner_recovery:
            result['recovery'] = self.recovery.observation()
        return result

    def reset(self, *, seed=None, options=None):
        # 새 에피소드 seed와 난이도를 정하고, 게임 및 에피소드 집계 상태를 초기화한다.
        if seed is not None:
            self.seed_rng = np.random.default_rng(seed)
        episode_seed = int(self.seed_rng.integers(0, 2**31 - 1))

        # 1) 난이도 변경은 에피소드 경계에서만 적용한다. 평가는 항상 마지막 단계다.
        wanted = (
            2
            if self.evaluation
            else stage_for_step(self.training_step, self.curriculum)
        )
        if wanted != self.stage:
            self.env.close()
            self.stage = wanted
            self.env = self._new_env()
        obs, info = self.env.reset(seed=episode_seed, options=options)
        self._last_main_obs = obs
        self.recovery = CornerRecovery((self.env.game.player.x, self.env.game.player.y))
        self.episode_seed = episode_seed

        # 2) 새 에피소드의 보상·전투·성장 집계를 0부터 시작한다.
        self.episode_return = self.task_return = 0.
        self.episode_damage = self.episode_hurt = 0.
        self.episode_shapes = 0
        self.crowded_steps = self.retreat_steps = 0
        self.crowded_hurt = 0.
        self.empty_steps = self.empty_idle_steps = 0
        self.stat_upgrades = 0
        self.evolutions = 0
        self.masteries = 0
        self.ship_history = [self.env.game.player.job]
        self._components = dict.fromkeys(
            ('damage', 'kills', 'shapes', 'hurt', 'death', 'time', 'shots'), 0.
        )
        info.update(episode_seed=episode_seed, stage=self.stage)
        return self._observation(obs), info

    def step(self, action):
        # 행동 변환 → 게임 진행 → 성장·보상 집계 → 종료 지표 작성 순서로 처리한다.
        from .growth import encode_growth_action

        # 1) 성장 행동을 기본 게임 입력으로 합치고, 진행 전 점수·성장 상태를 기억한다.
        main_action = encode_growth_action(action) if self.full_growth else action
        combat_action = int(action[0]) if self.full_growth else int(action)
        before = self._last_main_obs
        before_score = self.env.game.progression.score
        progression = self.env.game.progression
        before_stats = sum(progression.stat_levels.values())
        before_special = sum(progression.special_levels.values())
        before_ship = self.env.game.player.job
        cfg = self.sensor_cfg

        # 2) 진행 전 관찰로 목표 부재·적 밀집·후퇴 여부를 집계한다.
        enemies = before[self.builder.slices['enemies']].reshape(
            cfg.n_enemies, cfg.ENEMY_DIM
        )
        shapes = before[self.builder.slices['shapes']].reshape(
            cfg.n_shapes, cfg.SHAPE_DIM
        )
        if not enemies[:, 0].any() and not shapes[:, 0].any():
            self.empty_steps += 1
            self.empty_idle_steps += int(np.linalg.norm(before[:2]) < .1)
        scale = (
            np.array([cfg.grid_width, cfg.grid_height])
            * self.env.game_config.terrain.tile_size
            / 2
        )
        positions = enemies[:, 1:3] * scale
        near = (enemies[:, 0] > .5) & (np.linalg.norm(positions, axis=1) < 480)
        crowded = int(near.sum()) >= 2
        if crowded:
            self.crowded_steps += 1
            control = self.env.action_decoder.decode(main_action)
            toward = positions[near].sum(axis=0)
            if control.move_x * toward[0] + control.move_y * toward[1] < 0:
                self.retreat_steps += 1

        # 3) 게임을 한 step 진행한 뒤 스탯 강화·전직·숙련 변화량을 구분한다.
        obs, reward, terminated, truncated, info = self.env.step(main_action)
        self.stat_upgrades += sum(progression.stat_levels.values()) - before_stats
        changes = sum(progression.special_levels.values()) - before_special
        if self.env.game.player.job != before_ship:
            self.evolutions += changes
            self.ship_history.append(self.env.game.player.job)
        else:
            self.masteries += changes
        self._last_main_obs = obs
        if self.corner_recovery:
            ev = info['events']
            productive = (
                ev['enemy_damage'] > 0
                or ev['shape_damage_weighted'] > 0
                or info['score'] > before_score
            )
            self.recovery.update(
                self.env.game,
                obs[self.builder.slices['grid']].reshape(self.sensor_cfg.grid_shape())[
                    0
                ],
                attempted=combat_action // 6 != 0,
                productive=productive,
            )

        # 4) 이번 step의 전투 이벤트를 보상 항목으로 분해한다.
        events, cfg = info['events'], self.reward_config
        if crowded:
            self.crowded_hurt += events['damage_taken']
        parts = {
            'damage': cfg.enemy_damage * events['enemy_damage'],
            'kills': cfg.enemy_kill * events['enemies_killed'],
            'shapes': cfg.shape_damage * events['shape_damage_weighted']
            + cfg.shape_destroy * events['shape_destroyed_score'],
            'hurt': cfg.damage_taken * events['damage_taken'],
            'death': cfg.death if events['died'] else 0.,
            'time': cfg.time_step,
            'shots': -cfg.shot_cost * events['player_shots_fired'],
        }

        # 점수 목표에서는 기본 보상 대신 점수 증가분과 사망 벌점을 사용한다.
        if self.score_objective:
            parts = {
                'game_score': (info['score'] - before_score) / 200.,
                'death': -.5 if terminated else 0.,
            }
            reward = float(sum(parts.values()))

        # 5) 반환 보상과 에피소드 누적값을 갱신하고 info에 기록한다.
        task_reward = reward
        self.episode_return += reward
        self.task_return += task_reward
        self.episode_damage += info['events']['enemy_damage']
        self.episode_hurt += info['events']['damage_taken']
        self.episode_shapes += info['events']['shapes_destroyed']
        for key, value in parts.items():
            self._components[key] = self._components.get(key, 0.) + value
        info.update(
            stage=self.stage,
            episode_seed=self.episode_seed,
            reward_components=parts,
            task_reward=task_reward,
        )

        # 6) 종료 시에만 한 에피소드의 최종 지표 묶음을 작성한다.
        if terminated or truncated:
            info['episode_metrics'] = {
                'return': self.episode_return,
                'task_return': self.task_return,
                'kills': int(info['kills']),
                'score': int(info['score']),
                'damage': self.episode_damage,
                'hurt': self.episode_hurt,
                'shapes': int(self.episode_shapes),
                'steps': int(info['steps']),
                'seconds': info['frames'] / self.env.game_config.fps,
                'death': int(terminated),
                'stage': self.stage,
                'seed': self.episode_seed,
                'components': dict(self._components),
                'crowded_steps': self.crowded_steps,
                'retreat_steps': self.retreat_steps,
                'crowded_hurt': self.crowded_hurt,
                'empty_steps': self.empty_steps,
                'empty_idle_steps': self.empty_idle_steps,
                'corner_recoveries': self.recovery.count,
                'level': progression.level,
                'stat_upgrades': self.stat_upgrades,
                'evolutions': self.evolutions,
                'masteries': self.masteries,
                'stat_levels': dict(progression.stat_levels),
                'ship_history': list(self.ship_history),
                'upgrade_points': progression.upgrade_points,
                'pending_evolutions': len(progression.special_reward_levels),
            }
        return self._observation(obs), reward, terminated, truncated, info
