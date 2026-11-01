"""게임 규칙을 유지하면서 전술 관찰·분리된 행동·학습 보상을 구성한다."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import gymnasium as gym
import numpy as np

from space_z.config import EnvConfig, GameConfig, RewardConfig, TerrainConfig
from space_z.config.progression_definitions import SHIPS, STAT_UPGRADES
from space_z.game.actions import MOVE_VECTORS, wrap_angle
from space_z.rl import DiscreteActionDecoder, RewardCalculator, TankEnv
from space_z.rl.actions.action_decoder import ActionDecoder
from space_z.rl.observations.observation_builder import ObservationBuilder
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.observations.progression_observation import SHIP_KEYS, build_progression
from .score_reward import ScoreReward, reward_settings

ROTATIONS = np.array([-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1], dtype=np.float32)
PROGRESSION = slice(10, 133)
TARGET = slice(133, 142)
MOVEMENT = slice(142, 151)
RAYS = slice(151, 159)
OBSERVATION_SIZE = 164


@dataclass(frozen=True)
class EnvironmentSettings:
    """학습 난이도와 에피소드 길이. full은 기본 게임 규칙을 사용한다."""

    stage: str = "full"
    max_steps: int = 3000
    frame_skip: int = 4
    pure_rl: bool = False
    reward_profile: str = 'legacy'


def game_configuration(stage: str) -> GameConfig:
    if stage in ('range', 'moving_range'):
        return GameConfig(world_scale=2, world_area_scale=1.0, num_enemies=0, num_shapes=1,
                          shape_drift=0, shape_spawn_candidates=1, shape_respawn_dist_from_player=220,
                          terrain=TerrainConfig(enabled=False))
    if stage == 'aim':
        return GameConfig(world_scale=2, world_area_scale=1.0, num_enemies=0, num_shapes=18,
                          shape_respawn_dist_from_player=250, shape_spawn_candidates=3,
                          terrain=TerrainConfig(enabled=False))
    if stage == 'duel':
        configuration = GameConfig(world_scale=3, world_area_scale=1.0, num_enemies=2, num_shapes=30,
                                   shape_respawn_dist_from_player=300, terrain=TerrainConfig(enabled=False))
        configuration.enemy_ai = replace(configuration.enemy_ai, fire_angle_tol_deg=-1)
        return configuration
    if stage == 'arena':
        configuration = GameConfig(world_scale=3, world_area_scale=1.0, num_enemies=3, num_shapes=18,
                                   shape_respawn_dist_from_player=300, terrain=TerrainConfig(enabled=False))
        configuration.enemy_ai = replace(configuration.enemy_ai, sight_range=1500, hearing_range=3000)
        return configuration
    if stage == "full":
        return GameConfig()
    if stage == "combat":
        return GameConfig(num_enemies=3, num_shapes=40, terrain=TerrainConfig(enabled=False))
    if stage == "harvest":
        return GameConfig(num_enemies=0, num_shapes=45, terrain=TerrainConfig(enabled=False))
    raise ValueError(f"Unknown curriculum stage: {stage}")


class FactorizedActions(ActionDecoder):
    """이동·회전·발사와 선택적인 성장 요청을 독립적으로 해석한다."""

    def __init__(self, enable_growth=False, lock_movement=False) -> None:
        self.lock_movement = lock_movement
        self._base_decoder = DiscreteActionDecoder(
            enable_stat_upgrades=enable_growth, enable_evolution=enable_growth,
        )
        self.stat_count = len(STAT_UPGRADES) + 1 if enable_growth else 1
        self.evolution_count = self._base_decoder.action_space.n // (9 * 3 * 2 * self.stat_count)
        branches = [9, len(ROTATIONS), 2]
        if enable_growth:
            branches.extend([self.stat_count, self.evolution_count])
        self._space = gym.spaces.MultiDiscrete(branches)

    @property
    def action_space(self):
        return self._space

    def decode(self, action):
        action = np.asarray(action)
        if not self._space.contains(action):
            raise ValueError(f"Invalid factorized action: {action}")
        move, rotation, fire = map(int, action[:3])
        stat, evolution = map(int, action[3:]) if len(action) == 5 else (0, 0)
        encoded_action = (((move * 3) * 2 + fire) * self.stat_count + stat) * self.evolution_count + evolution
        control = self._base_decoder.decode(encoded_action)
        control.rotation = float(ROTATIONS[rotation])
        if self.lock_movement:
            control.move_x = 0.0
            control.move_y = 0.0
        return control


class TacticalObservation(ObservationBuilder):
    """현재 상태에서 계산한 조준 오차·회피 방향·성장 정보를 제공한다. 미래 상태는 사용하지 않는다."""

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, (OBSERVATION_SIZE,), dtype=np.float32)

    def reset(self):
        pass

    def close(self):
        pass

    def build(self, game):
        player = game.player
        result = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        weapon, damage, cooldown, bullet_speed, bullet_life = game.combat.weapon_values(player)
        result[:10] = (
            player.velocity_x / max(game.config.max_speed, 0.1),
            player.velocity_y / max(game.config.max_speed, 0.1),
            player.hp / max(player.max_hp, 1), math.sin(player.angle), math.cos(player.angle),
            player.cooldown / max(cooldown, 1), player.radius / 100,
            min(bullet_speed / 30, 1), min(bullet_life / 200, 1), min(damage / 100, 1),
        )
        result[PROGRESSION] = build_progression(game)

        candidates = []
        enemies = [enemy for enemy in game.enemies if enemy.alive]
        for entity, is_enemy in [(shape, False) for shape in game.shapes] + [(enemy, True) for enemy in enemies]:
            distance = math.hypot(entity.x - player.x, entity.y - player.y)
            if distance > 1200 or not game.terrain.line_of_sight(player.x, player.y, entity.x, entity.y):
                continue
            # 적은 가까이 있을 때 우선하되 먼 적을 추격하며 근처 천체를 모두 무시하지 않는다.
            priority_distance = distance * (0.65 if is_enemy else 1)
            candidates.append((priority_distance, distance, entity, is_enemy))
        desired_movement = np.zeros(2, dtype=np.float64)
        if candidates:
            _, distance, target, is_enemy = min(candidates, key=lambda value: value[0])
            lead_frames = min(60, distance / max(bullet_speed, 0.1))
            aim_x = target.x + target.velocity_x * lead_frames - player.x
            aim_y = target.y + target.velocity_y * lead_frames - player.y
            angle_error = wrap_angle(math.atan2(aim_y, aim_x) - player.angle)
            result[TARGET] = (
                1, aim_x / 1200, aim_y / 1200, distance / 1200, angle_error / math.pi,
                target.hp / max(target.max_hp, 1), float(is_enemy),
                float(distance <= bullet_speed * bullet_life + player.radius), target.radius / 1200,
            )
            direction = np.array([target.x - player.x, target.y - player.y]) / max(distance, 1)
            preferred_distance = 480 if is_enemy else 220
            approach = np.clip((distance - preferred_distance) / 180, -1, 1)
            desired_movement += direction * approach
            if is_enemy:
                desired_movement += np.array([-direction[1], direction[0]]) * 0.35
        else:
            desired_movement += [math.cos(game.frame / 240), math.sin(game.frame / 240)]

        nearest_enemy = min(enemies, key=lambda enemy: math.hypot(enemy.x-player.x, enemy.y-player.y), default=None)
        if nearest_enemy is not None:
            delta = np.array([player.x-nearest_enemy.x, player.y-nearest_enemy.y])
            distance = float(np.linalg.norm(delta))
            desired_movement += delta / max(distance, 1) * max(0, (430-distance)/180)
            result[159:162] = (delta[0]/1200, delta[1]/1200, min(distance/1200, 1))

        threat_count = 0
        for bullet in game.combat.bullets:
            if bullet.team == player.team:
                continue
            relative = np.array([bullet.x-player.x, bullet.y-player.y])
            velocity = np.array([bullet.velocity_x-player.velocity_x, bullet.velocity_y-player.velocity_y])
            time_to_closest = float(np.clip(-np.dot(relative, velocity)/max(np.dot(velocity, velocity), 1), 0, 18))
            closest = relative + velocity * time_to_closest
            distance = float(np.linalg.norm(closest))
            if distance < player.radius + bullet.radius + 90 and np.linalg.norm(relative) < 700:
                desired_movement -= closest / max(distance, 1) * min(2, 120/max(distance, 20))
                threat_count += 1
        result[162] = min(threat_count/5, 1)
        result[163] = min(game.frame/(60*300), 1)

        desired_movement /= max(1, float(np.linalg.norm(desired_movement)))
        result[MOVEMENT.start] = 0.15 if threat_count == 0 else -0.3
        for index, (direction_x, direction_y) in enumerate(MOVE_VECTORS[1:]):
            clearance = game.terrain.raycast(player.x, player.y, direction_x, direction_y, 320)
            result[RAYS.start+index] = clearance/320
            wall_penalty = max(0, (player.radius+100-clearance)/100)
            result[MOVEMENT.start+index+1] = np.dot(desired_movement, [direction_x, direction_y]) - wall_penalty * 2
        return np.clip(result, -1, 1)


def growth_choices(observation):
    """현재 관찰의 사용 가능 마스크로만 강화·진화 우선순위를 정한다."""
    progression = observation[PROGRESSION]
    levels = progression[25:30]
    available = progression[30:35] > 0.5
    priorities = np.array([1.2, 3, 2, 3, 2.5]) / (1 + levels * 8)
    if observation[2] < 0.5:
        priorities[0] += 5
    priorities[~available] = -1
    stat = int(np.argmax(priorities)) + 1 if available.any() else 0
    evolution = 0
    best_score = -math.inf
    for slot in range(4):
        offset = 35 + slot * 22
        if progression[offset] < 0.5:
            continue
        key = SHIP_KEYS[int(np.argmax(progression[offset+1:offset+22]))]
        ship = SHIPS[key]
        score = ship.damage * ship.speed * ship.life / ship.cooldown + ship.hp_bonus / 400
        if score > best_score:
            evolution, best_score = slot + 1, score
    return stat, evolution


def teacher_action(observation):
    movement = int(np.argmax(observation[MOVEMENT]))
    target = observation[TARGET]
    angle_error = float(target[4]) * math.pi
    rotation = int(np.argmin(np.abs(ROTATIONS - np.clip(angle_error / 0.28, -1, 1)))) if target[0] else 6
    tolerance = max(0.06, math.atan2(float(target[8]), max(float(target[3]), 0.01)))
    fire = int(target[0] > 0.5 and target[7] > 0.5 and abs(angle_error) <= tolerance)
    return np.array([movement, rotation, fire], dtype=np.int64)


class TrainingEnvironment(gym.Wrapper):
    """공통 게임에 학습 보상을 연결한다. 순수 RL에서는 성장도 정책이 선택한다."""

    def __init__(self, settings: EnvironmentSettings):
        if settings.reward_profile not in ('legacy', 'score_v2', 'score_v3', 'score_v4', 'score_v5', 'score_v6', 'score_v7'):
            raise ValueError(f'Unknown reward profile: {settings.reward_profile}')
        game_config = game_configuration(settings.stage)
        observation_builder = VectorObservation() if settings.pure_rl else TacticalObservation()
        action_decoder = FactorizedActions(enable_growth=settings.pure_rl, lock_movement=settings.stage in ('range', 'moving_range'))
        reward_config = RewardConfig()
        reward_calculator = RewardCalculator(reward_config)
        env_config = EnvConfig(frame_skip=settings.frame_skip, max_steps=settings.max_steps)
        super().__init__(TankEnv(game_config, observation_builder, action_decoder, reward_calculator, env_config))
        self.settings = settings
        self.last_observation = None
        self.episode_totals = {}
        self.score_reward = ScoreReward(reward_settings(settings.reward_profile)) if settings.reward_profile != 'legacy' else None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.settings.stage in ('range', 'moving_range'):
            self._position_range_shape()
            observation = self.unwrapped.observation_builder.build(self.unwrapped.game)
        self.last_observation = observation
        self.episode_totals = {"training_reward": 0.0, "shots": 0, "shapes_destroyed": 0, "damage_taken": 0.0}
        if self.score_reward is not None:
            self.score_reward.reset(self.unwrapped.game, observation)
        return observation, info

    def _position_range_shape(self):
        """사격 연습의 새 천체를 플레이어 주변 무작위 각도·거리에 둔다."""
        game = self.unwrapped.game
        shape = game.shapes[0]
        for _ in range(100):
            angle = float(game.rng.uniform(-math.pi, math.pi))
            distance = float(game.rng.uniform(220, 480))
            x, y = game.player.x + math.cos(angle)*distance, game.player.y + math.sin(angle)*distance
            if game.terrain.circle_fits(x, y, shape.radius):
                shape.x, shape.y = x, y
                shape.velocity_x = shape.velocity_y = 0.0
                if self.settings.stage == 'moving_range':
                    shape.velocity_x, shape.velocity_y = map(float, game.rng.uniform(-3, 3, size=2))
                shape.hp = shape.max_hp = min(7.0, shape.max_hp)
                return
        raise RuntimeError('No valid firing-range shape position')

    def step(self, action):
        if self.last_observation is None:
            raise RuntimeError("reset is required before step")
        game = self.unwrapped.game
        if not self.settings.pure_rl:
            stat, evolution = growth_choices(self.last_observation)
            if stat:
                game.progression.upgrade_stat(STAT_UPGRADES[stat-1].key)
            if evolution:
                game.progression.choose_reward(evolution-1)
        previous_score = game.progression.score
        observation, _, terminated, truncated, info = self.env.step(action)
        events = info["events"]
        if self.settings.stage in ('range', 'moving_range'):
            shape = game.shapes[0]
            distance = math.hypot(shape.x-game.player.x, shape.y-game.player.y)
            out_of_range = self.settings.stage == 'moving_range' and not 180 <= distance <= 600
            if events['shapes_destroyed'] or out_of_range:
                self._position_range_shape()
                observation = self.unwrapped.observation_builder.build(game)
        score_gain = info["score"] - previous_score
        reward = (
            score_gain / 100 + events["shape_damage_weighted"] / 400
            + events["enemy_damage"] / 200
            - events["damage_taken"] / max(game.player.max_hp, 1) * 2
            - 3 * float(terminated) + 0.001
        )
        if self.score_reward is not None:
            components = self.score_reward.calculate(game, observation, score_gain, events, terminated)
            reward = sum(components.values())
            for name, value in components.items():
                self.episode_totals[name] = self.episode_totals.get(name, 0.0) + value
        self.episode_totals["training_reward"] += reward
        self.episode_totals["shots"] += events["player_shots_fired"]
        self.episode_totals["shapes_destroyed"] += events["shapes_destroyed"]
        self.episode_totals["damage_taken"] += events["damage_taken"]
        self.last_observation = observation
        if terminated or truncated:
            info["experiment_episode"] = dict(self.episode_totals)
        return observation, float(reward), terminated, truncated, info


def make_environment(stage="full", max_steps=3000, pure_rl=False, reward_profile='legacy', frame_skip=4):
    settings = EnvironmentSettings(stage=stage, max_steps=max_steps, pure_rl=pure_rl,
                                   reward_profile=reward_profile, frame_skip=frame_skip)
    return TrainingEnvironment(settings)
