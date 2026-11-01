"""게임 점수와 실제 명중을 중심으로 학습 보상을 구성한다."""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class ScoreRewardSettings:
    """이벤트별 보상 배율과 조준·접근 진전 보상의 크기."""

    score_scale: float = 0.02
    shape_damage_scale: float = 0.03
    enemy_damage_scale: float = 0.015
    damage_fraction_penalty: float = 0.5
    death_penalty: float = 2.0
    shot_penalty: float = 0.001
    potential_scale: float = 0.8
    gamma: float = 0.995
    aimed_shot_bonus: float = 0.0
    projectile_threat_penalty: float = 0.0
    focus_nearest: bool = False


def reward_settings(profile):
    if profile == 'score_v2':
        return ScoreRewardSettings()
    if profile == 'score_v3':
        return replace(ScoreRewardSettings(), enemy_damage_scale=0.05, damage_fraction_penalty=2.0,
                       death_penalty=3.0, aimed_shot_bonus=0.15, projectile_threat_penalty=0.02)
    if profile == 'score_v4':
        return replace(reward_settings('score_v3'), focus_nearest=True)
    if profile == 'score_v5':
        return replace(reward_settings('score_v4'), potential_scale=0.2)
    if profile == 'score_v6':
        return replace(reward_settings('score_v5'), shape_damage_scale=0.01, enemy_damage_scale=0.01,
                       aimed_shot_bonus=0.02, potential_scale=0.1, damage_fraction_penalty=3.0,
                       death_penalty=6.0, projectile_threat_penalty=0.03)
    if profile == 'score_v7':
        # 2프레임 행동 두 번의 할인을 기존 4프레임 행동 한 번과 맞춘다.
        return replace(reward_settings('score_v6'), gamma=math.sqrt(0.995))
    raise ValueError(f'Unknown score reward profile: {profile}')


class ScoreReward:
    """직전 상태의 potential을 보관해 정렬·접근의 변화에만 보상을 준다."""

    def __init__(self, settings=None):
        self.settings = settings or ScoreRewardSettings()
        self.previous_potential = 0.0

    def potential(self, game, observation):
        player = game.player
        _, _, _, bullet_speed, bullet_life = game.combat.weapon_values(player)
        effective_range = bullet_speed * bullet_life
        entities = np.concatenate([
            observation[133:412].reshape(9, 31)[:, :3],
            observation[412:712].reshape(30, 10)[:, :3],
        ])
        best = 0.0
        candidates = []
        for index, (present, normalized_x, normalized_y) in enumerate(entities):
            if present < 0.5:
                continue
            x, y = float(normalized_x) * 1080, float(normalized_y) * 696
            distance = math.hypot(x, y)
            # 벽 너머를 향해 조준한 것만으로 보조 보상을 얻지 않게 한다.
            if not game.terrain.line_of_sight(player.x, player.y, player.x + x, player.y + y):
                continue
            candidates.append((distance * (0.8 if index < 9 else 1), x, y, distance))
        if self.settings.focus_nearest and candidates:
            candidates = [min(candidates, key=lambda value: value[0])]
        for _, x, y, distance in candidates:
            cosine = (x * float(observation[4]) + y * float(observation[3])) / max(distance, 1)
            alignment = max(0.0, cosine) ** 4
            proximity = math.exp(-max(0, distance - effective_range * 0.7) / 400)
            best = max(best, alignment * proximity)
        return best * self.settings.potential_scale

    def reset(self, game, observation):
        self.previous_potential = self.potential(game, observation)
        self.previous_observation = observation.copy()
        self.previous_origin = (game.player.x, game.player.y)
        weapon = game.combat.weapon_values(game.player)
        self.previous_weapon = (weapon[3], weapon[4])

    def shot_quality(self, game, observation, origin=None, weapon=None):
        """실제 사거리·시야선·표적 속도로 발사 직전 정렬 정도를 평가한다."""
        player = game.player
        if weapon is None:
            values = game.combat.weapon_values(player)
            weapon = (values[3], values[4])
        speed, life = weapon
        origin_x, origin_y = origin if origin is not None else (player.x, player.y)
        best = 0.0
        candidates = []
        for start, end, count, size, radius_index in [(133, 412, 9, 31, 8), (412, 712, 30, 10, 6)]:
            for entity in observation[start:end].reshape(count, size):
                if entity[0] < 0.5:
                    continue
                x, y = float(entity[1]) * 1080, float(entity[2]) * 696
                distance = math.hypot(x, y)
                radius = float(entity[radius_index]) * game.terrain.tile
                if distance > speed * life + radius:
                    continue
                if not game.terrain.line_of_sight(origin_x, origin_y, origin_x + x, origin_y + y):
                    continue
                candidates.append((distance * (0.8 if start == 133 else 1), distance, radius, x, y, entity))
        if self.settings.focus_nearest and candidates:
            candidates = [min(candidates, key=lambda value: value[0])]
        for _, distance, radius, x, y, entity in candidates:
            if distance > speed * life + radius:
                continue
            travel_time = min(life, distance / max(speed, 1))
            x += float(entity[3]) * game.config.max_speed * travel_time
            y += float(entity[4]) * game.config.max_speed * travel_time
            forward = x * float(observation[4]) + y * float(observation[3])
            sideways = y * float(observation[4]) - x * float(observation[3])
            angle = math.atan2(sideways, forward)
            tolerance = max(0.05, math.atan2(radius + game.config.bullet_radius, max(distance, 1)))
            best = max(best, math.exp(-0.5 * (angle / tolerance) ** 2))
        return best

    def projectile_threat(self, game, observation):
        """관찰된 적 탄환의 짧은 직선 궤적으로 근접 위험을 계산한다."""
        player = game.player
        bullets = observation[712:1224].reshape(64, 8)
        bullets = bullets[bullets[:, 0] > 0.5]
        if not len(bullets):
            return 0.0
        position = bullets[:, 1:3] * np.array([1080, 696])
        velocity = bullets[:, 3:5] * game.config.bullet_speed - np.array([player.velocity_x, player.velocity_y])
        time = np.clip(-(position * velocity).sum(axis=1) / np.maximum((velocity * velocity).sum(axis=1), 1e-6), 0, 18)
        separation = np.linalg.norm(position + velocity * time[:, None], axis=1)
        clearance = np.maximum(0, separation - player.radius - bullets[:, 5] * game.terrain.tile)
        return float(np.max(np.exp(-clearance / 60) * (1 - time / 24)))

    def calculate(self, game, observation, score_gain, events, terminated):
        settings = self.settings
        potential = 0.0 if terminated else self.potential(game, observation)
        quality = 0.0
        if settings.aimed_shot_bonus and events['player_shots_fired']:
            quality = self.shot_quality(game, self.previous_observation, self.previous_origin, self.previous_weapon)
        components = {
            'reward_score': score_gain * settings.score_scale,
            'reward_shape_damage': events['shape_damage_weighted'] * settings.shape_damage_scale,
            'reward_enemy_damage': events['enemy_damage'] * settings.enemy_damage_scale,
            'reward_damage_taken': -events['damage_taken'] / max(game.player.max_hp, 1) * settings.damage_fraction_penalty,
            'reward_death': -settings.death_penalty * float(terminated),
            'reward_shot_cost': -events['player_shots_fired'] * settings.shot_penalty,
            'reward_aim_progress': settings.gamma * potential - self.previous_potential,
            # 실제 발사가 있어야 보상을 주며 산탄 수에 비례해 보상이 부풀지 않도록 한다.
            'reward_aimed_shot': min(events['player_shots_fired'], 1) * quality * settings.aimed_shot_bonus,
            'reward_projectile_threat': -self.projectile_threat(game, observation) * settings.projectile_threat_penalty if settings.projectile_threat_penalty else 0.0,
        }
        self.previous_potential = potential
        self.previous_observation = observation.copy()
        self.previous_origin = (game.player.x, game.player.y)
        weapon = game.combat.weapon_values(game.player)
        self.previous_weapon = (weapon[3], weapon[4])
        return components
