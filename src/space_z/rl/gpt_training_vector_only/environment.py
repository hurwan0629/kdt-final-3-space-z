"""기존 게임 환경을 조립하고 실제 결과만 보상으로 사용한다."""
from dataclasses import asdict
import gymnasium as gym

from space_z.config import EnvConfig, GameConfig, ObsConfig, RewardConfig
from space_z.rl.environment import TankEnv
from space_z.rl.rewards import RewardCalculator
from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder
from space_z.rl.observations.vector_observation import VectorObservation

REWARD_COEFFICIENTS = dict(score_gain=0.02, shape_damage=0.01, enemy_damage=0.01,
                           damage_taken=-0.02, death=-3.0)


class OutcomeRewardEnvironment(gym.Wrapper):
    """점수·실제 피해·사망을 기록하며 관찰과 행동은 그대로 전달한다."""

    def __init__(self, env, reward_overrides=None):
        super().__init__(env)
        self.reward_coefficients = dict(REWARD_COEFFICIENTS)
        if reward_overrides:
            self.reward_coefficients.update(reward_overrides)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_score = info['score']
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        events = info['events']
        values = dict(
            score_gain=info['score'] - self.previous_score,
            shape_damage=events['shape_damage_weighted'], enemy_damage=events['enemy_damage'],
            damage_taken=events['damage_taken'], death=float(events['died']
        ))
        components = {name: values[name] * scale for name, scale in self.reward_coefficients.items()}
        self.previous_score = info['score']
        info['reward_components'] = components
        info['reward_inputs'] = values
        info['player_health'] = self.env.unwrapped.game.player.hp
        info['player_max_health'] = self.env.unwrapped.game.player.max_hp
        return observation, float(sum(components.values())), terminated, truncated, info


def make_environment(max_seconds=120, render_mode=None, reward_overrides=None):
    game_config = GameConfig()
    reward_config = RewardConfig(shape_damage=0, shape_destroy=0, enemy_damage=0, enemy_kill=0,
                                 damage_taken=0, death=0, time_step=0, shot_cost=0)
    environment = TankEnv(
        game_config=game_config, observation_builder=VectorObservation(ObsConfig()),
        action_decoder=RotationContinuousActionDecoder(), 
        reward_calculator=RewardCalculator(reward_config),
        env_config=EnvConfig(frame_skip=2, max_steps=round(max_seconds * game_config.fps / 2)),
        render_mode=render_mode,
    )
    return OutcomeRewardEnvironment(environment, reward_overrides)


def environment_configuration(reward_overrides=None):
    coefficients = dict(REWARD_COEFFICIENTS)
    if reward_overrides:
        coefficients.update(reward_overrides)
    return dict(game=asdict(GameConfig()), observation=asdict(ObsConfig()),
                frame_skip=2, reward_coefficients=coefficients,
                extra_policy_inputs=[], target_selection=False, imitation=False)
