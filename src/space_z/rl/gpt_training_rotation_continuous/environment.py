"""기존 게임·관찰·보상에 연속 회전 디코더만 연결한다."""
import gymnasium as gym

from space_z.config import EnvConfig, RewardConfig
from space_z.rl import RewardCalculator, TankEnv
from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder
from space_z.rl.observations.vector_observation import VectorObservation
from space_z.rl.gpt_training.environment import EnvironmentSettings, TrainingEnvironment, game_configuration
from space_z.rl.gpt_training.score_reward import ScoreReward, reward_settings


class RotationTrainingEnvironment(TrainingEnvironment):
    """원래 학습 래퍼의 보상·기록·종료 처리를 같은 인터페이스로 재사용한다."""

    def __init__(self, stage='full', max_steps=12000, reward_profile='score_v7', frame_skip=2):
        game_config = game_configuration(stage)
        observation_builder = VectorObservation()
        action_decoder = RotationContinuousActionDecoder()
        reward_config = RewardConfig()
        reward_calculator = RewardCalculator(reward_config)
        environment_config = EnvConfig(frame_skip=frame_skip, max_steps=max_steps)
        environment = TankEnv(
            game_config=game_config,
            observation_builder=observation_builder,
            action_decoder=action_decoder,
            reward_calculator=reward_calculator,
            env_config=environment_config,
        )
        gym.Wrapper.__init__(self, environment)
        self.settings = EnvironmentSettings(
            stage=stage, max_steps=max_steps, frame_skip=frame_skip,
            pure_rl=True, reward_profile=reward_profile,
        )
        self.last_observation = None
        self.episode_totals = {}
        self.score_reward = ScoreReward(reward_settings(reward_profile)) if reward_profile != 'legacy' else None
