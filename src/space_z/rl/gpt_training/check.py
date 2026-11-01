"""새 환경의 관찰·행동·시드 재현성과 PPO 저장·복원 계약을 검사한다."""
from pathlib import Path
import tempfile
from unittest.mock import patch

from gymnasium.utils.env_checker import check_env
import numpy as np
from stable_baselines3 import PPO
import torch

from .environment import make_environment, teacher_action, growth_choices, FactorizedActions, ROTATIONS
from space_z.config.progression_definitions import STAT_UPGRADES
from space_z.rl import DiscreteActionDecoder
from .train import save_model


def main():
    torch.set_num_threads(1)
    for stage in ["harvest", "combat", "full"]:
        environment = make_environment(stage, 40)
        try:
            check_env(environment.unwrapped, skip_render_check=True)
            first, _ = environment.reset(seed=123)
            second, _ = environment.reset(seed=123)
            np.testing.assert_array_equal(first, second)
            assert environment.observation_space.contains(first)
            for _ in range(40):
                action = teacher_action(second)
                assert environment.action_space.contains(action)
                stat, evolution = growth_choices(second)
                assert 0 <= stat <= 5 and 0 <= evolution <= 4
                second, reward, terminated, truncated, info = environment.step(action)
                assert np.isfinite(reward) and environment.observation_space.contains(second)
                if terminated or truncated:
                    assert "experiment_episode" in info
                    break
            print(f"PASS {stage}: deterministic observation, valid teacher actions, finite rewards and episode records")
        finally:
            environment.close()
    decoder = FactorizedActions(enable_growth=True)
    base_decoder = DiscreteActionDecoder(enable_stat_upgrades=True, enable_evolution=True)
    for move in range(9):
        for rotation in range(len(ROTATIONS)):
            for fire in range(2):
                for stat in range(decoder.stat_count):
                    for evolution in range(decoder.evolution_count):
                        actual = decoder.decode([move, rotation, fire, stat, evolution])
                        encoded = (((move * 3) * 2 + fire) * decoder.stat_count + stat) * decoder.evolution_count + evolution
                        expected = base_decoder.decode(encoded)
                        expected.rotation = float(ROTATIONS[rotation])
                        assert actual == expected
    print("PASS all factorized growth actions match the shared decoder")
    for stage in ["harvest", "combat", "full"]:
        environment = make_environment(stage, 20, pure_rl=True)
        try:
            check_env(environment.unwrapped, skip_render_check=True)
            observation, _ = environment.reset(seed=123)
            repeated, _ = environment.reset(seed=123)
            np.testing.assert_array_equal(observation, repeated)
            assert observation.shape == (5139,)
            with patch("space_z.rl.gpt_training.environment.growth_choices", side_effect=AssertionError("automatic growth called")), \
                 patch("space_z.rl.gpt_training.environment.TacticalObservation.build", side_effect=AssertionError("tactical observation called")):
                progression = environment.unwrapped.game.progression
                progression.upgrade_points = 1
                progression.special_reward_levels = [10]
                progression.open_rewards()
                expected_ship = progression.reward_choices[0].key
                environment.step(np.array([0, 4, 0, 0, 0]))
                assert progression.upgrade_points == 1 and progression.special_reward_levels == [10]
                environment.step(np.array([0, 4, 0, 1, 1]))
                assert progression.stat_levels[STAT_UPGRADES[0].key] == 1
                assert progression.upgrade_points == 0 and not progression.special_reward_levels
                assert environment.unwrapped.game.player.job == expected_ship
                while True:
                    observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
                    assert np.isfinite(reward) and environment.observation_space.contains(observation)
                    if terminated or truncated:
                        assert "experiment_episode" in info
                        break
            print(f"PASS pure PPO {stage}: raw observations, policy-controlled growth, finite rewards")
        finally:
            environment.close()
    environment = make_environment("harvest", 40, pure_rl=True)
    try:
        model = PPO("MlpPolicy", environment, n_steps=16, batch_size=16, n_epochs=1, seed=123)
        model.training_method = "pure_ppo"
        with patch("space_z.rl.gpt_training.train.imitate", side_effect=AssertionError("imitation called")), \
             patch("space_z.rl.gpt_training.train.collect_demonstrations", side_effect=AssertionError("demonstrations called")):
            model.learn(16)
        observation, _ = environment.reset(seed=1)
        expected = model.predict(observation, deterministic=True)[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "roundtrip.zip"
            save_model(model, path)
            assert path.is_file() and not path.with_suffix(".tmp.zip").exists()
            restored = PPO.load(path, env=environment)
            assert restored.training_method == "pure_ppo"
            np.testing.assert_array_equal(expected, restored.predict(observation, deterministic=True)[0])
        print("PASS real PPO atomic save/load: action and observation spaces match")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
