import importlib
import importlib.util
import unittest
from unittest.mock import patch
from space_z.config import ObsConfig

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'Optional training dependencies are not installed')
class TrainingEnvironmentTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('space_z.rl.training_wooki'),
                             'The isolated training package must exist')
        return importlib.import_module('space_z.rl.training_wooki.environment')

    def test_training_package_exists(self):
        self.module()

    def test_score_objective_rewards_score_delta_and_penalizes_death(self):
        env = self.module().TrainingEnv(score_objective=True)
        try:
            env.reset(seed=17)
            obs, _, _, _, info = env.env.step(0)
            info['score'] = env.env.game.progression.score + 200
            with patch.object(env.env, 'step', return_value=(obs, 999., False, False, info)):
                _, reward, _, _, _ = env.step(0)
            self.assertAlmostEqual(reward, 1.)
            with patch.object(env.env, 'step', return_value=(obs, 999., True, False, info)):
                _, reward, _, _, _ = env.step(0)
            self.assertAlmostEqual(reward, .5)
        finally:
            env.close()

    def test_score_objective_gives_no_free_survival_reward(self):
        env = self.module().TrainingEnv(score_objective=True)
        try:
            env.reset(seed=17)
            obs, _, _, _, info = env.env.step(0)
            info['score'] = env.env.game.progression.score
            with patch.object(env.env, 'step', return_value=(obs, 999., False, False, info)):
                _, reward, _, _, _ = env.step(0)
            self.assertEqual(reward, 0.)
        finally:
            env.close()

    def test_observation_bounds_and_seeded_reset_sequences(self):
        m = self.module()
        a, b = m.TrainingEnv(max_steps=4), m.TrainingEnv(max_steps=4)
        try:
            oa, ia = a.reset(seed=17)
            ob, ib = b.reset(seed=17)
            self.assertTrue(a.observation_space.contains(oa))
            self.assertEqual(oa['vector'].shape, (ObsConfig().vector_size,))
            self.assertEqual(oa['grid'].shape, ObsConfig().grid_shape())
            for key in oa:
                np.testing.assert_array_equal(oa[key], ob[key])
            for action in (0, 14, 23):
                sa, sb = a.step(action), b.step(action)
                self.assertEqual(sa[1:4], sb[1:4])
                np.testing.assert_array_equal(sa[0]['vector'], sb[0]['vector'])
            _, ia2 = a.reset()
            _, ib2 = b.reset()
            self.assertNotEqual(ia['episode_seed'], ia2['episode_seed'])
            self.assertEqual(ia2['episode_seed'], ib2['episode_seed'])
        finally:
            a.close()
            b.close()

    def test_timeout_is_not_death_and_episode_is_reported(self):
        env = self.module().TrainingEnv(max_steps=2)
        try:
            env.reset(seed=9)
            env.step(0)
            _, reward, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info['episode_metrics']['steps'], 2)
            self.assertEqual(info['episode_metrics']['death'], 0)
            self.assertIn('task_return', info['episode_metrics'])
        finally:
            env.close()

    def test_curriculum_changes_only_on_reset_and_eval_is_full_game(self):
        m = self.module()
        env = m.TrainingEnv()
        try:
            env.reset(seed=19)
            self.assertEqual(env.stage, 0)
            env.set_training_step(400000)
            env.step(0)
            self.assertEqual(env.stage, 0)
            env.reset(seed=19)
            self.assertEqual(env.stage, 2)
            self.assertEqual(len(env.unwrapped.game.enemies), 9)
            self.assertFalse(env.unwrapped.game_config.progression_enabled)
            self.assertFalse(env.unwrapped.game_config.limited_ammo)
        finally:
            env.close()
        test_env = m.TrainingEnv(evaluation=True)
        self.assertEqual(test_env.stage, 2)
        test_env.close()


if __name__ == '__main__':
    unittest.main()
