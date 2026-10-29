import importlib
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'Optional training dependencies are not installed')
class TrainingEnvironmentTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('space_z.rl.training'),
                             'The isolated training package must exist')
        return importlib.import_module('space_z.rl.training.environment')

    def test_training_package_exists(self):
        self.module()

    def test_destroy_reward_does_not_scale_with_inflated_game_points(self):
        m = self.module()
        from space_z.game.game_core import empty_events
        ev = empty_events()
        ev.update(shapes_destroyed=1, shape_destroyed_score=10000)
        a = m.reward_components(ev, 0., 0., False)
        ev['shape_destroyed_score'] = 25000
        b = m.reward_components(ev, 0., 0., False)
        self.assertEqual(a['shapes'], b['shapes'])
        ev = empty_events()
        ev['enemies_killed'] = 1
        self.assertGreater(m.reward_components(ev, 0., 0., False)['kills'], a['shapes'] * 30)

    def test_no_free_survival_reward_and_terminal_potential_is_zero(self):
        m = self.module()
        from space_z.game.game_core import empty_events
        self.assertEqual(sum(m.reward_components(empty_events(), 0., 0., False).values()), 0.)
        r = m.reward_components(empty_events(), .3, .4, True)
        self.assertAlmostEqual(r['potential'], -.3)
        r = m.reward_components(empty_events(), .3, .4, False)
        self.assertAlmostEqual(r['potential'], .995 * .4 - .3)

    def test_observation_bounds_and_seeded_reset_sequences(self):
        m = self.module()
        a, b = m.TrainingEnv(max_steps=4), m.TrainingEnv(max_steps=4)
        try:
            oa, ia = a.reset(seed=17)
            ob, ib = b.reset(seed=17)
            self.assertTrue(a.observation_space.contains(oa))
            self.assertEqual(oa['vector'].shape, (265,))
            self.assertEqual(oa['grid'].shape, (1, 21, 21))
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
            self.assertFalse(env.unwrapped.game_cfg.progression_enabled)
            self.assertFalse(env.unwrapped.game_cfg.limited_ammo)
        finally:
            env.close()
        test_env = m.TrainingEnv(evaluation=True)
        self.assertEqual(test_env.stage, 2)
        test_env.close()


if __name__ == '__main__':
    unittest.main()
