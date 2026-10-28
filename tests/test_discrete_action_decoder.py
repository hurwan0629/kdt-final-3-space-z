"""이산 행동 공간의 옵션·전체 조합·입력 검증을 게임 실행 없이 확인한다."""

from dataclasses import astuple
import math
import unittest

from gymnasium.spaces import Discrete
import numpy as np

from space_z.rl.actions.discrete_action_decoder import DiscreteActionDecoder
from space_z.rl.actions.player_control import PlayerControl


class DiscreteActionDecoderTests(unittest.TestCase):
    def test_option_combinations_define_fixed_space_sizes(self):
        for stat, evolution, count in (
            (False, False, 54),
            (True, False, 324),
            (False, True, 270),
            (True, True, 1620),
        ):
            with self.subTest(stat=stat, evolution=evolution):
                decoder = DiscreteActionDecoder(enable_stat_upgrades=stat, enable_evolution=evolution)
                self.assertIsInstance(decoder.action_space, Discrete)
                self.assertEqual(decoder.action_space.n, count)
                self.assertEqual(decoder.action_space.start, 0)

    def test_default_space_keeps_movement_rotation_and_fire_independent(self):
        decoder = DiscreteActionDecoder()
        self.assertEqual(decoder.decode(0), PlayerControl(0.0, 0.0, 0.0, False))
        self.assertEqual(decoder.decode(19), PlayerControl(1.0, 0.0, 0.0, True))
        self.assertEqual(decoder.decode(2), PlayerControl(0.0, 0.0, -1.0, False))

    def test_enabled_choices_map_to_stat_keys_and_current_evolution_indices(self):
        decoder = DiscreteActionDecoder(enable_stat_upgrades=True, enable_evolution=True)
        self.assertEqual(
            decoder.decode(643),
            PlayerControl(1.0, 0.0, -1.0, True, stat_upgrade_key="damage", evolution_choice=2),
        )
        self.assertEqual(decoder.decode(1).evolution_choice, 0)
        self.assertEqual(decoder.decode(5).stat_upgrade_key, "health")
        self.assertIsNone(decoder.decode(5).evolution_choice)
        last = decoder.decode(1619)
        self.assertAlmostEqual(last.move_x, -1 / math.sqrt(2))
        self.assertAlmostEqual(last.move_y, -1 / math.sqrt(2))
        self.assertEqual((last.rotation, last.want_fire, last.stat_upgrade_key, last.evolution_choice),
                         (1.0, True, "speed", 3))

    def test_every_action_has_a_unique_normalized_control(self):
        for stat in (False, True):
            for evolution in (False, True):
                decoder = DiscreteActionDecoder(enable_stat_upgrades=stat, enable_evolution=evolution)
                controls = set()
                for action in range(decoder.action_space.n):
                    control = decoder.decode(action)
                    controls.add(astuple(control))
                    speed = math.hypot(control.move_x, control.move_y)
                    self.assertTrue(speed == 0.0 or math.isclose(speed, 1.0))
                    self.assertIn(control.rotation, (-1.0, 0.0, 1.0))
                    self.assertIsInstance(control.want_fire, bool)
                    if not stat:
                        self.assertIsNone(control.stat_upgrade_key)
                    if not evolution:
                        self.assertIsNone(control.evolution_choice)
                self.assertEqual(len(controls), decoder.action_space.n)

    def test_gymnasium_samples_and_numpy_integers_are_accepted(self):
        decoder = DiscreteActionDecoder()
        self.assertEqual(decoder.decode(np.int64(19)), decoder.decode(19))
        decoder.action_space.seed(42)
        self.assertIsInstance(decoder.decode(decoder.action_space.sample()), PlayerControl)

    def test_non_integer_actions_are_rejected(self):
        decoder = DiscreteActionDecoder()
        for action in (True, False, 1.5, float("nan"), "1", None, np.array(1), np.array([1])):
            with self.subTest(action=action), self.assertRaises(TypeError):
                decoder.decode(action)

    def test_actions_outside_the_space_are_rejected(self):
        for enabled in (False, True):
            decoder = DiscreteActionDecoder(enable_stat_upgrades=enabled, enable_evolution=enabled)
            for action in (-1, decoder.action_space.n):
                with self.subTest(enabled=enabled, action=action), self.assertRaises(ValueError):
                    decoder.decode(action)

    def test_options_require_booleans(self):
        for option in ("enable_stat_upgrades", "enable_evolution"):
            for value in (1, "true", None):
                with self.subTest(option=option, value=value), self.assertRaises(TypeError):
                    DiscreteActionDecoder(**{option: value})


if __name__ == "__main__":
    unittest.main()
