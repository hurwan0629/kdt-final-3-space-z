"""보상 합산·코어의 실제 발사 이벤트·입력 검증을 화면 없이 확인한다."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from space_z.config import GameConfig, RewardConfig
from space_z.game.game_core import GameCore, empty_events
from space_z.game.types import CombatEvents
from space_z.rl.rewards import RewardCalculator


class RewardCalculatorTests(unittest.TestCase):
    def test_no_events_or_actual_shots_gives_zero_reward(self):
        calculator = RewardCalculator(RewardConfig(time_step=0.0))
        self.assertEqual(calculator.calculate(empty_events()), 0.0)

    def test_all_combat_rewards_and_shot_cost_are_combined(self):
        config = RewardConfig(
            shape_damage=0.5,
            shape_destroy=0.1,
            enemy_damage=0.2,
            enemy_kill=3.0,
            damage_taken=-0.4,
            death=-7.0,
            time_step=0.25,
            shot_cost=0.05,
        )
        events: CombatEvents = {
            "shape_damage_weighted": 4.0,
            "shape_destroyed_score": 20.0,
            "shapes_destroyed": 50,
            "enemy_damage": 5.0,
            "enemies_killed": 2,
            "damage_taken": 3.0,
            "died": True,
            "player_shots_fired": 3,
        }
        self.assertAlmostEqual(RewardCalculator(config).calculate(events), 2.9)

    def test_each_actual_projectile_is_charged_without_accumulating_state(self):
        calculator = RewardCalculator(RewardConfig(time_step=0.0, shot_cost=0.02))
        events = empty_events()
        events["player_shots_fired"] = 3
        before = events.copy()
        self.assertAlmostEqual(calculator.calculate(events), -0.06)
        self.assertAlmostEqual(calculator.calculate(events), -0.06)
        self.assertEqual(events, before)
        self.assertEqual(calculator.calculate(empty_events()), 0.0)

    def test_step_bonus_is_applied_once_and_zero_shot_cost_is_allowed(self):
        calculator = RewardCalculator(RewardConfig(time_step=0.25, shot_cost=0.0))
        events = empty_events()
        events["player_shots_fired"] = 10
        self.assertEqual(calculator.calculate(events), 0.25)

    def test_numpy_integer_shot_count_returns_python_float(self):
        calculator = RewardCalculator(RewardConfig(time_step=0.0, shot_cost=0.02))
        events = empty_events()
        events["player_shots_fired"] = np.int64(3)
        reward = calculator.calculate(events)
        self.assertIsInstance(reward, float)
        self.assertAlmostEqual(reward, -0.06)

    def test_invalid_shot_counts_are_rejected(self):
        calculator = RewardCalculator(RewardConfig())
        events = empty_events()
        events["player_shots_fired"] = -1
        with self.assertRaises(ValueError):
            calculator.calculate(events)
        for value in (1.5, True, "2"):
            events["player_shots_fired"] = value
            with self.subTest(value=value), self.assertRaises(TypeError):
                calculator.calculate(events)

    def test_invalid_shot_costs_are_rejected(self):
        for value in (-0.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RewardCalculator(RewardConfig(shot_cost=value))


class PlayerShotEventTests(unittest.TestCase):
    def setUp(self):
        config = GameConfig(num_enemies=0, num_shapes=0, limited_ammo=True)
        config.terrain.enabled = False
        self.game = GameCore(config, seed=7)

    def test_step_pop_and_reset_keep_shot_counts_in_the_correct_period(self):
        calculator = RewardCalculator(RewardConfig(time_step=0.0, shot_cost=0.02))
        self.game.step(0, want_fire=True)
        events = self.game.pop_events()
        self.assertEqual(events["player_shots_fired"], 1)
        self.assertAlmostEqual(calculator.calculate(events), -0.02)

        # 꺼낸 기록은 유지되고 다음 기간에는 같은 발사가 다시 집계되지 않는다.
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)
        self.assertEqual(events["player_shots_fired"], 1)
        self.game.step(0, want_fire=False)
        self.assertEqual(calculator.calculate(self.game.pop_events()), 0.0)

        self.game.player.cooldown = 0
        self.game.combat._fire(self.game.player, True)
        self.game.reset(seed=7)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)

    def test_blocked_requests_do_not_count_as_shots(self):
        player = self.game.player
        for reason in ("cooldown", "empty_ammo", "reload", "dead", "no_request"):
            with self.subTest(reason=reason):
                player.cooldown = 2 if reason == "cooldown" else 0
                player.ammo = 0 if reason == "empty_ammo" else player.max_ammo
                player.alive = reason != "dead"
                self.game.combat.reload_frames_left = 2 if reason == "reload" else 0
                self.game.combat._fire(player, want_fire=reason != "no_request")
                self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)
                self.assertEqual(self.game.combat.bullets, [])

    def test_each_muzzle_and_delayed_round_counts_only_when_emitted(self):
        player = self.game.player
        player.job = "twin_flank"
        self.game.combat._fire(player, True)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 2)
        self.assertEqual(len(player.pending_shots), 2)
        due_frame = player.pending_shots[0][0]

        self.game.frame = due_frame - 1
        self.game.combat._fire(player, False)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)

        # 방아쇠를 놓아도 예약된 나머지 포신이 실제 발사된 기간에 집계된다.
        self.game.frame = due_frame
        self.game.combat._fire(player, False)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 2)
        self.assertEqual(player.pending_shots, [])
        self.assertEqual(player.ammo, player.max_ammo - 4)

    def test_cancelled_delayed_rounds_do_not_count(self):
        player = self.game.player
        player.job = "twin_flank"
        self.game.combat._fire(player, True)
        self.game.pop_events()
        due_frame = player.pending_shots[0][0]
        self.assertTrue(self.game.combat.request_reload())
        self.game.frame = due_frame
        self.game.combat._fire(player, False)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)
        self.assertEqual(player.pending_shots, [])

    def test_enemy_projectiles_are_not_player_shots(self):
        enemy = replace(self.game.player, id=-1, team="enemy", x=self.game.player.x + 500)
        self.game.combat._fire(enemy, True)
        self.assertEqual(len(self.game.combat.bullets), 1)
        self.assertEqual(self.game.pop_events()["player_shots_fired"], 0)

    def test_projectile_that_hits_immediately_is_still_charged(self):
        with patch.object(self.game.combat, "_bullet_hit", return_value=True):
            self.game.combat._fire(self.game.player, True)
        self.assertEqual(self.game.combat.bullets, [])
        events = self.game.pop_events()
        self.assertEqual(events["player_shots_fired"], 1)
        calculator = RewardCalculator(RewardConfig(time_step=0.0, shot_cost=0.01))
        self.assertAlmostEqual(calculator.calculate(events), -0.01)


if __name__ == "__main__":
    unittest.main()
