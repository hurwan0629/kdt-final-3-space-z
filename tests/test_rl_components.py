"""새 RL 구성 요소의 입출력·재현성·종료·자원 소유권을 검증한다."""

import copy
from dataclasses import asdict
import math
import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from gymnasium.utils.env_checker import check_env

from space_z.config import EnvConfig, GameConfig, ObsConfig, RewardConfig, TerrainConfig
from space_z.game.entities import Bullet, Tank
from space_z.game.game_core import GameCore
from space_z.game.terrain import BREAKABLE, EMPTY, MUD, WALL, TileMap
from space_z.rl import (ContinuousActionDecoder, DiscreteActionDecoder, PixelObservation,
                        PolicyEvaluator, ResourceMonitor, RewardCalculator, TankEnv, VectorObservation)
from space_z.rl.observations.progression_observation import build_progression, SHIP_KEYS


def make_environment(builder=None, decoder=None, **kwargs):
    """짧은 검증용 빈 월드를 새 구성 요소에 연결한다."""
    config = GameConfig(num_enemies=0, num_shapes=0, world_scale=1, world_area_scale=1)
    config.terrain.enabled = False
    return TankEnv(config, builder or VectorObservation(), decoder or DiscreteActionDecoder(),
                   RewardCalculator(RewardConfig()), **kwargs)


class ContinuousActionTests(unittest.TestCase):
    def test_movement_bins_keep_fixed_strength_and_rotation(self):
        decoder = ContinuousActionDecoder()
        moves = set()
        for index in range(9):
            control = decoder.decode(np.array([-1+(index+.5)*2/9, .25, 1], np.float32))
            moves.add((control.move_x, control.move_y))
            self.assertAlmostEqual(control.rotation, .25)
            self.assertTrue(control.want_fire)
            self.assertAlmostEqual(math.hypot(control.move_x, control.move_y), 0 if index == 0 else 1)
        self.assertEqual(len(moves), 9)

    def test_options_clipping_and_growth(self):
        for stat, evolution in ((False, False), (True, False), (False, True), (True, True)):
            decoder = ContinuousActionDecoder(enable_stat_upgrades=stat, enable_evolution=evolution)
            self.assertEqual(decoder.action_space.shape, (3+stat+evolution,))
            control = decoder.decode(np.full(decoder.action_space.shape, -2., np.float32))
            self.assertEqual((control.move_x, control.move_y, control.rotation), (0, 0, -1))
            self.assertIsNone(control.stat_upgrade_key)
            self.assertIsNone(control.evolution_choice)
        decoder = ContinuousActionDecoder(enable_stat_upgrades=True, enable_evolution=True)
        control = decoder.decode(np.ones(5, np.float32))
        self.assertEqual(control.stat_upgrade_key, "speed")
        self.assertEqual(control.evolution_choice, 3)

    def test_invalid_inputs(self):
        decoder = ContinuousActionDecoder()
        for action in (3, [0., 0., 0.], np.array([0, 0, 0]), np.array([True]*3)):
            with self.assertRaises(TypeError):
                decoder.decode(action)
        for action in (np.zeros(2), np.zeros((1, 3)), np.array([0., math.nan, 0.]), np.array([0., 0., math.inf])):
            with self.assertRaises(ValueError):
                decoder.decode(action)
        with self.assertRaises(TypeError):
            ContinuousActionDecoder(enable_evolution=1)


class VectorObservationTests(unittest.TestCase):
    def setUp(self):
        self.env = make_environment()
        self.env.reset(seed=4)
        self.addCleanup(self.env.close)

    def test_fixed_layout_and_no_state_mutation(self):
        game, builder = self.env.game, self.env.observation_builder
        before = (asdict(game.player), copy.deepcopy(game.events), copy.deepcopy(game.rng.bit_generator.state))
        vector = builder.build(game)
        self.assertEqual(vector.shape, (5139,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(builder.observation_space.contains(vector))
        self.assertEqual(before, (asdict(game.player), game.events, game.rng.bit_generator.state))
        np.testing.assert_array_equal(vector[builder.slices["enemies"]], 0)
        vector[:] = -1
        self.assertTrue((builder.build(game)[builder.slices["enemies"]] == 0).all())

    def test_entities_filter_sort_masks_and_corner_overlap(self):
        game = self.env.game
        player = game.player
        def enemy(identifier, x, y=0):
            return Tank(identifier, "enemy", player.x+x, player.y+y, 30, 60, 10, 28)
        half_width = self.env.observation_builder.config.grid_width * game.terrain.tile / 2
        half_height = self.env.observation_builder.config.grid_height * game.terrain.tile / 2
        game.enemies = [enemy(2, 200), enemy(3, 50), enemy(4, half_width+20),
                        enemy(5, half_width+9, half_height+9)]
        game.combat.bullets = [Bullet(player.x, player.y, 0, 0, 20, 1, "player", 1, 1),
                               Bullet(player.x, player.y, 0, 0, 20, 2, "enemy", 1, 1)]
        builder = VectorObservation(ObsConfig(n_enemies=3))
        vector = builder.build(game)
        enemies = vector[builder.slices["enemies"]].reshape(3, -1)
        self.assertEqual(list(enemies[:, 0]), [1, 1, 0])
        self.assertLess(enemies[0, 1], enemies[1, 1])
        bullets = vector[builder.slices["bullets"]].reshape(64, -1)
        self.assertEqual(bullets[:, 0].sum(), 1)

    def test_grid_outside_mud_and_breakable_hp(self):
        game = self.env.game
        terrain_config = TerrainConfig(mud_enabled=True, breakable_enabled=True)
        game.terrain = TileMap([[EMPTY, MUD, BREAKABLE], [EMPTY, WALL, EMPTY], [EMPTY]*3],
                              48, (0, 0), [], "test", terrain_config)
        game.player.x, game.player.y = 24, 24
        game.terrain.hp[(2, 0)] /= 2
        builder = VectorObservation(ObsConfig(grid_width=5, grid_height=5))
        grid = builder.build(game)[builder.slices["grid"]].reshape(builder.grid_shape)
        self.assertEqual(grid[0, 0, 0], 1)
        self.assertEqual(grid[0, 2, 2], 0)
        self.assertEqual(grid[1, 2, 3], 1)
        self.assertEqual(grid[2, 2, 4], .5)
        self.assertEqual(grid[0, 2, 4], 1)

    def test_progression_flags_choices_and_disabled_growth(self):
        game = self.env.game
        game.progression._add_score(sum(60+i*40 for i in range(9)))
        before = (list(game.progression.special_reward_levels), game.progression.reward_choices)
        values = build_progression(game)
        self.assertEqual(values.shape, (123,))
        start = len(SHIP_KEYS)+4+5
        self.assertTrue((values[start:start+5] == 1).all())
        choices = values[start+5:].reshape(4, -1)
        self.assertTrue((choices[:, 0] == 1).all())
        self.assertEqual(before, (game.progression.special_reward_levels, game.progression.reward_choices))
        game.config.progression_enabled = False
        old_level = game.progression.level
        game.progression._add_score(99999)
        self.assertEqual(game.progression.level, old_level)

    def test_invalid_observation_config(self):
        for values in ({"n_enemies": -1}, {"grid_width": 2}, {"frame_stack": 0}, {"ray_max_dist": math.inf}):
            with self.assertRaises(ValueError):
                ObsConfig(**values)


class EnvironmentTests(unittest.TestCase):
    def test_gym_contract_for_vector_and_continuous(self):
        for decoder in (DiscreteActionDecoder(), ContinuousActionDecoder()):
            env = make_environment(decoder=decoder)
            try:
                check_env(env, skip_render_check=True)
            finally:
                env.close()

    def test_lifecycle_and_time_limit(self):
        env = make_environment(env_config=EnvConfig(frame_skip=3, max_steps=1))
        self.addCleanup(env.close)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual((info["frames"], info["steps"]), (3, 1))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=1)
        self.assertEqual(env.steps, 0)
        env.close()
        env.close()
        with self.assertRaises(RuntimeError):
            env.reset()

    def test_early_death_event_reward_and_no_extra_frames(self):
        env = make_environment()
        self.addCleanup(env.close)
        env.reset(seed=5)
        env.game.player.hp = 0
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["frames"], 1)
        self.assertTrue(info["events"]["died"])
        self.assertAlmostEqual(reward, -20.0001)
        self.assertFalse(env.game.events["died"])

    def test_fire_cost_counts_actual_bullets_and_events_are_copied(self):
        env = make_environment()
        self.addCleanup(env.close)
        env.reset(seed=5)
        _, reward, _, _, info = env.step(1)
        self.assertEqual(info["events"]["player_shots_fired"], 1)
        self.assertAlmostEqual(reward, -.0101)
        _, reward, _, _, info = env.step(1)
        self.assertEqual(info["events"]["player_shots_fired"], 0)
        self.assertAlmostEqual(reward, -.0001)
        info["stat_levels"]["health"] = 900
        self.assertEqual(env.game.progression.stat_levels["health"], 0)
        self.assertEqual(env.game.score_award_events, [])

    def test_growth_requests_apply_once_per_environment_step(self):
        env = make_environment(decoder=ContinuousActionDecoder(enable_stat_upgrades=True, enable_evolution=True))
        self.addCleanup(env.close)
        env.reset(seed=3)
        env.game.progression._add_score(sum(60+i*40 for i in range(9)))
        before = env.game.progression.upgrade_points
        env.step(np.array([-1, 0, -1, -.5, -.5], np.float32))
        self.assertEqual(env.game.progression.upgrade_points, before-1)
        self.assertEqual(env.game.progression.stat_levels["health"], 1)
        self.assertEqual(env.game.player.job, "twin")

    def test_fixed_seed_matches_direct_game_frames(self):
        env = make_environment()
        self.addCleanup(env.close)
        env.reset(seed=7)
        reference = GameCore(env.game_config, seed=7)
        decoder = env.action_decoder
        from space_z.game.actions import MOVE_VECTORS, ROT_DIR, encode_action
        for action in (0, 1, 15, 33, 53):
            control = decoder.decode(action)
            for _ in range(4):
                reference.step(encode_action(MOVE_VECTORS.index((control.move_x, control.move_y)),
                                             ROT_DIR.index(control.rotation)), want_fire=control.want_fire)
            _, _, _, _, info = env.step(action)
            self.assertEqual(asdict(env.game.player), asdict(reference.player))
            self.assertEqual([asdict(bullet) for bullet in env.game.combat.bullets],
                             [asdict(bullet) for bullet in reference.combat.bullets])
            self.assertEqual(info["events"], reference.pop_events())

    def test_continuous_rotation_changes_angle_without_movement_strength(self):
        env = make_environment(decoder=ContinuousActionDecoder())
        self.addCleanup(env.close)
        env.reset(seed=4)
        angle = env.game.player.angle
        env.step(np.array([-1., .25, -1.], np.float32))
        from space_z.game.actions import wrap_angle
        self.assertAlmostEqual(env.game.player.angle, wrap_angle(angle + .25 * 4 * env.game_config.turn_speed))

    def test_core_equivalence_includes_enemies_shapes_and_progression(self):
        config = GameConfig(num_enemies=2, num_shapes=3)
        env = TankEnv(config, VectorObservation(), DiscreteActionDecoder(), RewardCalculator(RewardConfig()))
        self.addCleanup(env.close)
        env.reset(seed=12)
        reference = GameCore(config, seed=12)
        from space_z.game.actions import MOVE_VECTORS, ROT_DIR, encode_action
        for action in (1, 15, 33, 53, 9):
            control = env.action_decoder.decode(action)
            for _ in range(4):
                reference.step(encode_action(MOVE_VECTORS.index((control.move_x, control.move_y)),
                                             ROT_DIR.index(control.rotation)), want_fire=control.want_fire)
            _, _, _, _, info = env.step(action)
            for name in ("enemies", "shapes"):
                self.assertEqual([asdict(entity) for entity in getattr(env.game, name)],
                                 [asdict(entity) for entity in getattr(reference, name)])
            self.assertEqual(asdict(env.game.player), asdict(reference.player))
            self.assertEqual([asdict(bullet) for bullet in env.game.combat.bullets],
                             [asdict(bullet) for bullet in reference.combat.bullets])
            self.assertEqual(env.game.progression.score, reference.progression.score)
            self.assertEqual(info["events"], reference.pop_events())

    def test_disabled_growth_ignores_requests(self):
        env = make_environment(decoder=ContinuousActionDecoder(enable_stat_upgrades=True))
        self.addCleanup(env.close)
        env.game_config.progression_enabled = False
        env.reset(seed=3)
        env.game.progression.upgrade_points = 5
        env.step(np.array([-1, 0, -1, -.5], np.float32))
        self.assertEqual(env.game.progression.upgrade_points, 5)
        self.assertEqual(env.game.progression.stat_levels["health"], 0)

    def test_bad_environment_settings_fail_before_game_creation(self):
        for config in (EnvConfig(frame_skip=0), EnvConfig(max_steps=True)):
            with patch("space_z.rl.environment.GameCore") as core:
                with self.assertRaises(ValueError):
                    make_environment(env_config=config)
                core.assert_not_called()


class PixelObservationTests(unittest.TestCase):
    def test_offscreen_stack_reset_and_game_immutability(self):
        builder = PixelObservation(ObsConfig(image_width=90, image_height=58))
        env = make_environment(builder=builder)
        self.addCleanup(env.close)
        display = pygame.display.get_surface()
        observation, _ = env.reset(seed=7)
        self.assertIs(pygame.display.get_surface(), display)
        self.assertEqual(observation["image"].shape, (4, 58, 90))
        self.assertTrue(env.observation_space.contains(observation))
        np.testing.assert_array_equal(observation["image"][0], observation["image"][-1])
        before = (asdict(env.game.player), copy.deepcopy(env.game.events), env.game.frame)
        again = builder.build(env.game)
        self.assertEqual(before, (asdict(env.game.player), env.game.events, env.game.frame))
        np.testing.assert_array_equal(again["image"], observation["image"])
        updated = env.step(0)[0]
        np.testing.assert_array_equal(updated["image"][:-1], observation["image"][1:])
        reset, _ = env.reset(seed=7)
        np.testing.assert_array_equal(reset["image"], observation["image"])
        np.testing.assert_array_equal(reset["progression"], build_progression(env.game))

    def test_dqn_ppo_observation_action_combinations(self):
        for builder_type in (VectorObservation, PixelObservation):
            for decoder_type in (DiscreteActionDecoder, ContinuousActionDecoder):
                env = make_environment(builder=builder_type(), decoder=decoder_type())
                try:
                    env.reset(seed=3)
                    result = env.step(env.action_space.sample())
                    self.assertTrue(env.observation_space.contains(result[0]))
                    self.assertTrue(math.isfinite(result[1]))
                finally:
                    env.close()


class EvaluationTests(unittest.TestCase):
    def test_repeat_seeds_summary_and_caller_owned_environment(self):
        env = make_environment(env_config=EnvConfig(frame_skip=2, max_steps=3))
        self.addCleanup(env.close)
        evaluator = PolicyEvaluator(env, lambda observation: 0)
        first = evaluator.evaluate([3, 7])
        self.assertEqual(first, evaluator.evaluate([3, 7]))
        self.assertEqual(first["episodes"][0]["game_frames"], 6)
        self.assertEqual(first["summary"]["environment_steps"], {"mean": 3., "std": 0.})
        self.assertIsNone(first["resources"])
        self.assertFalse(env._closed)
        for seeds in ([], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                evaluator.evaluate(seeds)

    def test_failure_stops_monitor(self):
        env = make_environment()
        self.addCleanup(env.close)
        monitor = ResourceMonitor(.01)
        self.addCleanup(monitor.close)
        def broken_policy(observation):
            raise ValueError("잘못된 모델 출력")
        with self.assertRaises(ValueError):
            PolicyEvaluator(env, broken_policy, monitor).evaluate([3])
        self.assertFalse(monitor._running)
        self.assertIsNone(monitor._thread)


class ResourceMonitorTests(unittest.TestCase):
    def test_periodic_cpu_ram_stop_restart_close(self):
        monitor = ResourceMonitor(.01)
        self.addCleanup(monitor.close)
        with self.assertRaises(RuntimeError):
            monitor.sample()
        with self.assertRaises(RuntimeError):
            monitor.stop()
        monitor.start()
        with self.assertRaises(RuntimeError):
            monitor.start()
        threading.Event().wait(.04)
        sample = monitor.sample()
        result = monitor.stop()
        self.assertGreaterEqual(result["sample_count"], 3)
        self.assertGreater(sample["process_ram_bytes"], 0)
        self.assertGreater(result["elapsed_seconds"], 0)
        self.assertIsNone(result["peak_gpu_device_memory_bytes"])
        self.assertEqual(result, monitor.stop())
        monitor.start()
        monitor.stop()
        monitor.close()
        monitor.close()
        with self.assertRaises(RuntimeError):
            monitor.start()

    def test_gpu_missing_does_not_disable_cpu_or_ram(self):
        monitor = ResourceMonitor(gpu_index=0)
        self.addCleanup(monitor.close)
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor.start()
        sample = monitor.sample()
        self.assertIsNone(sample["gpu_device_utilization_percent"])
        self.assertGreater(sample["process_ram_bytes"], 0)
        monitor.stop()

    def test_invalid_monitor_configuration(self):
        for values in ({"sample_interval_seconds": 0}, {"sample_interval_seconds": math.nan},
                       {"process_id": 0}, {"gpu_index": -1}):
            with self.assertRaises(ValueError):
                ResourceMonitor(**values)


if __name__ == "__main__":
    unittest.main()
