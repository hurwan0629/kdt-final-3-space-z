"""담당 객체 사이의 상태 전달과 읽기 전용 표시 계산을 검증한다."""
import os
import pickle
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from space_z.config import GameConfig
from space_z.game.game_core import GameCore
from space_z.rendering.progression_panel import build_player_stat_summary
from space_z.rendering.renderer import Renderer


class StateOwnershipTests(unittest.TestCase):
    def game(self, enemies=0):
        config = GameConfig(num_enemies=enemies, num_shapes=0)
        config.terrain.enabled = False
        return GameCore(config, seed=7)

    def test_elite_kill_heals_and_awards_once_before_events_are_consumed(self):
        game = self.game(enemies=1)
        game.player.hp = 40
        enemy = game.enemies[0]
        enemy.elite = True
        game.combat._damage_tank(enemy, 100000, True)
        game.combat._damage_tank(enemy, 100000, True)

        self.assertEqual(game.progression.kills, 1)
        self.assertEqual(game.player.hp, min(
            game.player.max_hp,
            40 + game.player.max_hp * game.config.enemy_ai.elite_heal_ratio,
        ))
        score = round(game.config.enemy_kill_score * game.config.enemy_ai.elite_score_scale)
        self.assertEqual(game.progression.score, score)
        self.assertEqual(game.pop_events()["enemies_killed"], 1)
        self.assertEqual(game.pop_events()["enemies_killed"], 0)
        self.assertEqual(game.pop_score_award_events(), [
            (0, enemy.x, enemy.y, score, "elite"),
        ])
        self.assertEqual(game.pop_score_award_events(), [])

    def test_invalid_growth_requests_leave_world_and_random_state_unchanged(self):
        game = self.game(enemies=2)
        before = pickle.dumps(game)
        self.assertFalse(game.progression.choose_reward(0))
        self.assertFalse(game.progression.upgrade_stat("health"))
        self.assertEqual(pickle.dumps(game), before)

    def test_displayed_recovery_matches_actual_recovery_without_mutating_state(self):
        game = self.game()
        game.progression.upgrade_points = 1
        self.assertTrue(game.progression.upgrade_stat("health"))
        game.player.hp = 30
        game.player.regen_bonus = .017
        game.frame = game.config.regen_delay
        game.player.last_hit_frame = 0
        before = pickle.dumps(game)
        stats = dict(build_player_stat_summary(game))
        self.assertEqual(pickle.dumps(game), before)

        previous_hp = game.player.hp
        game.combat._regen_player()
        actual_per_second = (game.player.hp - previous_hp) * game.config.fps
        self.assertEqual(stats["회복"], f"{actual_per_second:.1f} / 초")
        game.player.hp = game.player.max_hp
        game.combat._regen_player()
        self.assertEqual(game.player.hp, game.player.max_hp)

    def test_public_camera_and_legacy_path_preserve_center_and_coordinate_roundtrip(self):
        game = self.game()
        renderer = Renderer.__new__(Renderer)
        renderer.screen = pygame.Surface((960, 720))
        for view_size in ((960, 720), (1600, 1200)):
            renderer.view_width, renderer.view_height = view_size
            for position in ((0, 0), (game.config.world_width, game.config.world_height)):
                game.player.x, game.player.y = position
                renderer.update_camera(game.player)
                self.assertEqual(renderer.world_to_screen(*position), (480, 360))
                self.assertEqual(renderer.screen_to_world(480, 360), position)
                public_camera = (renderer.camera_x, renderer.camera_y)
                renderer._update_camera(game.player)
                self.assertEqual((renderer.camera_x, renderer.camera_y), public_camera)
                world = renderer.screen_to_world(120, 90)
                self.assertEqual(renderer.world_to_screen(*world), (120, 90))


if __name__ == "__main__":
    unittest.main()
