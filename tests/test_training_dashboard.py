import importlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional training dependencies are not installed')
class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_a_real_checkpoint_without_changing_training_control(self):
        self.assertIsNotNone(importlib.util.find_spec('space_z.rl.training.dashboard'),
                             'The visible training dashboard must exist')
        dashboard = importlib.import_module('space_z.rl.training.dashboard')
        from space_z.rl.training.runtime import make_model, make_vec_env, publish_checkpoint, atomic_json, read_json
        from space_z.rl.training.settings import TrainSettings
        import pygame
        cfg = TrainSettings(workers=1, rollout_steps=8, batch_size=8, max_steps=8)
        env = make_vec_env(cfg)
        model = make_model(cfg, env)
        env.reset()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                publish_checkpoint(model, env, directory, cfg)
                atomic_json(directory/'config.json', cfg.to_dict())
                atomic_json(directory/'control.json', {'command': 'run'})
                capture = directory/'capture.png'
                dashboard.run_dashboard(directory, capture=capture, max_frames=3, stop_on_close=False)
                image = pygame.image.load(str(capture))
                self.assertEqual(image.get_size(), (1360, 900))
                self.assertGreater(capture.stat().st_size, 20000)
                self.assertEqual(read_json(directory/'control.json')['command'], 'run')
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
