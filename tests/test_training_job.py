import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional training dependencies are not installed')
class TrainingJobTests(unittest.TestCase):
    def trainer(self):
        self.assertIsNotNone(importlib.util.find_spec('space_z.rl.training_wooki.wooki01'),
                             'A real trainer must connect environments, evaluation and checkpoints')
        return importlib.import_module('space_z.rl.training_wooki.wooki01')

    def test_short_job_updates_weights_and_can_resume_with_optimizer_state(self):
        trainer = self.trainer()
        from space_z.rl.training_wooki.settings import TrainSettings
        from space_z.rl.training_wooki.runtime import read_json, load_checkpoint
        cfg = TrainSettings(total_steps=32, workers=1, rollout_steps=16, batch_size=16,
                            epochs=1, max_steps=8, eval_steps=2, eval_episodes=1,
                            eval_every=32, checkpoint_every=16)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trainer.run_training(directory, cfg, subprocess=False)
            state = read_json(directory/'status.json')
            self.assertEqual(state['status'], 'completed')
            self.assertGreaterEqual(state['step'], 32)
            self.assertGreater(state['updates'], 0)
            self.assertTrue((directory/'episodes.csv').stat().st_size > 100)
            first = read_json(directory/'latest.json')
            model, env = load_checkpoint(directory, first)
            self.assertGreater(len(model.policy.optimizer.state), 0)
            env.close()
            cfg.total_steps = 48
            trainer.run_training(directory, cfg, resume=True, subprocess=False)
            state = read_json(directory/'status.json')
            self.assertEqual(state['status'], 'completed')
            self.assertGreaterEqual(state['step'], 48)
            self.assertGreater(read_json(directory/'latest.json')['step'], first['step'])

    def test_stop_before_learning_still_saves_a_valid_checkpoint(self):
        trainer = self.trainer()
        from space_z.rl.training_wooki.settings import TrainSettings
        from space_z.rl.training_wooki.runtime import atomic_json, read_json
        cfg = TrainSettings(total_steps=64, workers=1, rollout_steps=8, batch_size=8,
                            max_steps=8, eval_steps=2, eval_episodes=1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            atomic_json(directory/'control.json', {'command': 'stop'})
            trainer.run_training(directory, cfg, subprocess=False)
            self.assertEqual(read_json(directory/'status.json')['status'], 'stopped')
            self.assertIsNotNone(read_json(directory/'latest.json'))

    def test_resume_keeps_historical_best_after_recent_history_rolls_over(self):
        trainer = self.trainer()
        from space_z.rl.training_wooki.settings import TrainSettings
        with tempfile.TemporaryDirectory() as tmp:
            callback = trainer.LiveCallback(tmp, TrainSettings(), {
                'best': 100., 'evaluations': [{'kind': 'model', 'mean': 1.}]*100})
            self.assertEqual(callback.best, 100.)

    def test_best_checkpoint_score_recovers_from_an_interrupted_status_write(self):
        trainer = self.trainer()
        from space_z.rl.training_wooki.settings import TrainSettings
        from space_z.rl.training_wooki.runtime import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp)/'best.json', {'evaluation': {'mean': 42.}})
            callback = trainer.LiveCallback(tmp, TrainSettings(), {})
            self.assertEqual(callback.best, 42.)


if __name__ == '__main__':
    unittest.main()
