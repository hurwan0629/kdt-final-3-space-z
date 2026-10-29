import importlib
import importlib.util
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional training dependencies are not installed')
class RuntimeTests(unittest.TestCase):
    def runtime(self):
        self.assertIsNotNone(importlib.util.find_spec('space_z.rl.training.runtime'),
                             'The checkpoint/runtime implementation must exist')
        return importlib.import_module('space_z.rl.training.runtime')

    def test_runtime_exists(self):
        self.runtime()

    def test_status_reader_tolerates_partial_json_and_cleans_nonfinite_metrics(self):
        r = self.runtime()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'state.json'
            p.write_text('{', encoding='utf-8')
            self.assertEqual(r.read_json(p, {'state': 'waiting'}), {'state': 'waiting'})
            r.atomic_json(p, {'x': np.float32(1.5), 'loss': float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'x': 1.5, 'loss': None})

    def test_model_and_normalization_roundtrip_preserves_predictions_and_history(self):
        r = self.runtime()
        from space_z.rl.training.settings import TrainSettings
        cfg = TrainSettings(workers=1, rollout_steps=8, batch_size=8, history=4,
                            max_steps=8, eval_steps=8, eval_episodes=1)
        env = r.make_vec_env(cfg, subprocess=False)
        model = r.make_model(cfg, env)
        try:
            obs = env.reset()
            self.assertEqual(obs['vector'].shape, (1, 1060))
            self.assertEqual(obs['grid'].shape, (1, 4, 21, 21))
            self.assertTrue(np.all(obs['vector'][:, :-265] == 0))
            with tempfile.TemporaryDirectory() as tmp:
                entry = r.publish_checkpoint(model, env, Path(tmp), cfg)
                loaded, norm = r.load_checkpoint(Path(tmp), entry)
                a, _ = model.predict(obs, deterministic=True)
                b, _ = loaded.predict(obs, deterministic=True)
                np.testing.assert_array_equal(a, b)
                self.assertFalse(norm.training)
                norm.close()
        finally:
            env.close()

    def test_evaluation_does_not_update_training_observation_statistics(self):
        r = self.runtime()
        from space_z.rl.training.settings import TrainSettings
        cfg = TrainSettings(workers=1, rollout_steps=8, batch_size=8,
                            max_steps=8, eval_steps=2, eval_episodes=1)
        env = r.make_vec_env(cfg, subprocess=False)
        model = r.make_model(cfg, env)
        try:
            env.reset()
            norm = r.normalizer(env)
            count = float(norm.obs_rms['vector'].count)
            result = r.evaluate(model, env, cfg)
            self.assertEqual(float(norm.obs_rms['vector'].count), count)
            self.assertEqual(len(result['episodes']), 1)
            self.assertEqual(result['episodes'][0]['stage'], 2)
            self.assertEqual(result['episodes'][0]['steps'], 2)
        finally:
            env.close()

    def test_dqn_replay_receives_full_stacked_observations(self):
        r = self.runtime()
        from space_z.rl.training.settings import TrainSettings
        cfg = TrainSettings(algorithm='dqn', workers=1, max_steps=8)
        env = r.make_vec_env(cfg)
        model = r.make_model(cfg, env)
        try:
            model.learn(8)
            self.assertEqual(model.replay_buffer.size(), 8)
            self.assertEqual(model.replay_buffer.observations['vector'].shape[-1], 1060)
        finally:
            env.close()

    def test_corrupt_statistics_closes_the_environment_created_for_loading(self):
        r = self.runtime()
        from space_z.rl.training.settings import TrainSettings
        cfg = TrainSettings(workers=1, rollout_steps=8, batch_size=8)
        env = r.make_vec_env(cfg)
        model = r.make_model(cfg, env)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = r.publish_checkpoint(model, env, root, cfg)
                (root/entry['directory']/'obs_stats.pkl').write_bytes(b'broken')
                raw = r.base_vec(cfg)
                with patch.object(raw, 'close', wraps=raw.close) as close:
                    with patch.object(r, 'base_vec', return_value=raw):
                        with self.assertRaises(pickle.UnpicklingError):
                            r.load_checkpoint(root, entry)
                    try:
                        self.assertTrue(close.called, 'Failed checkpoint loads must close their environments')
                    finally:
                        raw.close()
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
