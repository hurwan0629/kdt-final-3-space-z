"""학습 폴더 import 없이 두 모델의 실제 예측·진화·설정 호환성을 검사한다."""
from contextlib import closing
import importlib.abc
import json
import sys


class RejectOtherRLFolders(importlib.abc.MetaPathFinder):
    """배포 실행이 기존 RL 패키지를 몰래 사용하는 경우 바로 실패시킨다."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'space_z.rl' or fullname.startswith('space_z.rl.'):
            raise ImportError(f'Packaged replay must not import {fullname}')
        return None


sys.meta_path.insert(0, RejectOtherRLFolders())

import numpy as np
import torch
from stable_baselines3.common.utils import check_for_correct_spaces
from play import load_model, make_environment, read_manifest


def main():
    torch.set_num_threads(1)
    manifest = read_manifest()
    results = {}
    for name in ['discrete', 'rotation']:
        model = load_model(name, manifest)
        with closing(make_environment(name, manifest, headless=True)) as environment:
            check_for_correct_spaces(environment, model.observation_space, model.action_space)
            observation, info = environment.reset(seed=61001)
            for _ in range(120):
                action, _ = model.predict(observation, deterministic=True)
                assert np.isfinite(action).all() and model.action_space.contains(action)
                observation, _, terminated, truncated, info = environment.step(action)
                assert observation.shape == (5139,) and np.isfinite(observation).all()
                assert not terminated and not truncated
            game = environment.game
            required_score = sum(game.config.level_xp_base + (level - 1) * game.config.level_xp_growth
                                 for level in range(1, 10))
            game.progression._add_score(required_score)
            expected_ship = game.progression.reward_choices[0].key
            growth_action = np.array([0, 4 if name == 'discrete' else 0.273, 0, 0, 1],
                                    dtype=np.int64 if name == 'discrete' else np.float32)
            control = environment.action_decoder.decode(growth_action)
            assert type(control.evolution_choice) is int
            environment.step(growth_action)
            assert game.player.job == expected_ship
            results[name] = dict(prediction_steps=120, game_frames=info['frames'],
                                 evolved_ship=game.player.job, original_rl_package_imported=False)
    assert not any(name == 'space_z.rl' or name.startswith('space_z.rl.') for name in sys.modules)
    print(json.dumps(dict(status='passed', models=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
