"""구조화 관찰·성장 마스크·보상·실제 PPO 업데이트의 계약을 확인한다."""
from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO

from .environment import make_environment
from .networks import GrowthMaskedPolicy, SpatialFeaturesExtractor
from .score_reward import ScoreReward, reward_settings
from .train import ROOT, evaluate, save_model
from .improve import checkpoint_ranking


def check_score_milestones():
    for end_step in (75, 220):
        with patch('space_z.rl.gpt_training.train.make_environment') as factory:
            environment = factory.return_value
            environment.unwrapped.game_config.fps = 1
            environment.unwrapped.environment_config.frame_skip = 1
            environment.reset.return_value = (np.zeros(1), {})
            environment.step.side_effect = [
                (np.zeros(1), 0, step == end_step, False,
                 dict(frames=step, score=10*step, kills=0,
                      experiment_episode=dict(shots=0, shapes_destroyed=0, damage_taken=0)))
                for step in range(1, end_step+1)
            ]
            report = evaluate(lambda observation: 0, [1], 220)
            episode = report['episodes'][0]
            for seconds in (60, 120, 200):
                assert episode[f'score_at_{seconds}_seconds'] == 10*min(seconds, end_step)
            assert report['summary']['score']['p25'] == 10*end_step
            expected_early = 10*min(120, end_step)
            assert checkpoint_ranking(report, 'early_stability')[0] == 2*expected_early
            assert checkpoint_ranking(report, 'score')[0] == 10*end_step
    print('PASS early-score milestones, early-death fallback, percentile summaries and selection rules')


def main():
    torch.set_num_threads(1)
    aligned_discount = reward_settings('score_v7').gamma
    assert abs(aligned_discount**2 - reward_settings('score_v6').gamma) < 1e-12
    check_score_milestones()
    from . import capture as capture_cli, evaluate as evaluate_cli, play as play_cli
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        for module in (capture_cli, evaluate_cli, play_cli):
            arguments = [module.__name__, str(ROOT / 'pure_ppo_model.zip')]
            if module is not play_cli:
                arguments.extend(['--output', str(Path(temporary) / 'unused')])
            for invalid in (['--max-seconds', 'nan'], ['--max-seconds', '1', '--max-steps', '2'], ['--frame-skip', '0']):
                with patch.object(sys, 'argv', arguments + invalid), redirect_stderr(StringIO()):
                    try:
                        module.main()
                    except SystemExit as error:
                        assert error.code == 2
                    else:
                        raise AssertionError('Invalid duration arguments accepted')
        assert not (Path(temporary) / 'unused').exists()
    print('PASS evaluation/play/capture duration arguments reject nonfinite, ambiguous and invalid-frequency inputs')
    frequency_environment = make_environment('full', 3, pure_rl=True, frame_skip=2)
    try:
        observation, _ = frequency_environment.reset(seed=123)
        for _ in range(3):
            observation, _, terminated, truncated, info = frequency_environment.step(np.array([0, 4, 0, 0, 0]))
        assert observation.shape == (5139,) and info['frames'] == 6 and truncated and not terminated
        print('PASS decision frequency: three actions advance six native game frames with unchanged observation shape')
    finally:
        frequency_environment.close()
    scenarios = [(stage, 'score_v2') for stage in ['aim', 'duel', 'arena', 'full']] + [('full', 'score_v3'), ('arena', 'score_v4'), ('range', 'score_v5'), ('moving_range', 'score_v5'), ('full', 'score_v6'), ('full', 'score_v7')]
    for stage, profile in scenarios:
        environment = make_environment(stage, 12, pure_rl=True, reward_profile=profile)
        try:
            observation, _ = environment.reset(seed=123)
            repeated, _ = environment.reset(seed=123)
            np.testing.assert_array_equal(observation, repeated)
            assert observation.shape == (5139,)
            for _ in range(12):
                observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
                assert environment.observation_space.contains(observation) and np.isfinite(reward)
                if terminated or truncated:
                    episode = info['experiment_episode']
                    assert abs(episode['training_reward'] - sum(value for key, value in episode.items() if key.startswith('reward_'))) < 1e-6
                    break
            print(f'PASS {stage}/{profile}: original observations, deterministic reset, reward component accounting')
        finally:
            environment.close()
    environment = make_environment('aim', 24, pure_rl=True, reward_profile='score_v2')
    try:
        model = PPO(GrowthMaskedPolicy, environment, n_steps=16, batch_size=16, n_epochs=1, seed=123,
                    policy_kwargs={'features_extractor_class': SpatialFeaturesExtractor})
        model.training_method = 'pure_ppo'
        observation, _ = environment.reset(seed=4)
        tensor = torch.as_tensor(observation[None])
        for _ in range(30):
            action, value, log_probability = model.policy(tensor)
            assert action[0, 3] == 0 and action[0, 4] == 0
            expected_value, expected_log, entropy = model.policy.evaluate_actions(tensor, action)
            torch.testing.assert_close(log_probability, expected_log)
            assert torch.isfinite(value).all() and torch.isfinite(entropy).all()
        # 강화·진화 가능 여부만 바꾸면 해당 선택지를 제외하지 않아야 한다.
        tensor[:, 40:45] = 1
        tensor[:, [45, 67, 89, 111]] = 1
        distribution = model.policy.get_distribution(tensor)
        assert (distribution.distribution[3].probs > 0).all()
        assert (distribution.distribution[4].probs > 0).all()
        before = [category.probs.detach().clone() for category in distribution.distribution]
        model.policy.enable_aim_residual()
        model.policy_kwargs['aim_residual'] = True
        after = model.policy.get_distribution(tensor)
        for expected, actual in zip(before, after.distribution):
            torch.testing.assert_close(expected, actual.probs)
        original = model.policy.features_extractor.terrain[0].weight.detach().clone()
        model.learn(32)
        assert not torch.equal(original, model.policy.features_extractor.terrain[0].weight)
        assert torch.count_nonzero(model.policy.aim_residual[-1].weight) > 0
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / 'check.zip'
            model.decision_frame_skip = 2
            save_model(model, checkpoint)
            restored = PPO.load(checkpoint, env=environment, gamma=aligned_discount)
            assert restored.gamma == restored.rollout_buffer.gamma == aligned_discount
            assert restored.decision_frame_skip == 2
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
        print('PASS structured CNN gradients, growth masking, PPO learning, ZIP reload and time-aligned rollout discount')
        shaping = ScoreReward()
        shaping.reset(environment.unwrapped.game, observation)
        initial_potential = shaping.previous_potential
        events = dict(shape_damage_weighted=0, enemy_damage=0, damage_taken=0, player_shots_fired=0)
        first = shaping.calculate(environment.unwrapped.game, observation, 0, events, False)['reward_aim_progress']
        last = shaping.calculate(environment.unwrapped.game, observation, 0, events, True)['reward_aim_progress']
        assert abs(first + shaping.settings.gamma * last + initial_potential) < 1e-6
        print('PASS potential shaping telescopes and terminal potential is zero')
        shaped = ScoreReward(reward_settings('score_v3'))
        artificial = np.zeros(5139, dtype=np.float32)
        artificial[4] = 1
        artificial[412:415] = [1, 300/1080, 0]
        artificial[418] = 22/48
        game = environment.unwrapped.game
        assert shaped.shot_quality(game, artificial) > 0.99
        reversed_observation = artificial.copy()
        reversed_observation[4] = -1
        assert shaped.shot_quality(game, reversed_observation) < 1e-8
        shaped.reset(game, artificial)
        components = shaped.calculate(game, artificial, 0, events, False)
        assert components['reward_aimed_shot'] == 0
        events['player_shots_fired'] = 10
        components = shaped.calculate(game, artificial, 0, events, False)
        assert abs(components['reward_aimed_shot'] - shaped.settings.aimed_shot_bonus) < 1e-6
        artificial[712:718] = [1, 250/1080, 0, -1, 0, 6/48]
        incoming = shaped.projectile_threat(game, artificial)
        artificial[715] = 1
        assert incoming > shaped.projectile_threat(game, artificial)
        print('PASS score_v3: aligned-shot reward, no reward without firing, pellet cap and incoming-projectile risk')
        focused = ScoreReward(reward_settings('score_v4'))
        artificial[422:425] = [1, -150/1080, 0]
        artificial[428] = 22/48
        assert shaped.shot_quality(game, artificial) > 0.99
        assert focused.shot_quality(game, artificial) < 1e-8
        print('PASS score_v4 nearest-entity focus and zero-initialized PPO aim residual')
    finally:
        environment.close()

    environment = make_environment('range', 80, pure_rl=True, reward_profile='score_v5')
    try:
        observation, _ = environment.reset(seed=8)
        player = environment.unwrapped.game.player
        position = (player.x, player.y)
        for _ in range(8):
            environment.step(np.array([8, 4, 0, 0, 0]))
        assert position == (player.x, player.y)
        shape = environment.unwrapped.game.shapes[0]
        assert 220 <= np.hypot(shape.x-player.x, shape.y-player.y) <= 480 and shape.max_hp == 7
        model = PPO(GrowthMaskedPolicy, environment, n_steps=16, batch_size=16, n_epochs=1, seed=123,
                    policy_kwargs={'features_extractor_class': SpatialFeaturesExtractor,
                                   'aim_residual': True, 'freeze_base_actor': True})
        original_encoder = {key: value.clone() for key, value in model.policy.features_extractor.state_dict().items()}
        original_actions = model.policy.action_net.weight.detach().clone()
        model.learn(64)
        for key, value in model.policy.features_extractor.state_dict().items():
            torch.testing.assert_close(value, original_encoder[key])
        torch.testing.assert_close(model.policy.action_net.weight, original_actions)
        assert torch.count_nonzero(model.policy.aim_residual[-1].weight) > 0
        tensor = torch.as_tensor(observation[None])
        previous_geometry = model.policy._aim_geometry(tensor)
        model.policy.unified_aim_input = True
        model.policy_kwargs['unified_aim_input'] = True
        torch.testing.assert_close(previous_geometry, model.policy._aim_geometry(tensor))
        enemy_observation = tensor.clone()
        enemy_observation[:, 133:412] = 0
        enemy_observation[:, 412:712] = 0
        enemy_observation[:, 133:139] = tensor[:, 412:418]
        enemy_observation[:, 141] = tensor[:, 418]
        torch.testing.assert_close(model.policy._aim_geometry(tensor), model.policy._aim_geometry(enemy_observation))
        blocked = tensor.clone()
        blocked[:, 1224:2529] = 1
        assert torch.count_nonzero(model.policy._aim_geometry(blocked)[:, 133:]) == 0
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'frozen.zip'
            save_model(model, path)
            restored = PPO.load(path, env=environment)
            assert restored.policy.base_actor_frozen
            assert restored.policy.unified_aim_input
            assert all(not parameter.requires_grad for parameter in restored.policy.features_extractor.parameters())
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
            restored.policy.set_actor_frozen(False)
            assert all(parameter.requires_grad for parameter in restored.policy.features_extractor.parameters())
        print('PASS stationary range, randomized spawn, frozen base preservation, PPO-only residual learning and reload/unfreeze')
        print('PASS unified aiming input: equivalent one-shape geometry, enemy/shape sharing and observed-grid visibility')
        model.policy.enable_independent_aim()
        model.policy_kwargs.update(independent_aim=True, unified_aim_input=True)
        model.learn(64)
        assert torch.count_nonzero(model.policy.aim_residual[-1].weight) > 0
        shape_distribution = model.policy.get_distribution(tensor)
        enemy_distribution = model.policy.get_distribution(enemy_observation)
        for branch in [1, 2]:
            torch.testing.assert_close(shape_distribution.distribution[branch].probs,
                                       enemy_distribution.distribution[branch].probs)
        changed_velocity = tensor.clone()
        changed_velocity[:, :2] = 1
        changed_distribution = model.policy.get_distribution(changed_velocity)
        for branch in [1, 2]:
            torch.testing.assert_close(shape_distribution.distribution[branch].probs,
                                       changed_distribution.distribution[branch].probs)
        model.policy.set_actor_frozen(False)
        model.policy.set_aim_frozen(True)
        model.policy.ignore_aim_health = True
        model.policy_kwargs.update(freeze_base_actor=False, freeze_aim=True, ignore_aim_health=True)
        changed_health = tensor.clone()
        changed_health[:, 417] = 0.2
        changed_health[:, 35] = 0.8
        torch.testing.assert_close(model.policy._aim_geometry(tensor), model.policy._aim_geometry(changed_health))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'independent.zip'
            save_model(model, path)
            restored = PPO.load(path, env=environment)
            assert restored.policy.independent_aim and restored.policy.aim_frozen
            assert not restored.policy.base_actor_frozen
            assert all(not parameter.requires_grad for parameter in restored.policy.aim_residual.parameters())
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
            fixed_aim = [parameter.detach().clone() for parameter in restored.policy.aim_residual.parameters()]
            previous_grid_weight = restored.policy.features_extractor.terrain[0].weight.detach().clone()
            restored.learn(64)
            for previous, current in zip(fixed_aim, restored.policy.aim_residual.parameters()):
                torch.testing.assert_close(previous, current, rtol=0, atol=0)
            assert not torch.equal(previous_grid_weight, restored.policy.features_extractor.terrain[0].weight)
        print('PASS independent aiming: scene/type invariance, frozen-aim reload and PPO-only controller updates')
        print('PASS full-body PPO update: aiming weights unchanged while terrain CNN learns')
    finally:
        environment.close()


if __name__ == '__main__':
    main()
