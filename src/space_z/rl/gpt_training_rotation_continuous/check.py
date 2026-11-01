"""디코더 경계·결합 확률·실제 PPO 갱신·저장 복원을 짧게 검증한다."""
from pathlib import Path
import hashlib
import math

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import RotationTrainingEnvironment
from .policy import RotationHybridDistribution, RotationContinuousPolicy
from .train import ROOT, write_json
from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder


def check_evolution_application():
    """진화 선택이 Python int로 전달되어 실제 기체에 적용되는지 확인한다."""
    environment = RotationTrainingEnvironment(stage='aim', max_steps=2, reward_profile='legacy')
    try:
        environment.reset(seed=4100)
        game = environment.unwrapped.game
        required_score = sum(game.config.level_xp_base + (level - 1) * game.config.level_xp_growth
                             for level in range(1, 10))
        game.progression._add_score(required_score)
        assert game.progression.level == 10 and game.progression.reward_choices
        expected_ship = game.progression.reward_choices[0].key
        action = np.array([0, 0.273, 0, 0, 1], dtype=np.float32)
        control = environment.unwrapped.action_decoder.decode(action)
        assert type(control.evolution_choice) is int
        environment.step(action)
        assert game.player.job == expected_ship and game.player.job != 'rookie'
        return dict(evolution_index_type='int', evolved_ship=game.player.job,
                    actual_evolution_applied=True, fixture_score_not_training_data=True)
    finally:
        environment.close()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4100)
    directory = ROOT / 'checks'
    directory.mkdir(exist_ok=True)
    decoder = RotationContinuousActionDecoder()
    control = decoder.decode(np.array([8, 0.273, 1, 5, 4], dtype=np.float32))
    assert abs(control.rotation - 0.273) < 1e-6 and control.want_fire and control.evolution_choice == 3
    assert type(control.evolution_choice) is int
    check_evolution_application()
    for invalid in [[0.5, 0, 0, 0, 0], [0, float('nan'), 0, 0, 0], [0, 1.1, 0, 0, 0],
                    [9, 0, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0], [True] * 5]:
        try:
            decoder.decode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    distribution = RotationHybridDistribution().proba_distribution(torch.zeros(2000, 22), torch.tensor([-1.2]))
    samples = distribution.sample()
    assert len(torch.unique(samples[:, 1])) > 1900
    assert torch.equal(samples[:, [0, 2, 3, 4]], samples[:, [0, 2, 3, 4]].round())
    zero_action = torch.zeros(2000, 5)
    expected = -math.log(9 * 2 * 6 * 5) - 0.5 * math.log(2 * math.pi) + 1.2
    assert torch.allclose(distribution.log_prob(zero_action), torch.full((2000,), expected), atol=1e-6)
    assert torch.isfinite(distribution.log_prob(samples)).all()
    # 별도의 참조식으로 0이 아닌 회전의 Jacobian 보정도 확인한다.
    angles = samples[:, 1]
    reference = (-math.log(9 * 2 * 6 * 5)
                 + torch.distributions.Normal(0, math.exp(-1.2)).log_prob(torch.atanh(angles))
                 - torch.log1p(-angles.square()))
    assert torch.allclose(distribution.log_prob(samples), reference, atol=2e-5)
    environments = DummyVecEnv([lambda: RotationTrainingEnvironment(max_steps=32)])
    try:
        model = PPO(RotationContinuousPolicy, environments, n_steps=32, batch_size=32, n_epochs=2,
                    device='cpu', seed=4100, gamma=math.sqrt(0.995))
        previous = PPO.load(ROOT.parent / 'gpt_training/pure_ppo_model.zip', device='cpu')
        assert previous.training_method == 'pure_ppo' and previous.policy.ignore_aim_health
        model.policy.initialize_from_discrete(previous.policy)
        observation = environments.reset()
        tensor = torch.as_tensor(observation)
        assert torch.allclose(model.policy.aim_features(tensor), previous.policy._aim_geometry(tensor), atol=1e-6)
        previous_distribution = previous.policy.get_distribution(tensor)
        current_distribution = model.policy.get_distribution(tensor)
        assert torch.allclose(current_distribution.fire.probs, previous_distribution.distribution[2].probs[:, 1], atol=1e-6)
        with torch.no_grad():
            actions, _, log_prob = model.policy(tensor)
            _, replay_log_prob, _ = model.policy.evaluate_actions(tensor, actions)
            assert torch.allclose(log_prob, replay_log_prob, atol=1e-6)
            assert actions[0, 3].item() == actions[0, 4].item() == 0
        before = model.policy.aim_net[-1].weight.detach().clone()
        model.learn(64)
        assert not torch.equal(before, model.policy.aim_net[-1].weight)
        assert torch.isfinite(model.policy.log_std).all()
        observation = environments.reset()
        action = model.predict(observation, deterministic=True)[0]
        model.save(directory / 'smoke.zip')
        reloaded = PPO.load(directory / 'smoke.zip', env=environments, device='cpu')
        assert np.array_equal(action, reloaded.predict(observation, deterministic=True)[0])
        assert reloaded.rollout_buffer.gamma == reloaded.gamma
        env = environments.envs[0].unwrapped
        old_angle = env.game.player.angle
        environments.step(np.array([[0, 0.273, 0, 0, 0]], dtype=np.float32))
        assert env.game.player.angle != old_angle
        result = dict(status='passed', smoke_ppo_steps=64, observation_size=5139,
                      continuous_rotation_unique_samples=int(torch.unique(samples[:, 1]).numel()),
                      checkpoint_sha256=hashlib.sha256((directory / 'smoke.zip').read_bytes()).hexdigest(),
                      assertions=['decoder_bounds', 'integral_choices', 'continuous_sampling', 'joint_log_probability',
                                  'tanh_jacobian', 'baseline_aim_features', 'fire_probability_transfer',
                                  'growth_masks', 'actual_game_evolution', 'rollout_log_probability_replay', 'actual_ppo_update',
                                  'save_load_predictions', 'gamma_buffer', 'actual_game_rotation'])
        write_json(directory / 'results.json', result)
        print(result)
    finally:
        environments.close()


if __name__ == '__main__':
    main()
