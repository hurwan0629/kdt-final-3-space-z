"""관찰 전용 입력과 attention·PPO 연결을 실제 객체로 확인한다."""
from copy import deepcopy
import json
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import make_environment, environment_configuration
from .experiment import new_experiment, save_model, update_report
from .policy import VectorOnlyPolicy
from .storage import event, now, sha256, write_json


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = new_experiment('001_contract_checks', '학습 전 관찰·정책·저장 연결 검증', 'verification',
        '5139차원만으로 행동을 만들며 빈 슬롯·개체 순서·저장 복원·PPO 갱신이 정상인지 확인한다.',
        dict(environment=environment_configuration(), verification_steps=128, training_seed=20261101))
    started = time.monotonic()
    results = {}
    environments = DummyVecEnv([lambda: make_environment(2) for _ in range(2)])
    model = None
    try:
        model = PPO(VectorOnlyPolicy, environments, n_steps=32, batch_size=32, n_epochs=1,
                    seed=20261101, device='cpu', verbose=0)
        model.policy.set_training_mode(False)
        observation = torch.as_tensor(environments.reset(), device=model.device)
        extractor = model.policy.features_extractor
        with torch.no_grad():
            features, weights = extractor.encode(observation, return_attention=True)
            assert tuple(features.shape) == (2, 64)
            assert tuple(weights.shape) == (2, 4, 1, 143)
            assert torch.isfinite(features).all()
            assert torch.allclose(weights.sum(-1), torch.ones((2, 4, 1)), atol=1e-6)
            results['tensor_shapes'] = dict(passed=True, observation=[2, 5139], features=list(features.shape),
                                            attention_weights=list(weights.shape), action_parameters=22)
            permuted = observation.clone()
            for start, end, count, size in [(133,412,9,31),(412,712,30,10),(712,1224,64,8)]:
                values = permuted[:, start:end].reshape(2,count,size).clone()
                permuted[:, start:end] = values[:, torch.arange(count-1,-1,-1)].reshape(2,-1)
            error = float((extractor(permuted) - features).abs().max())
            assert error < 1e-5, error
            results['entity_order_invariance'] = dict(passed=True, maximum_absolute_error=error)
            masked = observation.clone()
            masked[:,133:412] = 0
            baseline = extractor(masked)
            changed = masked.clone()
            enemy = changed[:,133:412].reshape(2,9,31)
            enemy[:,:,1:] = torch.linspace(-1,1,30).reshape(1,1,30)
            error = float((extractor(changed) - baseline).abs().max())
            assert error < 1e-7, error
            results['empty_slot_invariance'] = dict(passed=True, maximum_absolute_error=error)
            actions, values, probabilities = model.policy(observation)
            assert tuple(actions.shape) == (2,5) and torch.isfinite(probabilities).all()
            for action in actions.cpu().numpy():
                environments.envs[0].unwrapped.action_decoder.decode(action)
            results['hybrid_actions'] = dict(passed=True, shape=list(actions.shape), finite_log_probability=True)
        before = deepcopy(model.policy.features_extractor.state_dict())
        model.learn(128)
        changed = any(not torch.equal(before[name], value) for name, value in extractor.state_dict().items())
        assert changed and all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
        results['ppo_update'] = dict(passed=True, steps=model.num_timesteps, encoder_weights_changed=changed)
        checkpoint = directory / 'checkpoints/integration.zip'
        save_model(model,checkpoint)
        restored = PPO.load(checkpoint,device='cpu')
        observations = observation.cpu().numpy()
        expected,_ = model.predict(observations,deterministic=True)
        actual,_ = restored.predict(observations,deterministic=True)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
        results['checkpoint_restore'] = dict(passed=True, checkpoint='checkpoints/integration.zip', sha256=sha256(checkpoint))
        protected = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['shared_source_sha256']
        project = directory.parents[5]
        assert all(sha256(project / source)==expected for source,expected in protected.items())
        results['protected_sources'] = dict(passed=True, file_count=len(protected))
        write_json(directory / 'checks.json',results)
        write_json(directory / 'state.json',dict(status='completed',timesteps=128,verification_steps=128,
                   updated_at=now(),elapsed_seconds=time.monotonic()-started,checks_passed=len(results)))
        event(directory,'completed','모든 연결 검사를 통과했다. 검증용 PPO 128 step은 예비 학습 모델에 이전하지 않는다.',checks=results)
        print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        write_json(directory / 'checks.json',results)
        write_json(directory / 'state.json',dict(status='failed',timesteps=model.num_timesteps if model else 0,
                                               updated_at=now(),error=repr(error)))
        event(directory,'failed',repr(error))
        raise
    finally:
        environments.close()
        update_report()


if __name__ == '__main__':
    main()
