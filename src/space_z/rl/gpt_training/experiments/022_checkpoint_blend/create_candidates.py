"""고정된 순수 PPO 부모 가중치를 섞어 변화 폭이 작은 세 후보를 만든다."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from stable_baselines3 import PPO

from space_z.rl.gpt_training.environment import make_environment
from space_z.rl.gpt_training.train import save_model, write_json


def main():
    root = Path(__file__).resolve().parents[2]
    directory = Path(__file__).resolve().parent
    protocol = json.loads((root / 'phase3_blend_protocol.json').read_text(encoding='utf-8'))
    parents = {}
    torch.set_num_threads(1)
    for name in ('baseline', 'trained'):
        checkpoint = root / protocol[f'{name}_checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol[f'{name}_sha256']
        parents[name] = PPO.load(checkpoint, device='cpu').policy.state_dict()
    assert parents['baseline'].keys() == parents['trained'].keys()
    (directory / 'checkpoints').mkdir(exist_ok=False)
    with make_environment('full', 1, pure_rl=True, frame_skip=2) as environment:
        observation, _ = environment.reset(seed=505)
    candidates = []
    for trained_weight in protocol['candidate_trained_weights']:
        model = PPO.load(root / protocol['trained_checkpoint'], device='cpu')
        combined = {}
        for name, original in parents['baseline'].items():
            trained = parents['trained'][name]
            if torch.equal(original, trained):
                combined[name] = original.clone()
            else:
                assert original.is_floating_point() and original.shape == trained.shape
                combined[name] = torch.lerp(original, trained, trained_weight)
        model.policy.load_state_dict(combined)
        # 바뀐 가중치로 다시 학습할 때 부모의 Adam 모멘트를 이어 쓰지 않는다.
        model.policy.optimizer.state.clear()
        model.checkpoint_blend = {'baseline': protocol['baseline_checkpoint'],
                                  'trained': protocol['trained_checkpoint'],
                                  'trained_weight': trained_weight, 'additional_ppo_steps': 0}
        model.research_version = 'phase3_checkpoint_blend'
        checkpoint = directory / 'checkpoints' / f'blend_{round(trained_weight * 100):03d}.zip'
        save_model(model, checkpoint)
        restored = PPO.load(checkpoint, device='cpu')
        assert restored.training_method == 'pure_ppo' and restored.decision_frame_skip == 2
        assert restored.observation_space.shape == (5139,)
        assert restored.action_space.nvec.tolist() == [9, 9, 2, 6, 5]
        actual = restored.policy.state_dict()
        assert all(torch.equal(value, actual[name]) for name, value in combined.items())
        assert all(torch.equal(value, actual[name]) for name, value in parents['baseline'].items()
                   if name.startswith('aim_residual.'))
        assert (model.predict(observation, deterministic=True)[0]
                == restored.predict(observation, deterministic=True)[0]).all()
        candidates.append({'checkpoint': str(checkpoint.relative_to(root)), 'trained_weight': trained_weight,
                           'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    with ZipFile(directory / 'source_code.zip', 'w', ZIP_DEFLATED) as archive:
        for source in sorted(root.glob('*.py')):
            archive.write(source, source.name)
        archive.write(__file__, 'create_candidates.py')
    manifest = {'name': directory.name, 'status': 'candidates_created',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'goal': 'Preserve earlier policy stability while testing PPO weight averaging',
                'experiment_kind': 'checkpoint_weight_averaging', 'additional_ppo_steps': 0,
                'parent_training_timesteps': model.num_timesteps, 'demonstration_steps': 0,
                'recipe': 'torch.lerp(baseline_tensor, trained_tensor, trained_weight); copy identical tensors exactly',
                'optimizer_moments_cleared': True, 'protocol': 'phase3_blend_protocol.json',
                'candidates': candidates,
                'source_sha256': hashlib.sha256((directory / 'source_code.zip').read_bytes()).hexdigest()}
    write_json(directory / 'experiment.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
