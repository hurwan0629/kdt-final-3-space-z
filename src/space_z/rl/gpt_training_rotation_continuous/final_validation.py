"""최종 모델·재생·로그·수정 범위를 함께 검증하고 소스 사본을 보존한다."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .policy import RotationContinuousPolicy
from .train import ROOT, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    project = ROOT.parents[3]
    decoder = ROOT.parent / 'actions/rotation_continuous_action_decoder.py'
    sources = sorted(ROOT.glob('*.py')) + [decoder]
    for source in sources:
        ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    protected = read(ROOT / 'scope_before.json')
    assert len(protected) == 40
    for relative, expected in protected.items():
        assert digest(project / relative) == expected, relative
    selection = read(ROOT / 'corrected_candidate_selection.json')
    protocol = read(ROOT / 'decoder_correction_protocol.json')
    card = read(ROOT / 'selected_model.json')
    candidate = read(ROOT / 'corrected_heldout_candidate.json')[0]
    baseline = read(ROOT / 'corrected_heldout_baseline.json')[0]
    comparison = read(ROOT / 'corrected_heldout_comparison.json')
    model_path = ROOT / 'selected_model.zip'
    assert digest(model_path) == selection['shipped_sha256'] == candidate['checkpoint_sha256'] == card['sha256']
    assert digest(decoder) == protocol['decoder_sha256'] == candidate['decoder_source_sha256']
    assert not selection['heldout_used_for_selection']
    assert comparison['all_guards_passed']
    assert baseline['checkpoint_sha256'] == protocol['parent_sha256']
    assert [episode['seed'] for episode in candidate['episodes']] == protocol['heldout_seeds']
    replay_results = []
    for seed in [505, 61001]:
        directory = ROOT / f'visual_corrected_seed{seed}'
        metadata = read(directory / 'metadata.json')
        frames = read(directory / 'frames.json')
        assert metadata['checkpoint_sha256'] == digest(model_path)
        assert metadata['decoder_source_sha256'] == digest(decoder)
        assert metadata['replay_mode'] == 'greedy'
        assert any(frame['ship'] != 'rookie' for frame in frames)
        for frame in frames:
            assert (directory / frame['image']).stat().st_size > 1000
        if seed == 61001:
            episode = next(episode for episode in candidate['episodes'] if episode['seed'] == seed)
            for replay_key, evaluation_key in [('score', 'score'), ('kills', 'kills'), ('seconds', 'survival_seconds')]:
                assert frames[-1][replay_key] == episode[evaluation_key], (replay_key, frames[-1], episode)
        replay_results.append(frames[-1])
    experiments = [read(path) for path in sorted(ROOT.glob('experiments/*/experiment.json'))]
    assert len(experiments) == 5
    assert sum(record['additional_steps'] for record in experiments) == 2000000
    assert all(record['training_method'] == 'pure_ppo' and record['demonstration_steps'] == 0 for record in experiments)
    for path in ROOT.glob('experiments/*/source_code.zip'):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
    assert card['verification_ppo_steps'] == 128
    assert card['additional_training_after_fix'] == 0
    assert read(ROOT / 'checks/evolution_application.json')['actual_evolution_applied']
    assert read(ROOT / 'checks/results.json')['status'] == 'passed'
    assert (ROOT / 'checks/corrected_full_check_exit_code.txt').read_text(encoding='utf-8-sig').strip() == '0'
    model = PPO.load(model_path, device='cpu')
    parent = PPO.load(ROOT / 'experiments/003_body_stability/checkpoints/evaluated_000800000.zip', device='cpu')
    assert isinstance(model.policy, RotationContinuousPolicy)
    assert model.observation_space.shape == (5139,)
    assert np.array_equal(model.action_space.low, [0, -1, 0, 0, 0])
    assert np.array_equal(model.action_space.high, [8, 1, 1, 5, 4])
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    fixed_keys = [key for key in model.policy.state_dict()
                  if key.startswith(('features_extractor.', 'mlp_extractor.policy_net.', 'action_net.'))]
    assert fixed_keys
    assert all(torch.equal(model.policy.state_dict()[key], parent.policy.state_dict()[key]) for key in fixed_keys)
    events = EventAccumulator(str(ROOT / 'tensorboard/corrected_final_comparison')).Reload()
    assert len(events.Tags()['scalars']) == 16
    assert events.Tags()['images'] and len(events.Tags()['tensors']) == 2
    for side in ['baseline', 'candidate']:
        for metric, statistics in comparison[side].items():
            for statistic, expected in statistics.items():
                scalar = events.Scalars(f'heldout/{side}/{metric}_{statistic}')[-1]
                assert scalar.step == 2000000 and np.isclose(scalar.value, expected, atol=0.001)
    monitor = [json.loads(line) for line in (ROOT / 'all_process_monitor.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    maximum = max(record['owned_process_count'] for record in monitor)
    assert maximum <= 8
    assert card['process_limit_respected_entire_study'] is False
    archive_path = ROOT / 'final_source_code.zip'
    snapshot = sorted(set(sources + list(ROOT.glob('*.ps1')) + list(ROOT.glob('*.md'))
                          + [ROOT / name for name in ['protocol.json', 'decoder_correction_protocol.json',
                             'corrected_candidate_selection.json', 'selected_model.json', 'versions.json',
                             'environment.lock.txt']] + [project / relative for relative in protected
                                                        if Path(relative).suffix == '.py']))
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in snapshot:
            archive.write(source, str(source.relative_to(project)))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    write_json(ROOT / 'checks/final_validation.json', dict(
        status='passed', protected_files_unchanged=len(protected), parsed_python_files=len(sources),
        research_ppo_steps=2000000, verification_ppo_steps=128, demonstration_steps=0,
        selected_model_sha256=digest(model_path), decoder_sha256=digest(decoder),
        corrected_replays=replay_results, evaluation_replay_exact_match_seed=61001,
        fixed_body_tensor_count=len(fixed_keys), tensorboard_scalar_count=16,
        maximum_owned_processes_since_expanded_monitor=maximum,
        entire_study_process_limit_respected=False,
        final_source_archive_sha256=digest(archive_path), archived_files=len(snapshot),
    ))
    print('Final validation passed; source archive preserved.')


if __name__ == '__main__':
    main()
