"""독립 평가 전에 개발 성적만으로 모델·재생 방식을 고정한다."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import torch
from stable_baselines3 import PPO

from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder
from .policy import RotationContinuousPolicy
from .report import compare
from .train import ROOT, save_model, write_json


def main():
    torch.set_num_threads(1)
    selection_path = ROOT / 'candidate_selection.json'
    if selection_path.exists() or (ROOT / 'heldout_candidate.json').exists():
        raise FileExistsError('Candidate selection is frozen; do not reselect after independent evaluation')

    # 학습 예산·평가 완료 확인
    experiments = [json.loads(path.read_text(encoding='utf-8'))
                   for path in sorted((ROOT / 'experiments').glob('*/experiment.json'))]
    assert len(experiments) == 5
    assert all(experiment['status'] in ('completed', 'stopped') for experiment in experiments)
    assert all(experiment['training_method'] == 'pure_ppo' for experiment in experiments)
    research_steps = sum(experiment['additional_steps'] for experiment in experiments)
    assert research_steps == 2000000
    assert (ROOT / '005_extra_validation_exit_code.txt').read_text(encoding='utf-8-sig').strip() == '0'
    protected_sources = json.loads((ROOT / 'scope_before.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
               for path, expected in protected_sources.items())
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8-sig'))
    development_seeds = sorted(protocol['broader_development_seeds'])
    baseline_report = json.loads((ROOT / 'baseline_development.json').read_text(encoding='utf-8'))[0]
    baseline_episodes = sorted([episode for episode in baseline_report['episodes']
                                if episode['seed'] in development_seeds], key=lambda episode: episode['seed'])
    assert [episode['seed'] for episode in baseline_episodes] == development_seeds

    # 같은 개발 조건의 후보를 중복 없이 비교
    report_paths = sorted(set(ROOT.glob('*_broader_development.json'))
                          | set(ROOT.glob('*_sampled_development*.json')))
    candidates = []
    seen = set()
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding='utf-8'))
        if not isinstance(payload, list) or len(payload) != 1:
            continue
        report = payload[0]
        if report.get('max_seconds') != 200 or report.get('stage') != 'full' or report.get('frame_skip') != 2:
            continue
        episodes = sorted(report['episodes'], key=lambda episode: episode['seed'])
        if [episode['seed'] for episode in episodes] != development_seeds:
            continue
        mode = report.get('execution_mode', 'greedy')
        assert mode in ('greedy', 'sampled')
        identity = (report['checkpoint_sha256'], mode)
        if identity in seen:
            continue
        seen.add(identity)
        checkpoint_path = Path(report['checkpoint'])
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == report['checkpoint_sha256']
        comparison = compare(baseline_episodes, episodes)
        summary = comparison['candidate']
        candidates.append(dict(
            report=report_path.name, checkpoint=str(checkpoint_path), checkpoint_sha256=report['checkpoint_sha256'],
            replay_mode=mode, development_comparison=comparison,
            ranking=[summary['score_at_120_seconds']['mean'] + summary['score_at_120_seconds']['p25'],
                     summary['score']['mean'], summary['survival_seconds']['mean']],
        ))
    eligible_candidates = [candidate for candidate in candidates
                           if candidate['development_comparison']['all_guards_passed']]
    if not eligible_candidates:
        raise ValueError('No candidate passed the four development guards')
    selected_candidate = max(eligible_candidates, key=lambda candidate: candidate['ranking'])

    # 가중치·재생 방식 고정
    model = PPO.load(selected_candidate['checkpoint'], device='cpu')
    assert isinstance(model.policy, RotationContinuousPolicy) and model.training_method == 'pure_ppo'
    assert model.observation_space.shape == (5139,)
    assert model.action_space == RotationContinuousActionDecoder().action_space
    sampled = selected_candidate['replay_mode'] == 'sampled'
    shipped_checkpoint = ROOT / 'selected_model.zip'
    if getattr(model, 'stochastic_inference', False) == sampled:
        temporary = ROOT / 'selected_model.tmp.zip'
        shutil.copyfile(selected_candidate['checkpoint'], temporary)
        temporary.replace(shipped_checkpoint)
    else:
        model.stochastic_inference = sampled
        save_model(model, shipped_checkpoint)
    reloaded = PPO.load(shipped_checkpoint, device='cpu')
    assert getattr(reloaded, 'stochastic_inference', False) == sampled
    assert all(torch.equal(tensor, reloaded.policy.state_dict()[name])
               for name, tensor in model.policy.state_dict().items())
    shipped_hash = hashlib.sha256(shipped_checkpoint.read_bytes()).hexdigest()

    # 독립 평가 전 선택 근거 보존
    selection = dict(
        selected_at=datetime.now(timezone.utc).isoformat(), heldout_used_for_selection=False,
        research_ppo_steps=research_steps, verification_ppo_steps=64,
        selection_rule='Pass all four development guards, then maximize 120s mean+p25, final mean score, mean survival',
        selected=selected_candidate, shipped_checkpoint=shipped_checkpoint.name, shipped_sha256=shipped_hash,
        new_action_lineage_steps=model.num_timesteps, initial_parent_training_steps=protocol['parent_training_timesteps'],
        all_candidates=sorted(candidates, key=lambda candidate: candidate['ranking'], reverse=True),
        experiments=[dict(name=experiment['name'], status=experiment['status'],
                          additional_steps=experiment['additional_steps']) for experiment in experiments],
    )
    write_json(selection_path, selection)
    write_json(ROOT / 'selected_model.json', dict(
        status='selected_before_heldout', training_method='pure_ppo', demonstration_steps=0,
        checkpoint=shipped_checkpoint.name, sha256=shipped_hash, replay_mode=selected_candidate['replay_mode'],
        game_fps=60, decision_frame_skip=2, observation_size=5139,
        action_format=['move_category_9', 'continuous_rotation_minus1_to1', 'bernoulli_fire', 'stat_category_6', 'evolution_category_5'],
        research_ppo_steps=research_steps, verification_ppo_steps=64,
        new_action_lineage_steps=model.num_timesteps, initial_parent_training_steps=protocol['parent_training_timesteps'],
        selection_record=selection_path.name, original_default_model_replaced=False,
    ))
    print(json.dumps(dict(checkpoint=selected_candidate['checkpoint'], replay_mode=selected_candidate['replay_mode'],
                          shipped_sha256=shipped_hash, development=selected_candidate['development_comparison']['candidate']), indent=2))


if __name__ == '__main__':
    main()
