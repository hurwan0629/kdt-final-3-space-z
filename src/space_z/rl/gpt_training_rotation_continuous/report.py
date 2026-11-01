"""평가 JSON에서 짝지은 모델 비교와 학습 경과 그림을 만든다."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .train import ROOT, write_json


def summarize(episodes):
    metrics = ['score_at_120_seconds', 'score', 'survival_seconds', 'kills']
    return {metric: dict(mean=float(np.mean([episode[metric] for episode in episodes])),
                         p25=float(np.percentile([episode[metric] for episode in episodes], 25)))
            for metric in metrics}


def compare(baseline, candidate):
    """같은 seed 순서를 검증하고 평균 변화의 paired bootstrap 구간을 계산한다."""
    assert [episode['seed'] for episode in baseline] == [episode['seed'] for episode in candidate]
    baseline_summary, candidate_summary = summarize(baseline), summarize(candidate)
    random = np.random.default_rng(4100)
    changes = {}
    for metric in baseline_summary:
        differences = np.array([right[metric] - left[metric] for left, right in zip(baseline, candidate)])
        bootstraps = random.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
        changes[metric] = dict(mean_difference=float(differences.mean()),
                               paired_mean_95pct_interval=np.percentile(bootstraps, [2.5, 97.5]).tolist(),
                               wins=int((differences > 0).sum()), ties=int((differences == 0).sum()),
                               losses=int((differences < 0).sum()))
    guards = dict(
        early_mean=candidate_summary['score_at_120_seconds']['mean'] >= baseline_summary['score_at_120_seconds']['mean'],
        early_p25=candidate_summary['score_at_120_seconds']['p25'] >= baseline_summary['score_at_120_seconds']['p25'],
        final_mean=candidate_summary['score']['mean'] >= baseline_summary['score']['mean'],
        mean_survival=candidate_summary['survival_seconds']['mean'] >= baseline_summary['survival_seconds']['mean'],
    )
    return dict(baseline=baseline_summary, candidate=candidate_summary, paired_changes=changes,
                promotion_guards=guards, all_guards_passed=all(guards.values()))


def merge_development_evaluation(experiment_name, timesteps, extra_path, output_name):
    """본 학습기 4개와 별도 평가 4개를 체크포인트 해시 확인 후 합친다."""
    directory = ROOT / 'experiments' / experiment_name
    primary_records = [json.loads(line) for line in (directory / 'evaluations.jsonl').read_text().splitlines()]
    primary = next(record for record in primary_records if record['timesteps'] == timesteps)
    extra = json.loads(Path(extra_path).read_text())[0]
    assert primary['checkpoint_sha256'] == extra['checkpoint_sha256']
    assert primary.get('execution_mode', 'greedy') == extra.get('execution_mode', 'greedy')
    episodes = sorted(primary['episodes'] + extra['episodes'], key=lambda episode: episode['seed'])
    baseline_report = json.loads((ROOT / 'baseline_development.json').read_text())[0]
    baseline = [episode for episode in baseline_report['episodes'] if episode['seed'] >= 43000]
    assert len(episodes) == len({episode['seed'] for episode in episodes}) == 8
    assert [episode['seed'] for episode in episodes] == [episode['seed'] for episode in baseline]
    result = dict(checkpoint=extra['checkpoint'], checkpoint_sha256=extra['checkpoint_sha256'],
                  timesteps=timesteps, max_seconds=200, frame_skip=2, stage='full', episodes=episodes,
                  execution_mode=extra.get('execution_mode', 'greedy'), policy_seed=extra.get('policy_seed'),
                  summary=summarize(episodes))
    write_json(ROOT / (output_name + '.json'), [result])
    comparison = compare(baseline, episodes)
    write_json(ROOT / (output_name + '_comparison.json'), comparison)
    print(json.dumps(comparison))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected', action='store_true')
    args = parser.parse_args()
    prefix = 'corrected_' if args.corrected else ''
    protocol_name = 'decoder_correction_protocol.json' if args.corrected else 'protocol.json'
    baseline = json.loads((ROOT / f'{prefix}heldout_baseline.json').read_text(encoding='utf-8'))[0]
    candidate = json.loads((ROOT / f'{prefix}heldout_candidate.json').read_text(encoding='utf-8'))[0]
    protocol = json.loads((ROOT / protocol_name).read_text(encoding='utf-8-sig'))
    selection = json.loads((ROOT / f'{prefix}candidate_selection.json').read_text(encoding='utf-8'))
    assert len(protocol['heldout_seeds']) == 20
    assert [episode['seed'] for episode in baseline['episodes']] == protocol['heldout_seeds']
    assert [episode['seed'] for episode in candidate['episodes']] == protocol['heldout_seeds']
    assert baseline['max_seconds'] == candidate['max_seconds'] == 400
    assert baseline['frame_skip'] == candidate['frame_skip'] == 2
    assert baseline['checkpoint_sha256'] == protocol['parent_sha256']
    assert candidate['checkpoint_sha256'] == selection['shipped_sha256']
    assert hashlib.sha256((ROOT / 'selected_model.zip').read_bytes()).hexdigest() == selection['shipped_sha256']
    assert candidate.get('execution_mode', 'greedy') == selection['selected']['replay_mode']
    if args.corrected:
        assert candidate['decoder_source_sha256'] == protocol['decoder_sha256']
    comparison = compare(baseline['episodes'], candidate['episodes'])
    comparison.update(seed_count=len(baseline['episodes']), max_seconds=400,
                      baseline_sha256=baseline['checkpoint_sha256'], candidate_sha256=candidate['checkpoint_sha256'])
    comparison['reached_400_seconds_alive'] = {
        'baseline': sum(episode['truncated'] and not episode['terminated'] for episode in baseline['episodes']),
        'candidate': sum(episode['truncated'] and not episode['terminated'] for episode in candidate['episodes']),
    }
    write_json(ROOT / f'{prefix}heldout_comparison.json', comparison)
    with (ROOT / f'{prefix}heldout_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'baseline_score120', 'candidate_score120', 'baseline_score', 'candidate_score', 'baseline_life', 'candidate_life'])
        for previous, current in zip(baseline['episodes'], candidate['episodes']):
            writer.writerow([previous['seed'], previous['score_at_120_seconds'], current['score_at_120_seconds'],
                             previous['score'], current['score'], previous['survival_seconds'], current['survival_seconds']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, metric, title in zip(axes, ['score_at_120_seconds', 'score', 'survival_seconds'],
                                  ['Score at 120 seconds', 'Final score (400s cap)', 'Survival seconds']):
        for previous, current in zip(baseline['episodes'], candidate['episodes']):
            axis.plot([0, 1], [previous[metric], current[metric]], color='#a0a8b4', alpha=0.45, linewidth=1)
        means = [comparison[side][metric]['mean'] for side in ['baseline', 'candidate']]
        axis.plot([0, 1], means, color='#135cba', marker='o', linewidth=3, label='Mean')
        axis.set_xticks([0, 1], ['Previous PPO', 'Continuous rotation'])
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=0.2)
        axis.legend()
    figure.suptitle('20 paired unseen seeds / full game / 30Hz / candidate ' + candidate.get('execution_mode', 'greedy')
                   + (' / corrected evolution' if args.corrected else ''))
    figure.savefig(ROOT / f'{prefix}heldout_comparison.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    development_groups = {}
    expected_seeds = sorted(protocol['broader_development_seeds'])
    for report_path in sorted(ROOT.glob('*_broader_development.json')):
        development = json.loads(report_path.read_text(encoding='utf-8'))[0]
        if sorted(episode['seed'] for episode in development['episodes']) != expected_seeds:
            continue
        assert development['max_seconds'] == 200
        checkpoint_path = Path(development['checkpoint'])
        experiment = checkpoint_path.parent.parent
        step = development.get('timesteps')
        if step is None:
            if checkpoint_path.stem.startswith('evaluated_'):
                step = int(checkpoint_path.stem.removeprefix('evaluated_'))
            else:
                manifest = json.loads((experiment / 'experiment.json').read_text(encoding='utf-8'))
                step = manifest['best_timesteps']
        development_groups.setdefault(experiment.name, []).append((step, summarize(development['episodes'])))
    for experiment_name, records in sorted(development_groups.items()):
        records.sort(key=lambda record: record[0])
        for axis, metric in zip(axes, ['score_at_120_seconds', 'survival_seconds']):
            axis.plot([record[0] for record in records], [record[1][metric]['mean'] for record in records],
                      marker='o', label=experiment_name)
    development_baseline = selection['selected']['development_comparison']['baseline']
    for axis, metric in zip(axes, ['score_at_120_seconds', 'survival_seconds']):
        axis.axhline(development_baseline[metric]['mean'], color='#555555', linestyle='--', label='Previous model')
    for axis, title in zip(axes, ['Development mean score at 120s', 'Development mean survival (200s cap)']):
        axis.set_title(title)
        axis.set_xlabel('New-action lineage PPO steps (branches may overlap)')
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle('Historical pre-fix training (evolution requests ignored)' if args.corrected
                   else 'Same 8 development seeds / 200s cap / greedy inference')
    figure.savefig(ROOT / f'{prefix}training_progress.png', dpi=160)
    plt.close(figure)
    model_card = json.loads((ROOT / 'selected_model.json').read_text(encoding='utf-8'))
    model_card.update(status='evaluated_corrected_decoder' if args.corrected else 'evaluated',
                      selection_record=f'{prefix}candidate_selection.json',
                      independent_evaluation=f'{prefix}heldout_candidate.json',
                      comparison=f'{prefix}heldout_comparison.json', heldout_summary=comparison['candidate'],
                      recommended_over_previous_default=comparison['all_guards_passed'],
                      heldout_guards=comparison['promotion_guards'])
    process_records = [json.loads(line) for line in (ROOT / 'process_monitor.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    maximum_python = max(record['owned_python_count'] for record in process_records)
    model_card.update(process_limit=8, maximum_observed_training_evaluation_python=maximum_python,
                      graphify_initial_process_workers=24, overall_process_peak_measured=False,
                      process_limit_respected_entire_study=False,
                      process_limit_incident_records=['process_limit_incident.json', 'graphify_process_limit_incident.json'])
    if args.corrected:
        model_card.update(initial_results_obsolete=True, initial_results_archive='initial_decoder_results',
                          training_decoder_had_evolution_bug=True, additional_training_after_fix=0,
                          deployment_decoder_sha256=protocol['decoder_sha256'], evolution_integration_verified=True)
    model_card['verification_ppo_steps'] = sum(
        json.loads(path.read_text(encoding='utf-8'))['smoke_ppo_steps']
        for path in [ROOT / 'checks/pre_fix_results.json', ROOT / 'checks/results.json']
        if path.exists()
    )
    write_json(ROOT / 'selected_model.json', model_card)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(ROOT / f'tensorboard/{prefix}final_comparison')) as writer:
        for side in ['baseline', 'candidate']:
            for metric, statistics in comparison[side].items():
                for statistic, value in statistics.items():
                    writer.add_scalar(f'heldout/{side}/{metric}_{statistic}', value, 2000000)
        writer.add_text('selection/checkpoint', json.dumps(selection['selected'], ensure_ascii=False), 2000000)
        writer.add_text('selection/heldout_guards', json.dumps(comparison['promotion_guards']), 2000000)
        pixels = plt.imread(ROOT / f'{prefix}heldout_comparison.png')[:, :, :3]
        writer.add_image('heldout/paired_comparison', pixels.transpose(2, 0, 1), 2000000)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
