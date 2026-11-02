"""실험 원본에서 갤러리 HTML을 생성한다."""
import json
from .storage import ROOT, atomic_replace, now, read_jsonl, sha256


def build_report():
    experiments = []
    for directory in sorted((ROOT / 'experiments').iterdir()):
        manifest_path = directory / 'manifest.json'
        if not directory.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        state = json.loads((directory / 'state.json').read_text(encoding='utf-8'))
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        checks_path = directory / 'checks.json'
        confirmation_path = directory / 'confirmation.json'
        experiments.append(dict(manifest=manifest, state=state, config=config,
            evaluations=read_jsonl(directory / 'logs/evaluations.jsonl'),
            training=[dict(time=row['time'], timesteps=row['timesteps'])
                      for row in read_jsonl(directory / 'logs/training.jsonl')],
            episodes=read_jsonl(directory / 'logs/episodes.jsonl'),
            events=read_jsonl(directory / 'logs/events.jsonl'),
            checks=json.loads(checks_path.read_text(encoding='utf-8')) if checks_path.exists() else None,
            confirmation=json.loads(confirmation_path.read_text(encoding='utf-8')) if confirmation_path.exists() else None,
            root='../experiments/' + directory.name + '/',
            evidence_hashes={name:sha256(directory / name) for name in ['manifest.json','config.json','state.json']}))
    for experiment in experiments:
        episodes = experiment['episodes'] + [episode for record in experiment['evaluations'] for episode in record['episodes']]
        if experiment['confirmation']:
            episodes += experiment['confirmation']['parent']['episodes'] + experiment['confirmation']['selected']['episodes']
        for episode in episodes:
            episode.pop('telemetry', None)
    payload = json.dumps(dict(generated_at=now(), experiments=experiments), ensure_ascii=False, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (ROOT / 'report_template.html').read_text(encoding='utf-8')
    output = ROOT / 'reports/index.html'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(template.replace('__EXPERIMENT_DATA__', payload), encoding='utf-8')
    atomic_replace(temporary, output)
    return output


if __name__ == '__main__':
    print(build_report())
