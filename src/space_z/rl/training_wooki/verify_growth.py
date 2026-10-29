"""Reproducible growth verification, separating fixtures from natural play."""

# 성장 검증 입구: 기존 기능 테스트를 먼저 실행하고, 선택적으로 자연 플레이 성능을 평가한다.

import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

from space_z.config.progression_definitions import SHIPS
from .runtime import read_json, atomic_json, load_checkpoint, evaluate
from .settings import TrainSettings


def verify(
    run_dir=None,
    episodes=3,
    steps=1500,
    seed=1200001,
    output=None,
    mechanics_only=False,
    require_all_ships=False,
):
    # 기능 테스트 결과와 선택적 자연 플레이 결과를 구분해 검증 보고서와 종료 코드를 반환한다.
    root = Path(__file__).resolve().parents[4]
    directory = (
        Path(run_dir).resolve() if run_dir else root / 'runs' / 'growth-verification'
    )
    output = Path(output) if output else directory / 'verification.json'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    env['SDL_VIDEODRIVER'] = 'dummy'
    env['SDL_AUDIODRIVER'] = 'dummy'
    env['PYTHONIOENCODING'] = 'utf-8'

    # 1) 별도 프로세스에서 기존 성장 기능 테스트를 실행하고 성공 여부·로그를 수집한다.
    test = subprocess.run(
        [
            sys.executable,
            '-m',
            'unittest',
            'discover',
            '-s',
            'tests',
            '-p',
            'test_full_growth.py',
            '-v',
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    report = {
        'mechanics': {
            'passed': test.returncode == 0,
            'exit_code': test.returncode,
            'log': test.stdout + test.stderr,
        },
        'fixture_xp_used_in_performance': False,
    }
    if test.returncode:
        atomic_json(output, report)
        return report, 1

    # 2) 기능 검사만 요청한 경우가 아니면 저장된 성장 정책으로 자연 플레이를 평가한다.
    if not mechanics_only:
        if not run_dir:
            raise ValueError('Natural-play verification requires --run-dir')
        entry = read_json(directory / 'best.json') or read_json(
            directory / 'latest.json'
        )
        if not entry or not entry['settings'].get('full_growth'):
            raise ValueError('Select a full-growth checkpoint, not a combat-only run')
        cfg = replace(
            TrainSettings(**entry['settings']),
            workers=1,
            eval_episodes=episodes,
            eval_steps=steps,
        )
        model, training_env = load_checkpoint(
            directory, entry, cfg=cfg, evaluation=True
        )
        try:
            result = evaluate(model, training_env, cfg, seed_start=seed)
        finally:
            training_env.close()

        # 3) 실제 등장한 기체와 성장 기록을 집계한다. 미관찰 기체를 학습 완료로 간주하지 않는다.
        seen = set(result['ships_observed'])
        missing = sorted(set(SHIPS) - seen)
        report.update(
            checkpoint=entry['directory'],
            settings=cfg.to_dict(),
            performance=result,
            coverage={
                'max_natural_level': result['max_level'],
                'natural_stat_upgrades': sum(
                    e['stat_upgrades'] for e in result['episodes']
                ),
                'natural_evolutions': sum(e['evolutions'] for e in result['episodes']),
                'natural_masteries': sum(e['masteries'] for e in result['episodes']),
                'ships_seen': sorted(seen),
                'ships_not_seen': missing,
                'all_ships_observed': not missing,
            },
            evaluation_horizon_seconds=steps * 4 / 60,
            note='Mechanics fixtures prove integration; unseen ships are not claimed as mastered.',
        )
        code = 2 if require_all_ships and missing else 0
    else:
        code = 0

    # 4) 검증 보고서를 저장하고 호출자에게 결과와 종료 코드를 반환한다.
    atomic_json(output, report)
    print(f'Mechanics passed. Report: {output}', flush=True)
    if 'performance' in report:
        r = report['performance']
        print(
            f"Natural play: score={r['score']:.1f}, max level={r['max_level']}, "
            f"stats/episode={r['stat_upgrades']:.1f}, evolutions/episode={r['evolutions']:.1f}",
            flush=True,
        )
        print(
            f"Unobserved ship types: {len(report['coverage']['ships_not_seen'])}",
            flush=True,
        )
    return report, code


def main():
    # 검증 범위·평가 길이·seed를 해석하고, 검증 결과 코드를 프로세스 종료 코드로 사용한다.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--seed', type=int, default=1200001)
    p.add_argument('--output')
    p.add_argument('--mechanics-only', action='store_true')
    p.add_argument('--require-all-ships', action='store_true')
    a = p.parse_args()
    if a.episodes < 1 or a.steps < 1:
        p.error('episodes and steps must be positive')
    _, code = verify(
        a.run_dir,
        a.episodes,
        a.steps,
        a.seed,
        a.output,
        a.mechanics_only,
        a.require_all_ships,
    )
    raise SystemExit(code)


if __name__ == '__main__':
    main()
