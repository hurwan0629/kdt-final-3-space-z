"""One entry point for a visible job, headless job, resume, or independent viewer."""

# 실행 입구: 인자 해석 → 실행 폴더 선택 → 학습 실행 또는 대시보드 열기.

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

# 자식 프로세스가 PyTorch·NumPy를 불러오기 전에 병렬 연산 스레드 수를 제한한다.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from .runtime import atomic_json, read_json
from .wooki01 import parser, settings_from_args, run_training


def main():
    # 1) 실행 대상을 고른다. 2) 학습을 시작한다. 3) 화면 종료 시 학습 중지를 요청한다.
    p = parser(require_run=False)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()

    # 1) 지정 폴더·마지막 실행·새 실행 중 사용할 작업 폴더를 고른다.
    root = Path(__file__).resolve().parents[4]
    runs = root / 'runs'
    if args.run_dir:
        directory = Path(args.run_dir).resolve()
    elif args.resume or args.watch:
        last = read_json(runs / 'last_run.json')
        if not last:
            raise ValueError('아직 저장된 학습이 없습니다. 새 학습부터 실행하세요.')
        directory = Path(last['directory'])
    else:
        directory = runs / f'{args.algorithm}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    args.run_dir = str(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # 2) 보기 전용이면 학습 프로세스를 만들지 않고 대시보드만 연다.
    if args.watch:
        from .dashboard import run_dashboard

        run_dashboard(directory, stop_on_close=False)
        return
    cfg = settings_from_args(args)
    transferred = False

    # 3) 기존 전투 모델을 이식하는 경우 새 성장 설정과 초기 체크포인트를 준비한다.
    if args.init_combat:
        if args.resume:
            raise ValueError('Use --init-combat or --resume, not both')
        from .growth import migrate_combat

        cfg = migrate_combat(args.init_combat, directory)
        cfg.total_steps = args.steps
        cfg.workers = args.workers
        atomic_json(directory / 'config.json', cfg.to_dict())
        transferred = True
    import psutil

    existing = read_json(directory / 'status.json', {})
    if existing.get('pid') and existing.get('status') not in (
        'completed',
        'stopped',
        'failed',
    ):
        if psutil.pid_exists(existing['pid']):
            raise RuntimeError(
                '이 실행 폴더에서 이미 학습 중입니다. --watch로 화면만 여세요.'
            )
    atomic_json(runs / 'last_run.json', {'directory': str(directory)})
    atomic_json(directory / 'control.json', {'command': 'run'})

    # 4) 화면 없이 현재 프로세스에서 실행하거나, 학습을 별도 프로세스로 시작한다.
    if args.headless:
        run_training(directory, cfg, args.resume or transferred)
        return
    command = [
        sys.executable,
        '-u',
        '-m',
        'space_z.rl.training_wooki.wooki01',
        '--run-dir',
        str(directory),
        '--algorithm',
        cfg.algorithm,
        '--steps',
        str(cfg.total_steps),
        '--workers',
        str(cfg.workers),
        '--seed',
        str(cfg.seed),
    ]
    if args.resume or transferred:
        command.append('--resume')
    elif cfg.full_growth:
        command.append('--full-growth')
    if args.no_curriculum and not transferred:
        command.append('--no-curriculum')
    if args.smoke:
        command.append('--smoke')

    # 5) 학습 출력을 로그에 남기고 화면 종료 시 중지·저장 완료를 기다린다.
    with (directory / 'trainer.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        atomic_json(
            directory / 'status.json',
            dict(
                existing if args.resume else {},
                status='starting',
                pid=process.pid,
                algorithm=cfg.algorithm,
                target=cfg.total_steps,
                settings=cfg.to_dict(),
            ),
        )
        try:
            from .dashboard import run_dashboard

            run_dashboard(directory)
        finally:
            atomic_json(directory / 'control.json', {'command': 'stop'})
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                # 창 종료 후에도 학습이 남아 있으면 자식 프로세스까지 종료하고 상태를 기록한다.
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                state = read_json(directory / 'status.json', {})
                state.update(
                    status='failed',
                    error='종료 요청이 지연되어 중단했습니다. 마지막 체크포인트로 재개하세요.',
                )
                atomic_json(directory / 'status.json', state)


if __name__ == '__main__':
    main()
