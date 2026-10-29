"""One entry point for a visible job, headless job, resume, or independent viewer."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

# Limit each worker before importing PyTorch/NumPy in child processes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from .runtime import atomic_json, read_json
from .train import parser, settings_from_args, run_training


def main():
    p = parser(require_run=False)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    runs = root/'runs'
    if args.run_dir:
        directory = Path(args.run_dir).resolve()
    elif args.resume or args.watch:
        last = read_json(runs/'last_run.json')
        if not last:
            raise ValueError('아직 저장된 학습이 없습니다. 새 학습부터 실행하세요.')
        directory = Path(last['directory'])
    else:
        directory = runs/f'{args.algorithm}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    args.run_dir = str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if args.watch:
        from .dashboard import run_dashboard
        run_dashboard(directory, stop_on_close=False)
        return
    cfg = settings_from_args(args)
    transferred=False
    if args.init_combat:
        if args.resume:
            raise ValueError('Use --init-combat or --resume, not both')
        from .growth import migrate_combat
        cfg=migrate_combat(args.init_combat,directory)
        cfg.total_steps=args.steps;cfg.workers=args.workers
        atomic_json(directory/'config.json',cfg.to_dict())
        transferred=True
    import psutil
    existing = read_json(directory/'status.json', {})
    if existing.get('pid') and existing.get('status') not in ('completed', 'stopped', 'failed'):
        if psutil.pid_exists(existing['pid']):
            raise RuntimeError('이 실행 폴더에서 이미 학습 중입니다. --watch로 화면만 여세요.')
    atomic_json(runs/'last_run.json', {'directory': str(directory)})
    atomic_json(directory/'control.json', {'command': 'run'})
    if args.headless:
        run_training(directory, cfg, args.resume or transferred)
        return
    command = [sys.executable, '-u', '-m', 'space_z.rl.training_wooki.train', '--run-dir', str(directory),
               '--algorithm', cfg.algorithm, '--steps', str(cfg.total_steps), '--workers', str(cfg.workers),
               '--seed', str(cfg.seed)]
    if args.resume or transferred:
        command.append('--resume')
    elif cfg.full_growth:
        command.append('--full-growth')
    if args.no_curriculum and not transferred:
        command.append('--no-curriculum')
    if args.smoke:
        command.append('--smoke')
    with (directory/'trainer.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        atomic_json(directory/'status.json', dict(existing if args.resume else {},
                    status='starting', pid=process.pid, algorithm=cfg.algorithm,
                    target=cfg.total_steps, settings=cfg.to_dict()))
        try:
            from .dashboard import run_dashboard
            run_dashboard(directory)
        finally:
            atomic_json(directory/'control.json', {'command': 'stop'})
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                # Do not silently leave an unmonitored training job after window closure.
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                state = read_json(directory/'status.json', {})
                state.update(status='failed', error='종료 요청이 지연되어 중단했습니다. 마지막 체크포인트로 재개하세요.')
                atomic_json(directory/'status.json', state)


if __name__ == '__main__':
    main()
