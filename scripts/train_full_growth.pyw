"""Double-click entry point for full-growth PPO, including resume and transfer."""
import os
from pathlib import Path
import subprocess

folder=Path(__file__).resolve().parent
if (folder/'pyproject.toml').is_file():
    project=folder
elif (folder.parent/'pyproject.toml').is_file():
    project=folder.parent
else:
    project=folder/'kdt-final-3-game-subject'
python=project.parent/'.venv-rl'/'Scripts'/'pythonw.exe'
try:
    if not python.is_file():
        raise FileNotFoundError('먼저 전용 학습 환경 .venv-rl을 설치해 주세요.')
    directory=project/'runs'/'ppo-full-growth'
    source=project/'runs'/'ppo-score100-recovery'
    args=['--run-dir',str(directory),'--workers','4','--steps','2000000']
    if (directory/'latest.json').is_file():
        args+=['--resume']
    elif (source/'best.json').is_file():
        args+=['--init-combat',str(source)]
    else:
        args+=['--full-growth']
    env=os.environ.copy();env['PYTHONPATH']=str(project/'src')
    env['SDL_VIDEODRIVER']='windows'
    env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='1'
    env['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    env['MPLCONFIGDIR']=str(project/'runs'/'.matplotlib')
    with (project/'training-full-growth-launch.log').open('a',encoding='utf-8') as log:
        subprocess.Popen([str(python),'-u','-m','space_z.rl.training_wooki.launch',*args],
                         cwd=project,env=env,stdout=log,stderr=log,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
except Exception as error:
    from tkinter import Tk,messagebox
    root=Tk();root.withdraw()
    messagebox.showerror('Space Z 전체 성장 강화학습',str(error))
    root.destroy()
