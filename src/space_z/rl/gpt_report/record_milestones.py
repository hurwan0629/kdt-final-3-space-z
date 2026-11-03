"""고정 체크포인트를 기본 게임에서 재생하고 실험별 비교 영상을 기록한다."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
RL = ROOT.parent
VIDEOS = ROOT / 'videos'
MANIFEST = VIDEOS / 'manifest.json'
ATLAS = ROOT / 'experiment_atlas.html'
EXPERIMENTS = ('EXP001', 'EXP023', 'EXP018', 'EXP036', 'EXP045')
SEEDS = (505,)
RESULT_PREFIX = 'VIDEO_RESULT '


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save_manifest(manifest):
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(MANIFEST)


def publish(manifest):
    """촬영 메타데이터만 보고서에 반영하고 기존 실험 기록은 보존한다."""
    html = ATLAS.read_text(encoding='utf-8')
    match = re.search(r'(<script type="application/json" id="report-data">)(.*?)(</script>)', html, re.S)
    data = json.loads(match[2])
    fields = ('experiment', 'file', 'status', 'seed', 'checkpoint_step', 'frame_skip', 'selection_reason',
              'source_checkpoint', 'checkpoint_sha256', 'score', 'kills', 'survival_seconds',
              'recorded_at', 'video_bytes', 'video_sha256', 'edited', 'diagram_preview',
              'total_training_steps', 'training_step_accounting', 'candidate_before_extra_3x')
    data['videoReplays'] = dict(protocol=manifest['protocol'],
                               clips=[{key: clip[key] for key in fields if key in clip} for clip in manifest['clips']])
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    ATLAS.write_text(html[:match.start(2)] + payload + html[match.end(2):], encoding='utf-8')


def prepare():
    """촬영 전에 모델 바이트와 시드·조건·선택 이유를 고정한다."""
    if MANIFEST.exists():
        manifest = read_json(MANIFEST)
        if [(clip['experiment'], clip['seed']) for clip in manifest['clips']] != [(exp, seed) for exp in EXPERIMENTS for seed in SEEDS]:
            raise ValueError('Existing recording plan differs; preserve or reconcile it before recording')
        return manifest
    html = ATLAS.read_text(encoding='utf-8')
    data = json.loads(re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S)[1])
    records = {record['id']: record for record in data['records']}
    (VIDEOS / '_checkpoints').mkdir(parents=True, exist_ok=True)
    clips = []
    for experiment_id in EXPERIMENTS:
        record = records[experiment_id]
        folder = RL / record['folder']
        source = RL / record['network']['checkpoint']
        reason = '마지막 체크포인트: 실행별 선택 모델 기록 없음'
        best = folder / 'checkpoints/best.zip'
        if best.exists() and experiment_id != 'EXP023':
            source, reason = best, '기존 개발 평가에서 선택한 체크포인트'
        if experiment_id == 'EXP023':
            reason = '객체 인코더 도입 전 원본 벡터 MLP 실험의 마지막 체크포인트'
        if experiment_id == 'EXP036':
            reason = '당시 공개한 행동 주기 2프레임 후보, 추가 학습 0스텝'
        blob = source.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        checkpoint = VIDEOS / '_checkpoints' / f'{experiment_id}_{digest[:12]}.zip'
        if checkpoint.exists() and sha256(checkpoint) != digest:
            raise ValueError(f'Existing frozen checkpoint differs: {checkpoint}')
        checkpoint.write_bytes(blob)
        with zipfile.ZipFile(checkpoint) as archive:
            model_data = json.loads(archive.read('data'))
        frame_skip = model_data.get('decision_frame_skip') or record['trainEnv'].get('frame_skip') or 4
        for seed in SEEDS:
            number = len(clips) + 1
            clips.append(dict(number=number, experiment=experiment_id, title=record['title'], seed=seed,
                              file=f'원본/{number:03d}_{experiment_id}_seed{seed}.mp4', status='pending',
                              checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                              source_checkpoint=source.relative_to(RL).as_posix(), checkpoint_sha256=digest,
                              checkpoint_step=model_data['num_timesteps'], selection_reason=reason,
                              frame_skip=frame_skip, historical_network_sources=record['network']['sources']))
    manifest = dict(created=datetime.now(timezone.utc).isoformat(),
                    protocol=dict(environment='current default GameConfig; no curriculum', seeds=SEEDS,
                                  deterministic=True, max_seconds=400, video_fps=30,
                                  game_viewport=[1280, 800], encoded_size=[1280, 848], camera_zoom=0.35,
                                  timing='game frames / game fps; held frames duplicated for frame_skip=4',
                                  decoder='current real-game decoders, including corrected evolution indexing',
                                  caveats=['Same seed fixes the initial state; actions can change later random events.',
                                           'These retrospective replays are separate from historical evaluation results.',
                                           'Current policy implementation is used; historical source references are retained.',
                                           '30 fps samples decision boundaries, not all 60 simulation frames.']), clips=clips)
    save_manifest(manifest)
    return manifest


def record_clip(clip, protocol, ffmpeg, benchmark=False):
    """기존 환경 인터페이스로만 게임을 진행하고 실제 렌더러의 화면을 인코딩한다."""
    os.environ.update(SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1')
    sys.path.insert(0, str(RL.parents[1]))
    import gymnasium as gym
    import numpy as np
    import pygame
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import check_for_correct_spaces
    from space_z.config import EnvConfig, GameConfig, RewardConfig
    from space_z.rl import ContinuousActionDecoder, DiscreteActionDecoder, RewardCalculator, TankEnv, VectorObservation
    from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder
    from space_z.rl.gpt_training.environment import FactorizedActions
    from space_z.rendering.renderer import Renderer

    started = time.perf_counter()
    checkpoint = ROOT / clip['checkpoint']
    if sha256(checkpoint) != clip['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    torch.set_num_threads(1)
    torch.manual_seed(clip['seed'])
    model = PPO.load(checkpoint, device='cpu')
    if isinstance(model.action_space, gym.spaces.Discrete):
        decoder = DiscreteActionDecoder(enable_stat_upgrades=True, enable_evolution=True)
    elif isinstance(model.action_space, gym.spaces.MultiDiscrete):
        decoder = FactorizedActions(enable_growth=True)
    elif np.all(model.action_space.low == -1):
        decoder = ContinuousActionDecoder(enable_stat_upgrades=True, enable_evolution=True)
    else:
        decoder = RotationContinuousActionDecoder()
    seconds = 10 if benchmark else protocol['max_seconds']
    config = GameConfig(camera_zoom=protocol['camera_zoom'], window_width=1280, window_height=800)
    environment = TankEnv(config, VectorObservation(), decoder, RewardCalculator(RewardConfig()),
                          EnvConfig(frame_skip=clip['frame_skip'], max_steps=math.ceil(seconds * config.fps / clip['frame_skip'])))
    check_for_correct_spaces(environment, model.observation_space, model.action_space)
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    pygame.font.init()
    renderer = Renderer(config, visible=False)
    canvas = pygame.Surface(tuple(protocol['encoded_size']))
    font = pygame.font.SysFont('consolas', 17)
    filename = 'preview.mp4' if benchmark else clip['file']
    destination = VIDEOS / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.mp4')
    encoder_log = VIDEOS / f'{destination.stem}.encoder.log'
    # shortcut: 현재 비교 묶음은 무음이다. 게임에 음향이 추가되면 별도 오디오 트랙을 연결한다.
    command = [str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x848', '-r', str(protocol['video_fps']), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-threads', '1',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    frames_written = 0
    trace = hashlib.sha256()
    event_totals = {}
    try:
        observation, info = environment.reset(seed=clip['seed'])
        initial_state = dict(map=info['map'], player=asdict(environment.game.player),
                             observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest())
        with encoder_log.open('wb') as error_log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
            try:
                while True:
                    action = model.predict(observation, deterministic=True)[0]
                    if isinstance(model.action_space, gym.spaces.Discrete):
                        action = action.item()
                    observation, reward, terminated, truncated, info = environment.step(action)
                    trace.update(np.asarray(action).tobytes())
                    trace.update(observation.tobytes())
                    for key, value in info['events'].items():
                        if isinstance(value, (int, float)):
                            event_totals[key] = event_totals.get(key, 0) + value
                    renderer.draw(environment.game, show_controls=False)
                    canvas.fill((12, 16, 28))
                    canvas.blit(renderer.screen, (0, 48))
                    header = (f"{clip['experiment']} | total training steps {clip.get('total_training_steps', clip['checkpoint_step']):,} | seed {clip['seed']} | "
                              f"decision {clip['frame_skip']} frames | greedy | {info['frames']/config.fps:.1f}s")
                    footer = (f"SCORE {info['score']}   KILLS {info['kills']}   HP {environment.game.player.hp:.1f}"
                              f"   LEVEL {info['level']}   {info['job']}")
                    canvas.blit(font.render(header, True, (228, 233, 247)), (16, 4))
                    canvas.blit(font.render(footer, True, (151, 216, 196)), (16, 25))
                    required_frames = math.ceil(info['frames'] * protocol['video_fps'] / config.fps)
                    frame_bytes = pygame.image.tobytes(canvas, 'RGB')
                    for _ in range(required_frames - frames_written):
                        encoder.stdin.write(frame_bytes)
                    frames_written = required_frames
                    if terminated or truncated:
                        break
            finally:
                encoder.stdin.close()
                exit_code = encoder.wait(timeout=90)
        if exit_code:
            raise RuntimeError(encoder_log.read_text(encoding='utf-8', errors='replace'))
        partial.replace(destination)
        encoder_log.unlink()
        if benchmark:
            pygame.image.save(canvas, str(VIDEOS / 'preview.png'))
        modules = {Path(module.__file__).resolve() for name, module in sys.modules.copy().items()
                   if name.startswith('space_z') and getattr(module, '__file__', None)}
        sources = {path.relative_to(RL.parents[1]).as_posix(): sha256(path) for path in modules if path.is_relative_to(RL.parents[1])}
        return dict(status='completed', recorded_at=datetime.now(timezone.utc).isoformat(),
                    game_config=asdict(config), source_hashes=sources,
                    versions=dict(python=sys.version, torch=torch.__version__, pygame=pygame.version.ver,
                                  gymnasium=gym.__version__),
                    initial_state=initial_state, action_observation_sha256=trace.hexdigest(),
                    score=info['score'], kills=info['kills'], survival_seconds=info['frames']/config.fps,
                    game_frames=info['frames'], environment_steps=info['steps'], health=environment.game.player.hp,
                    level=info['level'], ship=info['job'], terminated=terminated, truncated=truncated,
                    event_totals=event_totals, video_frames=frames_written,
                    video_duration=frames_written/protocol['video_fps'], wall_seconds=time.perf_counter()-started,
                    video_bytes=destination.stat().st_size, video_sha256=sha256(destination))
    finally:
        environment.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--clip', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.ffmpeg.is_file():
        parser.error('Provide an existing ffmpeg executable')
    manifest = prepare()
    if args.clip is not None or args.benchmark:
        number = args.clip or 1
        if not 1 <= number <= len(manifest['clips']):
            parser.error('Invalid clip number')
        result = record_clip(manifest['clips'][number-1], manifest['protocol'], args.ffmpeg, args.benchmark)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
        return
    pending = [clip for clip in manifest['clips'] if clip['status'] != 'completed']
    for clip in manifest['clips']:
        if clip['status'] == 'completed' and (not (VIDEOS / clip['file']).is_file()
                                             or sha256(VIDEOS / clip['file']) != clip['video_sha256']):
            raise ValueError(f"Completed video missing or changed: {clip['file']}")
    def run(clip):
        command = [sys.executable, '-B', str(Path(__file__)), '--ffmpeg', str(args.ffmpeg.resolve()), '--clip', str(clip['number'])]
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if process.returncode:
            raise RuntimeError(process.stderr[-6000:] + process.stdout[-2000:])
        return json.loads(next(line[len(RESULT_PREFIX):] for line in process.stdout.splitlines() if line.startswith(RESULT_PREFIX)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run, clip): clip for clip in pending}
        for task in as_completed(tasks):
            clip = tasks[task]
            try:
                clip.update(task.result())
                print(f"{clip['file']} score={clip['score']} seconds={clip['survival_seconds']:.2f} wall={clip['wall_seconds']:.1f}", flush=True)
            except Exception as error:
                clip.update(status='failed', error=str(error))
                print(f"FAILED {clip['file']}: {error}", flush=True)
            save_manifest(manifest)
            publish(manifest)
    if any(clip['status'] != 'completed' for clip in manifest['clips']):
        raise SystemExit('Incomplete recordings; inspect videos/manifest.json and rerun to resume')


if __name__ == '__main__':
    main()
