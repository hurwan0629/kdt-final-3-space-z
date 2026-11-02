"""고정 정책의 벽 대응을 계측하고 원자료와 독립 실행 HTML을 만든다. PPO 갱신은 없다."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from space_z.game.actions import MOVE_VECTORS
from space_z.game.entities import Bullet
from space_z.game.terrain import BREAKABLE
from space_z.rendering.renderer import Renderer
from space_z.rl.gpt_training_rotation_continuous.environment import RotationTrainingEnvironment

DECISIONS_PER_SECOND = 30
SOURCE_FILES = [
    ROOT / 'policy.py', ROOT / 'environment.py',
    ROOT.parent / 'gpt_training/networks.py', ROOT.parent / 'gpt_training/environment.py',
    ROOT.parent / 'environment.py', ROOT.parent / 'actions/rotation_continuous_action_decoder.py',
    ROOT.parent / 'observations/vector_observation.py', ROOT.parent / 'observations/progression_observation.py',
    ROOT.parents[1] / 'game/terrain.py', ROOT.parents[1] / 'game/game_core.py',
    ROOT.parents[1] / 'game/combat.py', ROOT.parents[1] / 'game/player_progression.py',
    ROOT.parents[1] / 'rendering/renderer.py',
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def expected_wall_grid(game) -> np.ndarray:
    """실제 solid 배열과 월드 경계에서 입력과 같은 범위의 참조 격자를 얻는다."""
    terrain, player = game.terrain, game.player
    columns = np.arange(45) + math.floor(player.x / terrain.tile) - 22
    rows = np.arange(29) + math.floor(player.y / terrain.tile) - 14
    inside = (rows[:, None] >= 0) & (rows[:, None] < terrain.rows)
    inside = inside & (columns[None, :] >= 0) & (columns[None, :] < terrain.cols)
    cells = np.asarray(terrain.solid, dtype=bool)
    sampled = cells[np.clip(rows, 0, terrain.rows - 1)[:, None], np.clip(columns, 0, terrain.cols - 1)]
    return np.where(inside, sampled, True)


def target_visibility(model, observation, game) -> dict:
    """실제로 조준망에 전달되는 표적을 월드 좌표로 복원해 정확한 시야선과 비교한다."""
    with torch.no_grad():
        features = model.policy.aim_features(torch.as_tensor(observation).unsqueeze(0))[0, 160:169].numpy()
    if features[0] < 0.5:
        return dict(selected=False, blocked=False)
    sine, cosine = map(float, observation[3:5])
    forward, sideways = map(float, features[1:3])
    x = game.player.x + (forward * cosine - sideways * sine) * 1080
    y = game.player.y + (forward * sine + sideways * cosine) * 1080
    clear = game.terrain.line_of_sight(game.player.x, game.player.y, x, y)
    return dict(selected=True, blocked=not clear, world_position=[x, y])


class BulletObserver:
    """원래 전투 처리를 한 번만 호출하고 실제 벽에 소멸한 플레이어 탄환을 센다."""

    def __init__(self, game):
        self.game = game
        self.combat = game.combat
        self.original_move = self.combat._move_bullets
        self.original_hit = self.combat._bullet_hit
        self.removed_by_entity = set()
        self.wall_events = []
        self.excluded_breakable = 0
        self.excluded_boundary = 0
        self.combat._move_bullets = self.move_bullets
        self.combat._bullet_hit = self.bullet_hit

    def bullet_hit(self, bullet):
        removed = self.original_hit(bullet)
        if removed and bullet.team == 'player':
            self.removed_by_entity.add(id(bullet))
        return removed

    def move_bullets(self):
        # 개체에 먼저 명중해 사라진 탄환을 벽 충돌로 중복 분류하지 않는다.
        self.removed_by_entity.clear()
        candidates = []
        terrain = self.game.terrain
        for bullet in self.combat.bullets:
            if bullet.team != 'player':
                continue
            distance = math.hypot(bullet.velocity_x, bullet.velocity_y)
            if not distance:
                continue
            reach, tile = terrain.sweep_circle(bullet.x, bullet.y, bullet.velocity_x / distance,
                                               bullet.velocity_y / distance, distance, bullet.radius)
            if tile is not None:
                candidates.append((bullet, tile, [bullet.x, bullet.y], reach))
        self.original_move()
        survivors = {id(bullet) for bullet in self.combat.bullets}
        for bullet, (column, row), origin, reach in candidates:
            if id(bullet) in survivors or id(bullet) in self.removed_by_entity:
                continue
            if not (0 <= column < terrain.cols and 0 <= row < terrain.rows):
                self.excluded_boundary += 1
            elif terrain.tiles[row][column] == BREAKABLE:
                self.excluded_breakable += 1
            else:
                self.wall_events.append(dict(game_frame=self.game.frame, wall_tile=[column, row],
                                             origin=origin, impact=[bullet.x, bullet.y], reach=reach))

    def close(self):
        self.combat._move_bullets = self.original_move
        self.combat._bullet_hit = self.original_hit


def update_replay_digest(digest, observation, action, info, game):
    digest.update(np.asarray(observation, dtype='<f4').tobytes())
    digest.update(np.asarray(action, dtype='<f4').tobytes())
    digest.update(np.asarray([game.player.x, game.player.y, game.player.hp], dtype='<f8').tobytes())
    digest.update(encode_json({key: info[key] for key in ['score', 'kills', 'frames', 'job', 'events']}).encode())


def scan_episode(model, seed, max_seconds, write_record):
    environment = RotationTrainingEnvironment(max_steps=max_seconds * DECISIONS_PER_SECOND, reward_profile='legacy')
    observation, info = environment.reset(seed=seed)
    game = environment.unwrapped.game
    assert not game.config.terrain.breakable_enabled
    observer = BulletObserver(game)
    digest = hashlib.sha256()
    total_steps = total_shots = wall_shots = moving_steps = stationary_steps = 0
    grid_checks = grid_mismatches = selected_target_checks = blocked_target_checks = 0
    stationary_seconds = 0.0
    bucket = dict(steps=0, shots_fired=0, permanent_wall_bullets=0, wall_stationary_steps=0,
                  wall_stationary_seconds=0.0, moving_steps=0, grid_checks=0,
                  grid_mismatch_samples=0, selected_target_checks=0, blocked_target_checks=0)
    segments, samples, shot_windows, visibility_windows = [], [], [], []
    current_segment = None
    terrain_hash = hashlib.sha256(np.asarray(game.terrain.solid, dtype=np.uint8).tobytes()).hexdigest()
    try:
        while True:
            frame = game.frame
            previous_position = [game.player.x, game.player.y]
            action, _ = model.predict(observation, deterministic=True)
            move_x, move_y = MOVE_VECTORS[int(action[0])]
            move_requested = bool(move_x or move_y)
            probe_distance = max(1.0, game.movement_speed(game.player) * 2)
            reach, hit_tile = game.terrain.sweep_circle(*previous_position, move_x, move_y,
                                                       probe_distance, game.player.radius) if move_requested else (None, None)
            permanent_wall = hit_tile is not None and 0 <= hit_tile[0] < game.terrain.cols and 0 <= hit_tile[1] < game.terrain.rows
            permanent_wall = permanent_wall and game.terrain.tiles[hit_tile[1]][hit_tile[0]] != BREAKABLE
            grid_difference = None
            if total_steps % 30 == 0:
                reference = expected_wall_grid(game)
                grid_difference = int(np.count_nonzero(observation[1224:2529].reshape(29, 45) != reference))
                grid_checks += 1
                grid_mismatches += grid_difference > 0
            visibility = None
            if total_steps % 10 == 0:
                visibility = target_visibility(model, observation, game)
                selected_target_checks += int(visibility['selected'])
                blocked_target_checks += int(visibility['blocked'])
                if visibility['blocked']:
                    visibility_windows.append(dict(seed=seed, time_seconds=frame / 60, step=total_steps,
                                                   kind='visibility', **visibility))
            wall_count_before = len(observer.wall_events)
            next_observation, _, terminated, truncated, info = environment.step(action)
            update_replay_digest(digest, next_observation, action, info, game)
            displacement = math.dist(previous_position, [game.player.x, game.player.y])
            stalled = bool(move_requested and permanent_wall and reach <= 1.0 and displacement <= 0.5)
            moving_steps += int(move_requested)
            stationary_steps += int(stalled)
            stationary_seconds += (info['frames'] - frame) / 60 if stalled else 0.0
            if stalled:
                if current_segment is None:
                    current_segment = dict(seed=seed, start_seconds=frame / 60, start_step=total_steps,
                                           end_seconds=info['frames'] / 60, kind='movement')
                current_segment['end_seconds'] = info['frames'] / 60
            elif current_segment is not None:
                segments.append(current_segment)
                current_segment = None
            new_wall_events = observer.wall_events[wall_count_before:]
            wall_shots += len(new_wall_events)
            fired = int(info['events']['player_shots_fired'])
            total_shots += fired
            bucket['steps'] += 1
            bucket['shots_fired'] += fired
            bucket['permanent_wall_bullets'] += len(new_wall_events)
            bucket['wall_stationary_steps'] += int(stalled)
            bucket['wall_stationary_seconds'] += (info['frames'] - frame) / 60 if stalled else 0.0
            bucket['moving_steps'] += int(move_requested)
            bucket['grid_checks'] += int(grid_difference is not None)
            bucket['grid_mismatch_samples'] += int(grid_difference is not None and grid_difference > 0)
            bucket['selected_target_checks'] += int(visibility is not None and visibility['selected'])
            bucket['blocked_target_checks'] += int(visibility is not None and visibility['blocked'])
            record = dict(type='step', seed=seed, step=total_steps, time_seconds=frame / 60,
                          end_seconds=info['frames'] / 60, position=previous_position,
                          next_position=[game.player.x, game.player.y], displacement_px=displacement,
                          action=action.tolist(), wall_probe_reach_px=reach, wall_probe_tile=hit_tile,
                          stationary_against_wall=stalled, grid_mismatch_cells=grid_difference,
                          selected_target_visibility=visibility, shots_fired=fired,
                          permanent_wall_bullet_count=len(new_wall_events), wall_impacts=new_wall_events,
                          score=info['score'], hp=game.player.hp, ship=game.player.job)
            if total_steps % 30 == 0 or stalled or new_wall_events or (visibility and visibility['blocked']) or terminated or truncated:
                write_record(record)
            if new_wall_events:
                shot_windows.append(dict(seed=seed, step=total_steps, time_seconds=frame / 60,
                                         count=len(new_wall_events), kind='shooting'))
            total_steps += 1
            if total_steps % 30 == 0 or terminated or truncated:
                second_summary = dict(type='second_summary', seed=seed, end_seconds=info['frames'] / 60,
                                      score=info['score'], **bucket)
                write_record(second_summary)
                samples.append(second_summary)
                bucket = {key: 0.0 if key.endswith('_seconds') else 0 for key in bucket}
            observation = next_observation
            if total_steps % 1500 == 0:
                print(f'scan seed={seed} seconds={info["frames"]/60:.1f} walls={wall_shots} stationary_steps={stationary_steps}', flush=True)
            if terminated or truncated:
                break
        if current_segment is not None:
            segments.append(current_segment)
        for segment in segments:
            segment['duration_seconds'] = segment['end_seconds'] - segment['start_seconds']
        long_segments = [segment for segment in segments if segment['duration_seconds'] >= 1.0 - 1e-8]
        assert total_shots >= wall_shots
        assert terrain_hash == hashlib.sha256(np.asarray(game.terrain.solid, dtype=np.uint8).tobytes()).hexdigest()
        summary = dict(seed=seed, steps=total_steps, seconds=info['frames'] / 60, score=info['score'],
                       kills=info['kills'], ship=game.player.job, terminated=terminated, truncated=truncated,
                       grid_checks=grid_checks, grid_mismatch_samples=grid_mismatches,
                       moving_steps=moving_steps, wall_stationary_steps=stationary_steps,
                       wall_stationary_seconds=stationary_seconds,
                       long_stationary_seconds=sum(item['duration_seconds'] for item in long_segments),
                       stationary_segments=long_segments, shots_fired=total_shots, permanent_wall_bullets=wall_shots,
                       excluded_boundary_bullets=observer.excluded_boundary, excluded_breakable_bullets=observer.excluded_breakable,
                       selected_target_checks=selected_target_checks, blocked_target_checks=blocked_target_checks,
                       trajectory_sha256=digest.hexdigest(), terrain_sha256=terrain_hash, samples=samples)
        write_record(dict(type='episode_summary', **summary))
        candidates = sorted(long_segments, key=lambda item: -item['duration_seconds'])
        candidates += visibility_windows[::max(1, len(visibility_windows) // 3)]
        candidates += sorted(shot_windows, key=lambda item: -item['count'])[:5]
        return summary, candidates
    finally:
        observer.close()
        environment.close()


def choose_clips(candidates, summaries):
    selected = []
    for kind in ['movement', 'visibility', 'shooting']:
        matching = [item for item in candidates if item['kind'] == kind]
        matching.sort(key=lambda item: -item.get('duration_seconds', item.get('count', 0)))
        for item in matching:
            timestamp = item.get('start_seconds', item.get('time_seconds'))
            if any(other['seed'] == item['seed'] and abs(other['center_seconds'] - timestamp) < 6 for other in selected):
                continue
            selected.append(dict(seed=item['seed'], center_seconds=timestamp, kind=kind,
                                 duration_seconds=item.get('duration_seconds')))
            break
    if not selected:
        selected.append(dict(seed=summaries[0]['seed'], center_seconds=5.0, kind='reference', duration_seconds=None))
    for clip in selected:
        clip['start_step'] = max(0, int((clip['center_seconds'] - 2) * DECISIONS_PER_SECOND))
        clip['end_step'] = int((clip['center_seconds'] + 2) * DECISIONS_PER_SECOND)
        clip['frames'] = []
    return selected


def capture_frame(renderer, game, observation, action, visibility) -> str:
    """실제 렌더러 화면과 정책 입력 격자를 같은 행동 전 시점에 그린다."""
    renderer.draw(game, show_controls=False)
    canvas = pygame.Surface((1370, 540))
    canvas.fill('#f2f5fa')
    canvas.blit(pygame.transform.smoothscale(renderer.screen, (960, 540)), (0, 0))
    font = pygame.font.SysFont('malgungothic,segoeui,arial', 16)
    def label(text, x, y, color='#24324b'):
        canvas.blit(font.render(text, True, color), (x, y))
    label('정책에 들어가는 벽 격자 · 45 × 29', 980, 18)
    grid = observation[1224:2529].reshape(29, 45)
    reference = expected_wall_grid(game)
    scale, origin_x, origin_y = 8, 980, 56
    for row in range(29):
        for column in range(45):
            rectangle = (origin_x + column * scale, origin_y + row * scale, scale - 1, scale - 1)
            pygame.draw.rect(canvas, '#35445f' if grid[row, column] else '#dce4ef', rectangle)
            if bool(grid[row, column]) != bool(reference[row, column]):
                pygame.draw.rect(canvas, '#d72473', rectangle, 2)
    terrain, player = game.terrain, game.player
    player_pixel = (origin_x + (22 + player.x / terrain.tile % 1) * scale,
                    origin_y + (14 + player.y / terrain.tile % 1) * scale)
    pygame.draw.circle(canvas, '#008982', player_pixel, max(3, player.radius / terrain.tile * scale), 2)
    movement = MOVE_VECTORS[int(action[0])]
    pygame.draw.line(canvas, '#008982', player_pixel,
                     (player_pixel[0] + movement[0] * 35, player_pixel[1] + movement[1] * 35), 3)
    pygame.draw.line(canvas, '#db7900', player_pixel,
                     (player_pixel[0] + math.cos(player.angle) * 55, player_pixel[1] + math.sin(player.angle) * 55), 2)
    if visibility.get('selected'):
        target = visibility['world_position']
        target_pixel = (player_pixel[0] + (target[0] - player.x) / terrain.tile * scale,
                        player_pixel[1] + (target[1] - player.y) / terrain.tile * scale)
        pygame.draw.line(canvas, '#bd3156' if visibility['blocked'] else '#5366d9', player_pixel, target_pixel, 2)
        pygame.draw.circle(canvas, '#bd3156', target_pixel, 4, 1)
    label(f'{game.frame/60:.2f}초 · {game.progression.score}점 · {player.job}', 980, 316)
    label(f'이동 번호 {int(action[0])} / 발사 요청 {int(action[2])}', 980, 348)
    label('청록: 실제 기체 위치와 이동 요청', 980, 386)
    label('주황: 포신 방향 / 선: 조준망 표적', 980, 414)
    label('진분홍 테두리: 입력·실제 지형 불일치', 980, 442)
    label('실제 이동 결과는 아래 타임라인에서 확인', 980, 482)
    buffer = io.BytesIO()
    pygame.image.save(canvas, buffer, 'frame.png')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def replay_and_capture(model, summary, clips, max_seconds):
    environment = RotationTrainingEnvironment(max_steps=max_seconds * DECISIONS_PER_SECOND, reward_profile='legacy')
    observation, info = environment.reset(seed=summary['seed'])
    game = environment.unwrapped.game
    renderer = None
    digest = hashlib.sha256()
    try:
        for step in range(summary['steps']):
            action, _ = model.predict(observation, deterministic=True)
            for clip in clips:
                if clip['seed'] != summary['seed'] or not clip['start_step'] <= step <= clip['end_step'] or step % 15:
                    continue
                if renderer is None:
                    pygame.display.init()
                    pygame.font.init()
                    game.config.camera_zoom = 0.35
                    renderer = Renderer(game.config, visible=False)
                visibility = target_visibility(model, observation, game)
                clip['frames'].append(dict(time_seconds=game.frame / 60, position=[game.player.x, game.player.y],
                                           action=action.tolist(), visibility=visibility,
                                           image=capture_frame(renderer, game, observation, action, visibility)))
            observation, _, terminated, truncated, info = environment.step(action)
            update_replay_digest(digest, observation, action, info, game)
            if terminated or truncated:
                break
        assert digest.hexdigest() == summary['trajectory_sha256'], '계측 없는 재생과 전체 관찰·행동·전투 결과가 다름'
        summary['uninstrumented_replay_matches'] = True
        print(f'replay verified seed={summary["seed"]} frames={info["frames"]}', flush=True)
    finally:
        environment.close()


def self_check(model):
    """실제 충돌 처리에서 벽·선행 개체 명중을 구분하고 관찰 정렬 검사를 확인한다."""
    environment = RotationTrainingEnvironment(max_steps=4, reward_profile='legacy')
    observation, _ = environment.reset(seed=505)
    game, terrain = environment.unwrapped.game, environment.unwrapped.game.terrain
    observer = BulletObserver(game)
    try:
        reference = expected_wall_grid(game)
        actual = observation[1224:2529].reshape(29, 45)
        assert np.array_equal(actual, reference)
        assert not np.array_equal(np.roll(actual, 1, axis=1), reference)
        pygame.display.init()
        pygame.font.init()
        game.config.camera_zoom = 0.35
        renderer = Renderer(game.config, visible=False)
        action, _ = model.predict(observation, deterministic=True)
        assert capture_frame(renderer, game, observation, action, target_visibility(model, observation, game)).startswith('data:image/png;base64,')
        enemy = game.enemies[0]
        game.enemies, game.shapes = [], []
        for row in range(1, terrain.rows - 1):
            for column in range(1, terrain.cols - 1):
                if not terrain.is_solid_tile(column, row) and terrain.is_solid_tile(column + 1, row):
                    position = terrain.center(column, row)
                    break
            else:
                continue
            break
        else:
            raise AssertionError('벽 충돌 fixture 위치를 찾지 못함')
        def bullet():
            return Bullet(*position, terrain.tile, 0.0, 3, game.player.id, 'player', 1.0, 1.0)
        game.combat.bullets = [bullet()]
        game.combat._move_bullets()
        assert len(observer.wall_events) == 1 and not game.combat.bullets
        enemy.x, enemy.y = position[0] + 8, position[1]
        enemy.radius, enemy.hp, enemy.alive = 2, 100, True
        game.enemies = [enemy]
        game.combat.bullets = [bullet()]
        game.combat._move_bullets()
        assert len(observer.wall_events) == 1 and not game.combat.bullets
        assert enemy.hp < 100
        target_visibility(model, observation, environment.unwrapped.game)
        return dict(grid_alignment=True, shifted_grid_detected=True, headless_capture=True, actual_wall_disposal=True,
                    entity_before_wall_not_double_counted=True, ppo_updates=0)
    finally:
        observer.close()
        environment.close()


def write_report(path, data):
    """검토한 집계와 동일 시점 캡처를 외부 의존성 없는 HTML에 넣는다."""
    payload = encode_json(data).replace('</', '<\\/')
    path.write_text(HTML_TEMPLATE.replace('__REPORT_DATA__', payload), encoding='utf-8')


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Space Z · 벽 대응 진단</title><style>
:root{color-scheme:light;--ink:#1e2b41;--muted:#5d6c82;--line:#dce3ec;--accent:#176b66;--warn:#aa3f38}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:var(--ink);font:16px/1.7 "Malgun Gothic","Segoe UI",sans-serif}main{max-width:1200px;margin:auto;padding:48px 28px 70px}header{border-bottom:2px solid var(--ink);padding-bottom:25px;margin-bottom:28px}.eyebrow{font:700 12px/1.5 monospace;letter-spacing:.12em;color:var(--accent)}h1{font-size:34px;line-height:1.4;letter-spacing:-.04em;margin:14px 0}h2{font-size:23px;letter-spacing:-.025em;margin:0 0 14px}h3{font-size:18px;margin:12px 0}p{margin:8px 0 14px}.muted,small{color:var(--muted)}section{background:white;border:1px solid var(--line);border-radius:12px;padding:25px;margin:24px 0}.summary{border-left:5px solid var(--accent)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:16px;background:#f7f9fb;border:1px solid var(--line);border-radius:8px}.card strong{display:block;font-size:27px;line-height:1.5}.card span{font-size:13px;color:var(--muted)}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}th,td{padding:12px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}.finding{padding:15px 0;border-bottom:1px solid var(--line)}.finding:last-child{border:0}.tag{font-size:12px;font-weight:bold;padding:3px 8px;background:#e6f4ef;color:#13615b;border-radius:4px;margin-right:8px}.tag.warn{background:#fff0e5;color:#a44918}.tag.note{background:#edf0f8;color:#4c5c88}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.row>*{margin:0}select,button{font:inherit;padding:9px 12px;border:1px solid #b9c6d8;border-radius:6px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{background:#eef4f8}button:disabled{opacity:.45;cursor:default}label{font-size:14px}.viewer{width:100%;display:block;margin:18px 0 8px;border-radius:8px;border:1px solid var(--line)}input[type=range]{width:100%;accent-color:var(--accent)}.frame-caption{font-size:14px;min-height:30px}.timeline{height:112px;width:100%;display:block}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--muted)}.dot{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}code,pre{font:13px/1.6 Consolas,monospace}pre{background:#f5f7fa;padding:15px;white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:15px}summary{cursor:pointer;font-weight:600}.bar{background:#e5ebf1;height:9px;border-radius:8px;overflow:hidden;width:150px;display:inline-block;margin-right:8px}.bar i{display:block;height:100%;background:#317f86}a{color:#176b83}ul,ol{padding-left:22px}.callout{background:#fff7e9;padding:14px;border-radius:6px}footer{font-size:13px;color:var(--muted);margin-top:30px}@media(max-width:700px){main{padding:24px 15px}h1{font-size:27px}section{padding:18px}.cards{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}.controls{display:none}details{display:block}header{margin-bottom:15px}}
</style></head><body><main>
<header><div class="eyebrow">SPACE Z / BEHAVIOR DIAGNOSTICS</div><h1 id="headline">벽 앞에서 실제로 어떤 행동을 했는가</h1><p>선택된 연속 회전 모델 · 고정 가중치 · 파괴 불가능한 벽 · 학습 업데이트 0회</p><small id="date"></small></header>
<section class="summary"><h2>이번 재생에서 확인한 결과</h2><div id="conclusions"></div><p class="muted">세 시드에서 최대 200초를 관찰한 진단입니다. 학습 전후 성능 비교나 모든 시작 상황에 대한 일반화 평가는 아닙니다.</p></section>
<div class="cards" id="cards"></div>
<section><h2>시드별 관측 결과</h2><p class="muted">탄환 비율의 분모는 실제 생성한 플레이어 탄환입니다. 발사 버튼 요청 횟수와는 다릅니다.</p><div class="table-wrap"><table><thead><tr><th>시드</th><th>관찰 시간</th><th>점수 / 기체</th><th>1초 이상 정체 구간</th><th>벽에 제거된 탄환 / 실제 발사</th><th>격자 불일치 표본</th><th>계측 없는 재생</th></tr></thead><tbody id="episodes"></tbody></table></div></section>
<section><h2>문제 장면 직접 보기</h2><p>게임 화면과 오른쪽 격자는 모두 <b>행동을 적용하기 전 같은 시점</b>입니다. 격자는 모델의 벽 입력이며 청록 화살표는 요청한 이동입니다. 선이 벽을 가로지르는지와 실제 기체 이동을 함께 확인하세요.</p><div class="row controls"><label for="clip">장면</label><select id="clip"></select><button id="play" type="button">재생</button><span id="frame-time"></span></div><img id="frame" class="viewer" alt="실제 게임 화면과 같은 시점의 정책 입력 벽 격자"><label for="scrubber">장면 시간 이동</label><input id="scrubber" type="range" min="0" value="0" step="1"><div id="frame-caption" class="frame-caption"></div><div class="legend"><span>청록: 기체 위치·이동 요청</span><span>주황: 포신 방향</span><span>분홍 표적선: 정확한 시야선에서 벽에 차단</span></div></section>
<section><h2>이동 정체가 발생한 시간대</h2><p class="muted">빨간 구간은 동일한 벽 방향 정체 조건이 1초 이상 이어진 구간입니다. 파란 점은 관측된 개별 벽 탄환 소멸 시점입니다. 점수나 벽 충돌률의 비교 척도가 아닙니다.</p><div id="timelines"></div></section>
<section><h2>확인된 사실과 해석을 구분하면</h2><div id="findings"></div></section>
<section><h2>측정 방법과 판단 범위</h2><div class="two"><div><h3>실행 조건</h3><ul><li>동일 모델·시드·greedy 행동을 두 번 재생합니다.</li><li>실제 게임은 60Hz, 정책 결정은 30Hz입니다.</li><li>원래 게임·보상·관찰·모델 코드를 변경하지 않습니다.</li><li>기본 맵의 파괴 가능한 벽 기능은 꺼져 있습니다. 월드 경계 탄환은 별도 집계합니다.</li><li>모든 시드는 이미 관측한 개발·회귀 확인용입니다. 새 독립 테스트가 아닙니다.</li></ul></div><div><h3>지표 정의</h3><ul><li><b>벽 방향 정체:</b> 이동 요청이 있고, 기체 반지름을 고려한 그 방향 벽 여유가 1px 이하이며, 한 결정의 실제 변위가 0.5px 이하인 상태입니다. 1초 이상 연속 구간을 따로 집계합니다.</li><li><b>벽 탄환 소멸:</b> 원래 탄환 이동 처리가 실제로 제거한 플레이어 탄환 중 개체 선행 명중·경계·파괴벽을 제외합니다.</li><li><b>격자 일치:</b> 매 1초 모델의 29×45 solid 채널을 같은 시점의 월드 지형과 대조합니다.</li><li><b>조준 표적 차단:</b> 매 1/3초 실제 조준망 입력의 선택 표적을 복원하고 정확한 중심 시야선으로 다시 검사합니다.</li></ul></div></div><p class="callout">정체 조건은 관측 기반 후보입니다. 적·천체와의 접촉이나 전술적 선택을 모두 배제한 인과 판정은 아닙니다. 벽에 탄환이 소모되었다고 모든 발사가 잘못됐다고 단정하지 않습니다. 시야선은 표적 중심 기준이며 탄환 반지름·다중 포신 판정과 구분합니다.</p><details><summary>계측의 정확성과 재현성</summary><div id="verification"></div></details><details><summary>원자료·모델·소스와 재실행</summary><p><a href="trace.jsonl" download>trace.jsonl 원자료</a> · HTML은 이미지와 데이터를 포함하므로 단독으로 열 수 있습니다.</p><pre id="provenance"></pre></details></section>
<footer>실제 재생 결과로 작성한 로컬 진단 보고서. 가중치 갱신·추가 PPO 학습·모델 승격은 수행하지 않았습니다.</footer>
</main><script id="report-data" type="application/json">__REPORT_DATA__</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent), E=D.episodes, fmt=(n,d=1)=>Number(n).toLocaleString('ko-KR',{maximumFractionDigits:d}), sum=k=>E.reduce((a,e)=>a+e[k],0);
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const seconds=sum('seconds'), stuck=sum('long_stationary_seconds'), walls=sum('permanent_wall_bullets'), fired=sum('shots_fired'), mismatch=sum('grid_mismatch_samples'), grids=sum('grid_checks'), targetBlocked=sum('blocked_target_checks'), targetChecks=sum('selected_target_checks');
$('date').textContent=`실행 ${D.metadata.created_at_kst} · 시드 ${E.map(e=>e.seed).join(', ')} · 관측 합계 ${fmt(seconds)}초`;
$('conclusions').innerHTML=`<p><b>${stuck>=1?'벽 쪽으로 이동을 요청하며 제자리에 머무는 구간이 재현됐습니다.':'정의한 1초 이상 벽 방향 정체는 이번 표본에서 재현되지 않았습니다.'}</b> 연속 정체 합계 ${fmt(stuck)}초, 벽에 제거된 탄환 ${fmt(walls,0)}발(${fired?fmt(walls/fired*100):'—'}%)을 관측했습니다.</p><p>지형 입력 대조 ${fmt(grids,0)}회 중 불일치 ${fmt(mismatch,0)}회. ${targetBlocked?`조준망에 전달된 표적을 정확한 시야선으로 확인했을 때 ${fmt(targetBlocked,0)}/${fmt(targetChecks,0)}회가 벽에 차단됐습니다.`:'선택 표적의 중심 시야선 차단은 이번 표본에서 검출되지 않았습니다.'}</p>`;
$('cards').innerHTML=[['연속 벽 방향 정체',`${fmt(stuck)}초`],['벽에 소모된 실제 탄환',`${fmt(walls,0)} / ${fmt(fired,0)}발`],['격자 대조 불일치',`${fmt(mismatch,0)} / ${fmt(grids,0)}회`],['전체 재생 일치',`${E.filter(e=>e.uninstrumented_replay_matches).length} / ${E.length}시드`]].map(([label,value])=>`<div class="card"><span>${label}</span><strong>${value}</strong></div>`).join('');
$('episodes').innerHTML=E.map(e=>`<tr><td>${e.seed}</td><td>${fmt(e.seconds)}초</td><td>${fmt(e.score,0)} / ${esc(e.ship)}</td><td>${e.stationary_segments.length}개 · ${fmt(e.long_stationary_seconds)}초</td><td>${e.permanent_wall_bullets} / ${e.shots_fired} (${e.shots_fired?fmt(e.permanent_wall_bullets/e.shots_fired*100):'—'}%)</td><td>${e.grid_mismatch_samples} / ${e.grid_checks}</td><td>${e.uninstrumented_replay_matches?'전체 일치':'미검증'}</td></tr>`).join('');
const names={movement:'벽 방향 이동 정체',visibility:'조준 표적 시야선 차이',shooting:'실제 탄환 벽 소멸',reference:'참고 장면'};
const clips=D.clips.filter(c=>c.frames.length);let active=0,index=0,timer=null;
$('clip').innerHTML=clips.map((c,i)=>`<option value="${i}">시드 ${c.seed} · ${names[c.kind]} · ${fmt(c.center_seconds)}초</option>`).join('');
function show(){const c=clips[active],f=c?.frames[index];if(!f){$('frame-caption').textContent='이 조건에서 캡처할 장면이 없습니다.';return;}$('frame').src=f.image;$('scrubber').max=c.frames.length-1;$('scrubber').value=index;$('frame-time').textContent=`${fmt(f.time_seconds,2)}초 · ${index+1}/${c.frames.length}`;const prev=c.frames[Math.max(0,index-1)],travel=Math.hypot(f.position[0]-prev.position[0],f.position[1]-prev.position[1]);$('frame-caption').textContent=`행동 전 위치 (${fmt(f.position[0])}, ${fmt(f.position[1])})px · 이동 번호 ${f.action[0]} · 발사 요청 ${f.action[2]} · 직전 캡처 이후 실제 변위 ${fmt(travel)}px · ${f.visibility.selected?(f.visibility.blocked?'선택 표적 중심 시야선: 벽 차단':'선택 표적 중심 시야선: 통과'):'조준망 선택 표적 없음'}`;}
function stop(){clearInterval(timer);timer=null;$('play').textContent='재생';}
$('clip').onchange=()=>{stop();active=Number($('clip').value);index=0;show();};$('scrubber').oninput=()=>{stop();index=Number($('scrubber').value);show();};$('play').onclick=()=>{if(timer){stop();return;}$('play').textContent='일시정지';timer=setInterval(()=>{index=(index+1)%clips[active].frames.length;show();},500);};show();
$('timelines').innerHTML=E.map(e=>{const width=1000,left=60,range=900/D.metadata.max_seconds;const segments=e.stationary_segments.map(s=>`<rect x="${left+s.start_seconds*range}" y="25" width="${Math.max(1,(s.end_seconds-s.start_seconds)*range)}" height="24" fill="#b74743"><title>${s.start_seconds.toFixed(2)}–${s.end_seconds.toFixed(2)}초 · ${s.duration_seconds.toFixed(2)}초 정체</title></rect>`).join('');const points=D.wall_times.filter(t=>t.seed===e.seed).map(t=>`<circle cx="${left+t.time_seconds*range}" cy="62" r="2" fill="#377eaf"><title>${t.time_seconds.toFixed(2)}초 · 벽 탄환 ${t.count}발</title></circle>`).join('');const ticks=[0,50,100,150,200].filter(t=>t<=D.metadata.max_seconds).map(t=>`<text x="${left+t*range}" y="93" text-anchor="middle" font-size="13" fill="#596982">${t}s</text>`).join('');return `<h3>시드 ${e.seed} · 관찰 ${fmt(e.seconds)}초</h3><svg class="timeline" viewBox="0 0 ${width} 105" role="img" aria-label="시드 ${e.seed}의 이동 정체와 벽 탄환 소멸 시간대"><rect x="60" y="20" width="900" height="58" fill="#f1f4f8"/><rect x="60" y="20" width="${e.seconds*range}" height="58" fill="#e4ebe9"/>${segments}${points}${ticks}</svg>`;}).join('');
$('findings').innerHTML=`<div class="finding"><span class="tag">관측</span><b>벽 입력 전달</b><p>${mismatch?`격자 위치 대조에서 불일치가 ${mismatch}회 검출됐습니다. 해당 시점의 원자료와 캡처를 우선 확인해야 합니다.`:`${grids}개 표본에서 solid 채널이 실제 월드 지형과 일치했습니다. 적어도 이 표본에서는 벽 정보의 전체 누락이나 타일 위치 오류가 발견되지 않았습니다.`} 이는 CNN이 벽을 유용하게 표현하거나 정책이 회피를 학습했다는 증명은 아닙니다.</p></div><div class="finding"><span class="tag ${stuck>=1?'warn':'note'}">${stuck>=1?'재현':'미재현'}</span><b>벽 방향 이동 정체</b><p>1초 이상 연속 구간 ${E.reduce((n,e)=>n+e.stationary_segments.length,0)}개, 총 ${fmt(stuck)}초입니다. 입력이 맞아도 이런 행동이 나올 수 있으므로 특징 표현·정책·보상과 주변 전투 상황을 구분해 후속 검토해야 합니다.</p></div><div class="finding"><span class="tag">관측</span><b>실제 사격의 벽 차단</b><p>실제 발사 ${fired}발 중 ${walls}발이 맵 내부의 파괴 불가능한 벽에 제거됐습니다. 다중 포신의 일부 탄환이나 이전에 발사한 탄환도 포함되며, 이 수치만으로 모델의 의도나 모든 발사의 전술적 가치를 판단하지 않습니다.</p></div><div class="finding"><span class="tag ${targetBlocked?'warn':'note'}">${targetBlocked?'불일치 관측':'미검출'}</span><b>조준망의 선택 표적과 정확한 시야선</b><p>선택 표적이 존재한 ${targetChecks}회 중 ${targetBlocked}회는 정확한 월드 중심 시야선에서 벽에 차단됐습니다. 현재 조준 전처리는 20개 격자 표본과 타일 중앙 가정을 사용합니다. ${targetBlocked?'근사 가시성 판정이 실제 지형과 어긋나는 경로가 있으므로, 사격 문제를 추가 학습만의 문제로 단정하기 전에 해당 경로를 확인하는 것이 우선입니다.':'이 표본에서 차이가 없다는 사실이 모든 위치의 가시성 판정을 보장하지는 않습니다.'}</p></div><div class="finding"><span class="tag note">다음 판단</span><b>이번 보고서는 수정·학습 방향을 고르기 위한 근거입니다.</b><p>재현한 장면에서 입력·전처리 오류를 먼저 확인하고, 남은 벽 회피 행동은 별도 학습 실험으로 비교하는 순서를 제안합니다. 전직 수정 후 추가 학습과 탐색 보상 변경은 이번 진단에 포함하지 않았습니다.</p></div>`;
$('verification').innerHTML=`<ul><li>실제 벽 충돌과 벽 앞 개체 선행 명중 구분 self-check: ${D.self_check.actual_wall_disposal&&D.self_check.entity_before_wall_not_double_counted?'통과':'실패'}</li><li>입력 격자를 한 칸 이동시킨 음성 대조군 검출: ${D.self_check.shifted_grid_detected?'통과':'실패'}</li><li>계측한 첫 재생과 계측 없는 캡처 재생의 모든 결정별 관찰·행동·점수·처치·전투 이벤트 해시 일치: ${E.every(e=>e.uninstrumented_replay_matches)?'모든 시드 통과':'미완료'}</li><li>실행 전후 보호 소스와 모델 SHA256: ${D.protected_files_unchanged?'동일':'확인 필요'}</li><li>원자료는 매 1초 요약과 정체·벽 탄환·시야선 차이 구간 중심으로 저장합니다. 기록이 없는 결정은 오류나 0으로 간주하지 않습니다.</li></ul>`;
$('provenance').textContent=JSON.stringify({metadata:D.metadata,self_check:D.self_check,episodes:E.map(({samples,...e})=>e),sources:D.sources},null,2);
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'selected_model.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/walls')
    parser.add_argument('--seeds', type=int, nargs='+', default=[505, 61001, 61102])
    parser.add_argument('--seconds', type=int, default=200)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0 or not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive duration and distinct nonnegative seeds are required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = PPO.load(args.checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    checks = self_check(model)
    if args.self_check:
        print(encode_json(checks), flush=True)
        return
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error('Use a new output directory inside gpt_training_rotation_continuous')
    protected = {str(path): sha256(path) for path in [args.checkpoint, Path(__file__), *SOURCE_FILES]}
    output.mkdir(parents=True)
    metadata = dict(created_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                    source_sha256_before_run=protected,
                    seeds=args.seeds, max_seconds=args.seconds, replay_mode='greedy',
                    frames_per_decision=2, game_fps=60, parallel_environment_workers=1,
                    ppo_updates=0, max_capture_interval_seconds=0.5,
                    rerun_command='python -B -m space_z.rl.gpt_training_rotation_continuous.diagnose_walls --output <new-directory>')
    episodes, candidates, wall_times = [], [], []
    with (output / 'trace.jsonl').open('w', encoding='utf-8') as trace:
        def write_record(record):
            trace.write(encode_json(record) + '\n')
            if record['type'] == 'step' and record['permanent_wall_bullet_count']:
                wall_times.append(dict(seed=record['seed'], time_seconds=record['time_seconds'],
                                       count=record['permanent_wall_bullet_count']))
        write_record(dict(type='metadata', **metadata))
        for seed in args.seeds:
            summary, found = scan_episode(model, seed, args.seconds, write_record)
            episodes.append(summary)
            candidates.extend(found)
            trace.flush()
            print(encode_json(dict(seed=seed, score=summary['score'], seconds=summary['seconds'],
                                   stationary_seconds=summary['long_stationary_seconds'], wall_bullets=summary['permanent_wall_bullets'],
                                   visibility_mismatches=summary['blocked_target_checks'])), flush=True)
        clips = choose_clips(candidates, episodes)
        for episode in episodes:
            replay_and_capture(model, episode, clips, args.seconds)
        assert all(sha256(Path(path)) == fingerprint for path, fingerprint in protected.items())
        report = dict(metadata=metadata, episodes=episodes, clips=clips, wall_times=wall_times,
                      self_check=checks, protected_files_unchanged=True, sources=protected)
        write_record(dict(type='verification', protected_files_unchanged=True, self_check=checks,
                          replay_verified_seeds=[episode['seed'] for episode in episodes], sources=protected))
    write_report(output / 'report.html', report)
    print(f'Report: {output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
