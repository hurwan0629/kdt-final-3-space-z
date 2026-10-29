"""Local training monitor: real training metrics plus an explicitly labelled policy demo."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np
import pygame
import psutil
import torch

from .runtime import atomic_json, read_json, load_checkpoint, normalizer
from .settings import TrainSettings, STAGE_NAMES

BG, PANEL, EDGE = (9, 15, 29), (17, 27, 46), (44, 64, 86)
WHITE, MUTED, CYAN, GREEN, GOLD, RED = ((230, 240, 249), (139, 159, 180),
    (80, 212, 242), (111, 226, 163), (246, 203, 106), (255, 115, 132))
SIZE = (1360, 900)
MOVES = ('정지', '위', '우상', '오른쪽', '우하', '아래', '좌하', '왼쪽', '좌상')
ROTS = ('유지', '반시계', '시계')
STATUS = {'starting': '준비 중', 'baseline': '무작위 기준선 평가', 'training': '학습 중',
          'evaluating': '고정 난이도 평가 중', 'paused': '학습 일시정지', 'stopping': '저장 후 중지 중',
          'stopped': '저장하고 중지됨', 'completed': '학습 예산 완료', 'failed': '학습 오류'}


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    value = max(0, int(seconds))
    return f'{value//3600:02}:{value//60%60:02}:{value%60:02}'


class UI:
    def __init__(self):
        sans = 'malgungothic,notosanscjkkr,segoeui,arial'
        self.fonts = {s: pygame.font.SysFont(sans, s, bold=s >= 19) for s in (13, 14, 16, 19, 24, 30)}

    def text(self, surface, value, xy, size=16, color=WHITE, max_width=None):
        value = str(value)
        font = self.fonts[size]
        if max_width:
            while value and font.size(value)[0] > max_width:
                value = value[:-2]
            if value != str(value):
                value += '…'
        surface.blit(font.render(value, True, color), xy)

    def box(self, screen, rect):
        pygame.draw.rect(screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(screen, EDGE, rect, 1, border_radius=10)

    def plot(self, screen, rect, title, series, baseline=None):
        self.box(screen, rect)
        self.text(screen, title, (rect.x+14, rect.y+10), 14)
        area = pygame.Rect(rect.x+48, rect.y+40, rect.width-66, rect.height-67)
        pygame.draw.line(screen, EDGE, area.bottomleft, area.bottomright)
        pygame.draw.line(screen, EDGE, area.topleft, area.bottomleft)
        points = [(float(x), float(y), float(sd or 0)) for x, y, sd in series
                  if y is not None and math.isfinite(y)]
        if not points:
            self.text(screen, '실제 기록이 쌓이면 표시됩니다', (area.x+18, area.y+28), 14, MUTED)
            return
        ys = [y+s for _, y, s in points] + [y-s for _, y, s in points]
        if baseline is not None:
            ys.append(baseline)
        lo, hi = min(ys), max(ys)
        if hi-lo < 1:
            lo -= .5
            hi += .5
        xmin, xmax = min(x for x, _, _ in points), max(x for x, _, _ in points)
        def pixel(x, y):
            return (int(area.x+(x-xmin)/max(1., xmax-xmin)*area.width),
                    int(area.bottom-(y-lo)/(hi-lo)*area.height))
        if baseline is not None:
            yy = pixel(xmin, baseline)[1]
            for xx in range(area.x, area.right, 10):
                pygame.draw.line(screen, GOLD, (xx, yy), (min(xx+5, area.right), yy))
        xy = [pixel(x, y) for x, y, _ in points]
        if len(xy) > 1:
            pygame.draw.lines(screen, CYAN, False, xy, 2)
        for (x, y, sd), point in zip(points, xy):
            if sd:
                pygame.draw.line(screen, MUTED, pixel(x, y-sd), pixel(x, y+sd))
            pygame.draw.circle(screen, GREEN, point, 3)
        self.text(screen, f'{hi:.1f}', (rect.x+4, area.y-6), 13, MUTED)
        self.text(screen, f'{lo:.1f}', (rect.x+4, area.bottom-12), 13, MUTED)
        self.text(screen, f'{int(xmax):,} step', (area.right-110, area.bottom+5), 13, MUTED)


class Demo:
    def __init__(self, directory):
        self.directory = directory
        self.model = self.env = self.obs = self.renderer = None
        self.entry = None
        self.last_check = 0
        self.last_action = 0
        self.last_stat=self.last_evolution=0
        self.last_parts = {}
        self.probs = np.zeros(54)
        self.error = ''
        self.game_key = None

    def poll(self, force=False):
        if not force and time.monotonic()-self.last_check < 2.:
            return False
        self.last_check = time.monotonic()
        entry = (read_json(self.directory/'best.json')
                 or read_json(self.directory/'latest.json'))
        if not entry or entry == self.entry:
            return False
        try:
            cfg = replace(TrainSettings(**entry['settings']), workers=1, torch_threads=1)
            model, env = load_checkpoint(self.directory, entry, cfg=cfg)
            obs = env.reset()
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            self.error = f'정책 읽기 대기: {error}'
            return False
        if self.env is not None:
            self.env.close()
        self.model, self.env, self.obs, self.entry = model, env, obs, entry
        self.error = ''
        return True

    @property
    def raw(self):
        if self.env is None:
            return None
        inner = self.env
        while hasattr(inner, 'venv'):
            inner = inner.venv
        return inner.envs[0].env

    def advance(self, deterministic=False):
        if self.model is None:
            return
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(self.obs)
            if self.entry['settings']['algorithm'] == 'ppo':
                distribution=self.model.policy.get_distribution(tensor).distribution
                categorical=distribution[0] if self.entry['settings'].get('full_growth') else distribution
                self.probs = categorical.probs[0].cpu().numpy()
            else:
                self.probs = torch.softmax(self.model.q_net(tensor)[0], dim=0).cpu().numpy()
            action = self.model.predict(self.obs, deterministic=deterministic)[0]
        if self.entry['settings'].get('full_growth'):
            self.last_action,self.last_stat,self.last_evolution=(int(x) for x in action[0])
        else:
            self.last_action = int(action[0])
        self.obs, _, _, infos = self.env.step(action)
        self.last_parts = infos[0].get('reward_components', {})

    def render_game(self, screen):
        if self.raw is None:
            return
        from space_z.rendering.renderer import Renderer
        game = self.raw.env.game
        key = (game.config.world_width, game.config.world_height)
        if key != self.game_key:
            self.renderer = Renderer(game.config, 'Space Z · 강화학습 모니터', visible=False)
            # Keep the existing game's renderer completely unchanged.
            pygame.display.set_mode(SIZE)
            self.renderer.screen = pygame.Surface((960, 640)).convert()
            self.renderer.set_zoom(.6)
            self.game_key = key
        self.renderer.draw(game, show_controls=False)
        screen.blit(self.renderer.screen, (0, 62))

    def close(self):
        if self.env is not None:
            self.env.close()


def draw_panel(ui, screen, state, demo, deterministic, speed):
    ui.box(screen, pygame.Rect(972, 62, 376, 824))
    x = 989
    step, target = state.get('step', 0), state.get('target', 2000000)
    ui.text(screen, '실제 학습 작업', (x, 76), 19)
    ui.text(screen, f'{step:,} / {target:,} step', (x, 109), 19, CYAN)
    pygame.draw.rect(screen, EDGE, (x, 141, 338, 6), border_radius=3)
    pygame.draw.rect(screen, CYAN, (x, 141, int(338*min(1., step/max(1, target))), 6), border_radius=3)
    phase = min(2, max(0, state.get('stage', 0)))
    growth_mode=(state.get('settings') or {}).get('full_growth',False)
    ui.text(screen, '전체 성장 · 레벨/스탯/전직 포함' if growth_mode else STAGE_NAMES[phase], (x, 158), 16, GOLD)
    ui.text(screen, f"{state.get('workers', 4)} 환경 · {state.get('fps', 0):.0f} step/s · CPU", (x, 184), 14, MUTED)
    ui.text(screen, f"경과 {duration(state.get('elapsed'))}  예상 {duration(state.get('eta'))}", (x, 208), 14)
    ui.text(screen, f"에피소드 {state.get('episode_count', 0)} · 업데이트 {state.get('updates', 0)}", (x, 231), 14)
    ui.text(screen, '평가 최고 정책 시연 · 별도 환경', (x, 265), 19)
    if demo.raw is not None:
        raw, game = demo.raw, demo.raw.env.game
        if raw.full_growth:
            from space_z.config.progression_definitions import SHIPS
            heading=f"Lv.{game.progression.level} {SHIPS[game.player.job].title} · {demo.entry['step']:,} step"
        else:
            heading=f"정책 저장 시점 {demo.entry['step']:,} step"
        ui.text(screen, heading, (x, 295), 14, MUTED)
        ui.text(screen, f'시연 체력 {game.player.hp:.0f}  생존 {game.frame/60:.1f}s  처치 {game.progression.kills}', (x, 319), 14)
        summary = (f'획득 점수 {game.progression.score:,} · 스탯 {raw.stat_upgrades} · 전직 {raw.evolutions}'
                   if raw.full_growth else f'획득 점수 {game.progression.score:,} · 목표 100초'
                   if raw.score_objective else
                   f'시연 보상 {raw.episode_return:+.2f}  기본 과제 {raw.task_return:+.2f}')
        ui.text(screen, summary, (x, 343), 14, GREEN)
        a = demo.last_action
        ui.text(screen, f'행동 {a:02d} · {MOVES[a//6]} / {ROTS[(a//2)%3]} · 발사 {a%2} · {speed}x', (x, 367), 14)
        label = 'PPO 행동 확률' if demo.entry['settings']['algorithm'] == 'ppo' else 'DQN Q값 softmax 시각화'
        ui.text(screen, label + (' · 결정론' if deterministic else ' · 탐색'), (x, 400), 14, CYAN)
        for i, idx in enumerate(np.argsort(demo.probs)[-3:][::-1]):
            ui.text(screen, f'{MOVES[idx//6]} / {ROTS[(idx//2)%3]}', (x, 424+i*21), 13)
            width = int(125*demo.probs[idx])
            pygame.draw.rect(screen, EDGE, (x+149, 430+i*21, 125, 5))
            pygame.draw.rect(screen, CYAN, (x+149, 430+i*21, width, 5))
            ui.text(screen, f'{demo.probs[idx]*100:.0f}%', (x+286, 423+i*21), 13, MUTED)
        ui.text(screen, '시연 누적 보상 분해', (x, 497), 14, GOLD)
        labels = [('damage', '적 피해'), ('kills', '적 처치'), ('shapes', '천체'),
                  ('hurt', '피격'), ('death', '사망'), ('time', '시간'), ('shots', '발사 비용')]
        if raw.score_objective:
            labels = [('game_score', '점수 기여'), ('death', '사망')]
        for i, (key, label) in enumerate(labels):
            ui.text(screen, f'{label:<9} {raw._components.get(key, 0):+7.3f}', (x, 522+i*19), 13)
        if raw.full_growth:
            levels=game.progression.stat_levels
            ui.text(screen,f"체{levels['health']} 공{levels['damage']} 사{levels['range']} 연{levels['rapid']} 이{levels['speed']}",(x,575),13,CYAN)
            ui.text(screen,f'남은 포인트 {game.progression.upgrade_points} · 전직 대기 {len(game.progression.special_reward_levels)}',(x,599),13)
            ui.text(screen,f'숙련 적용 {raw.masteries}회',(x,623),13,MUTED)
        # Show the latest wall-channel input, not the colored game picture.
        grid = raw._observation(raw.builder.build(game))['grid'][0]
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                pygame.draw.rect(screen, (125, 147, 167) if grid[r, c] else (28, 45, 65),
                                 (x+225+c*2, 523+r*2, 2, 2))
        pygame.draw.circle(screen, CYAN, (x+277, 575), 3)
        ui.text(screen, '벽 관찰 45×29', (x+220, 633), 13, MUTED)
        if raw.corner_recovery and raw.recovery.remaining:
            ui.text(screen, '모서리 탈출 중 · 이동 방향 유지', (x,652),13,CYAN)
        elif raw.corner_recovery and raw.recovery.count:
            ui.text(screen, f'모서리 탈출 시도 {raw.recovery.count}회', (x,652),13,CYAN)
        elif raw.crowded_steps:
            ui.text(screen, f'다수 적 근접 중 후퇴 {raw.retreat_steps/raw.crowded_steps:.0%}',
                    (x, 652), 13, CYAN)
    else:
        ui.text(screen, '첫 정책 저장을 기다리는 중…', (x, 304), 14, MUTED)
    ui.text(screen, '학습 안정성 · 실제 최적화 기록', (x, 678), 16)
    loss = state.get('loss', {})
    for i, (key, label) in enumerate([('entropy_loss', 'entropy loss'), ('approx_kl', 'approx KL'),
                                    ('value_loss', 'value loss'), ('explained_variance', '설명 분산')]):
        value = loss.get(key)
        ui.text(screen, f'{label}: {value:.4f}' if value is not None else f'{label}: --', (x, 706+i*22), 14, MUTED)
    ui.text(screen, '전투 54 × 스탯 6 × 전직 5' if growth_mode else 'main 관찰 × 4시점 · 정책 54행동', (x, 808), 13, CYAN)
    caption = ('100초 획득 점수가 가장 높은 정책을 보존합니다'
               if (state.get('settings') or {}).get('objective') == 'score100'
               else '학습 보상과 게임 점수는 서로 다릅니다')
    ui.text(screen, caption, (x, 833), 13, MUTED)
    if state.get('error'):
        ui.text(screen, state['error'], (12, 680), 14, RED, 940)


def run_dashboard(directory, capture=None, max_frames=None, stop_on_close=True):
    directory = Path(directory)
    pygame.init()
    torch.set_num_threads(1)
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption('Space Z · PPO/DQN 실시간 강화학습')
    clock, ui, demo = pygame.time.Clock(), UI(), Demo(directory)
    deterministic, speed, elapsed, frames = True, 1, 0., 0
    state = {}
    quitting = False
    try:
        while not quitting:
            dt = min(.2, clock.tick(30)/1000.)
            state = read_json(directory/'status.json', state)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    quitting = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        command = read_json(directory/'control.json', {}).get('command', 'run')
                        atomic_json(directory/'control.json', {'command': 'run' if command == 'pause' else 'pause'})
                    if event.key == pygame.K_d:
                        deterministic = not deterministic
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = min(4, speed*2)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(1, speed//2)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.Rect(1100, 12, 100, 34).collidepoint(event.pos):
                        command = read_json(directory/'control.json', {}).get('command', 'run')
                        atomic_json(directory/'control.json', {'command': 'run' if command == 'pause' else 'pause'})
                    if pygame.Rect(1210, 12, 138, 34).collidepoint(event.pos):
                        quitting = True
            demo.poll(force=frames == 0)
            if demo.model is not None:
                elapsed += dt*15*speed
                for _ in range(min(8, int(elapsed))):
                    demo.advance(deterministic)
                    elapsed -= 1
            screen.fill(BG)
            demo.render_game(screen)
            ui.text(screen, 'SPACE Z  /  LEARNING LAB', (16, 13), 24, WHITE)
            status = state.get('status', 'starting')
            ui.text(screen, STATUS.get(status, status), (470, 19), 16, RED if status == 'failed' else CYAN)
            if state.get('pid') and status not in ('completed', 'stopped', 'failed') and not psutil.pid_exists(state['pid']):
                ui.text(screen, '학습 프로세스가 종료됐습니다 · trainer.log 확인', (470, 19), 14, RED)
            pygame.draw.rect(screen, EDGE, (1100, 12, 100, 34), border_radius=6)
            ui.text(screen, '재개' if status == 'paused' else '일시정지', (1113, 19), 14)
            pygame.draw.rect(screen, (91, 39, 53), (1210, 12, 138, 34), border_radius=6)
            ui.text(screen, '저장 후 종료', (1225, 19), 14)
            draw_panel(ui, screen, state, demo, deterministic, speed)
            eps = state.get('episodes', [])
            score_mode = (state.get('settings') or {}).get('objective') == 'score100'
            ui.plot(screen, pygame.Rect(8, 712, 469, 150),
                    '훈련 누적 점수 · 레벨 1부터 성장' if (state.get('settings') or {}).get('full_growth') else '100초 획득 점수 · 사망 시 종료' if score_mode else '훈련 과제 보상 · 난이도 변경 시 주의',
                    [(e.get('global_step', i), e['score'] if score_mode else e['task_return'], 0) for i, e in enumerate(eps)])
            evaluations = state.get('evaluations', [])
            baseline = state.get('baseline') or {}
            ui.plot(screen, pygame.Rect(486, 712, 474, 150),
                    '100초 점수 평가 · 평균±표준편차' if score_mode else '고정 난이도 평가 · 평균±표준편차 / 노랑: 무작위',
                    [(e['step'], e['mean'], e['std']) for e in evaluations], baseline.get('mean'))
            ui.text(screen, 'P 학습 일시정지/재개   D 결정론 시연   +/- 시연 속도   ESC 저장 후 종료', (14, 875), 14, MUTED)
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if capture:
            Path(capture).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(capture))
    finally:
        if stop_on_close:
            atomic_json(directory/'control.json', {'command': 'stop'})
        demo.close()
        pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--capture')
    p.add_argument('--frames', type=int)
    p.add_argument('--watch-only', action='store_true', help='Closing this viewer does not stop training')
    a = p.parse_args()
    run_dashboard(a.run_dir, a.capture, a.frames, stop_on_close=not a.watch_only)


if __name__ == '__main__':
    main()
