"""Read-only CMD monitor for wooki02; closing it never stops training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time


def display(directory):
    try:
        state = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        print('상태 파일을 기다리는 중입니다.', flush=True)
        return
    step, target = state.get('step', 0), state.get('target', 2_000_000)
    fps = state.get('fps', 0.)
    eta = state.get('eta')
    remaining = f'{eta / 3600:.1f}시간' if eta is not None else '계산 중'
    names = {'training': '학습 중', 'evaluating': '평가 중', 'baseline': '무작위 기준선 평가',
             'starting': '시작 중', 'stopped': '중단됨', 'completed': '완료', 'failed': '오류',
             'paused': '일시정지', 'stopping': '저장 후 중단 중'}
    print('\n' + '=' * 76)
    label = 'WOOKI01' if directory.name == 'ppo-full-growth' else 'WOOKI02'
    print(f"{label} | {datetime.now():%H:%M:%S} | {names.get(state.get('status'), state.get('status'))}")
    print(f'진행: {step:,} / {target:,} ({100 * step / max(1, target):.2f}%)')
    print(f"속도: {fps:.1f} step/s | 병렬: {state.get('workers', 0)}개 | 예상 남은 시간: {remaining}")
    print(f"완료한 판: {state.get('episode_count', 0)} | PPO 업데이트: {state.get('updates', 0)}")
    age = time.time() - state.get('updated', time.time())
    if age > 30 and state.get('status') not in ('completed', 'stopped', 'failed'):
        print(f'확인 필요: 상태 기록이 {age:.0f}초 동안 갱신되지 않았습니다.')
    episodes = state.get('episodes', [])
    if episodes:
        recent = episodes[-10:]
        print(f"최근 {len(recent)}판 평균: 보상 {statistics.mean(e['return'] for e in recent):+.3f}"
              f" | 게임 점수 {statistics.mean(e['score'] for e in recent):.1f}")
        ep = episodes[-1]
        parts = ep.get('components', {})
        print('-' * 76)
        print(f"마지막 종료 판: 보상 {ep['return']:+.3f} | 점수 {ep['score']:,} | 처치 {ep['kills']}")
        print(f"생존 {ep['seconds']:.1f}초 | 레벨 {ep.get('level', 1)} | 받은 피해 {ep['hurt']:.1f}")
        print(f"보상 분해: 점수 {parts.get('game_score', 0):+.3f}"
              f" / 피격 {parts.get('hurt', 0):+.3f}"
              f" / 벽 막힘 {parts.get('wall_stall', 0):+.3f}"
              f" / 사망 {parts.get('death', 0):+.3f}")
        print(f"스탯: {ep.get('stat_levels', {})} | 전직: {' -> '.join(ep.get('ship_history', []))}")
    evaluations = state.get('evaluations', [])
    if evaluations:
        print(f"최근 평가 평균 게임 점수: {evaluations[-1]['mean']:.1f}")
    if state.get('error'):
        print(f"오류: {state['error']}")
    print('5초마다 갱신 | 이 창은 기록 조회 전용입니다. Ctrl+C 또는 창 닫기로 조회만 종료합니다.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runs/ppo-wooki02')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            display(args.run_dir)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n조회 창을 종료합니다. 학습 프로세스는 계속 실행됩니다.')


if __name__ == '__main__':
    main()
