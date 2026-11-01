"""커리큘럼 학습 체크포인트가 게임을 플레이하는 모습을 창으로 보거나 GIF로 저장한다.

실행 (프로젝트 루트):
    python -m space_z.rl.watch_haewoon                       # 최근 체크포인트, 원래 게임(긴 판)
    python -m space_z.rl.watch_haewoon --stage 6 --episodes 3
    python -m space_z.rl.watch_haewoon --gif runs/curriculum/play.gif --seconds 90
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "curriculum")
    parser.add_argument("--checkpoint", type=Path, help="model.zip 경로 (기본: state.json의 latest)")
    parser.add_argument("--stage", type=int, default=11, help="플레이할 단계 (기본 11: 원래 게임, 긴 판)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="확률적으로 행동 선택 (기본은 결정적)")
    parser.add_argument("--gif", type=Path, help="창 대신 이 경로에 저장 (.webp 권장: GIF보다 훨씬 작음, .gif도 가능)")
    parser.add_argument("--seconds", type=float, default=600.0, help="녹화 길이(게임 시간, 초)")
    parser.add_argument("--width", type=int, default=480, help="영상 가로 크기")
    parser.add_argument("--every", type=int, default=4, help="N 프레임마다 하나만 저장 (4면 4배속 영상)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # train_haewoon은 import 시 화면 없는 드라이버를 기본값으로 지정하므로 창 모드는 먼저 덮어쓴다.
    os.environ["SDL_VIDEODRIVER"] = "dummy" if args.gif else ("windows" if sys.platform == "win32" else "")
    if not os.environ["SDL_VIDEODRIVER"]:
        del os.environ["SDL_VIDEODRIVER"]

    import pygame
    from stable_baselines3 import PPO
    from space_z.rl import train_haewoon as tc
    from space_z.rl.environment import TankEnv

    checkpoint = args.checkpoint
    if checkpoint is None:
        state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8"))
        checkpoint = args.run_dir / state["latest"] / "model.zip"
    model = PPO.load(checkpoint, device="cpu")
    max_steps = 9000 if args.stage >= 11 else 3000

    class WatchEnv(tc.CurriculumEnv):
        def _build(self) -> TankEnv:
            return TankEnv(tc.stage_game_config(self.stage), tc.VectorObservation(self.obs_config), self.decoder,
                           tc.RewardCalculator(tc.REWARD), tc.EnvConfig(frame_skip=4, max_steps=self.max_steps),
                           render_mode="human")

    env = WatchEnv(args.stage, max_steps)
    print(f"체크포인트 {checkpoint}, 단계 {tc.STAGES[args.stage]['name']}", flush=True)
    frames = []
    step_limit = int(args.seconds * env.env.metadata["render_fps"])
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        done, step = False, 0
        while not done:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated or env.env._closed
            step += 1
            if args.gif:
                if step % args.every == 0 or done:
                    frames.append(capture(pygame, args.width))
                if step >= step_limit:
                    break
        print(f"판 {episode + 1}: 점수 {info['score']:.0f}, 처치 {info['kills']}, 레벨 {info['level']}, "
              f"직업 {info['job']}, {info['steps']} step", flush=True)
        if args.gif or env.env._closed:
            break
    env.env.close()
    if args.gif:
        save_animation(frames, args.gif, env.env.metadata["render_fps"])
        print(f"저장: {args.gif} ({len(frames)} 프레임, {args.every}배속)", flush=True)


def capture(pygame, width: int):
    from PIL import Image
    surface = pygame.display.get_surface()
    image = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.BILINEAR)


def save_animation(frames, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"save_all": True, "duration": round(1000 / fps), "loop": 0}
    if path.suffix.lower() == ".gif":
        frames = [frame.quantize(colors=64) for frame in frames]
        options["optimize"] = True
    else:
        options["quality"] = 60
    frames[0].save(path, append_images=frames[1:], **options)


if __name__ == "__main__":
    main()
