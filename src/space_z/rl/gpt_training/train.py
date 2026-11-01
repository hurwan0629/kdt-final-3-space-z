"""순수 PPO를 기본으로 학습하고 고정 조건 평가를 실험별로 기록한다."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

# 작업 프로세스도 공유 소스 옆에 바이트코드 캐시를 쓰지 않도록 한다.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from torch.utils.tensorboard import SummaryWriter

from .environment import EnvironmentSettings, game_configuration, make_environment, teacher_action

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def save_model(model, path):
    temporary = path.with_suffix(".tmp.zip")
    model.save(temporary)
    temporary.replace(path)


def evaluate(action_selector, seeds, max_steps, stage="full", pure_rl=False, policy_seed=None, frame_skip=4):
    episodes = []
    for seed in seeds:
        with torch.random.fork_rng(devices=[]), closing(make_environment(stage, max_steps, pure_rl=pure_rl, frame_skip=frame_skip)) as environment:
            if policy_seed is not None:
                torch.manual_seed(policy_seed + seed)
            observation, info = environment.reset(seed=seed)
            done = False
            score_milestones = {}
            fps = environment.unwrapped.game_config.fps
            actual_frame_skip = environment.unwrapped.environment_config.frame_skip
            horizon_seconds = max_steps * actual_frame_skip / fps
            score_times = [seconds for seconds in (60, 120, 200) if seconds <= horizon_seconds]
            while not done:
                action = action_selector(observation)
                observation, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
                for seconds in score_times:
                    if seconds not in score_milestones and info['frames'] >= seconds * fps:
                        score_milestones[seconds] = info['score']
            episode = {
                "seed": seed, "score": info["score"], "kills": info["kills"],
                "survival_seconds": info["frames"] / environment.unwrapped.game_config.fps,
                "terminated": terminated, "truncated": truncated,
                "ship_class": environment.unwrapped.game.player.job,
                "stat_levels": dict(environment.unwrapped.game.progression.stat_levels),
                **info["experiment_episode"],
            }
            for seconds in score_times:
                episode[f'score_at_{seconds}_seconds'] = score_milestones.get(seconds, info['score'])
            episodes.append(episode)
    metrics = ["score", "kills", "survival_seconds", "shots", "shapes_destroyed", "damage_taken"]
    metrics.extend(key for key in episodes[0] if key.startswith('score_at_'))
    summary = {
        metric: {"mean": float(np.mean([episode[metric] for episode in episodes])),
                 "std": float(np.std([episode[metric] for episode in episodes])),
                 "median": float(np.median([episode[metric] for episode in episodes])),
                 "p25": float(np.percentile([episode[metric] for episode in episodes], 25)),
                 "min": float(min(episode[metric] for episode in episodes)),
                 "max": float(max(episode[metric] for episode in episodes))}
        for metric in metrics
    }
    return {"stage": stage, "max_steps": max_steps, "frame_skip": actual_frame_skip, "max_seconds": horizon_seconds,
            "pure_rl": pure_rl, "policy_seed": policy_seed, "episodes": episodes, "summary": summary}


def policy_action(model, observation, deterministic=True):
    return model.predict(observation, deterministic=deterministic)[0]


def monitored_environment(stage, max_steps, pure_rl=False):
    return Monitor(make_environment(stage, max_steps, pure_rl=pure_rl))


class ExperimentLog(BaseCallback):
    def __init__(self, run_directory, checkpoint_interval):
        super().__init__()
        self.run_directory = run_directory
        self.checkpoint_interval = checkpoint_interval
        self.next_checkpoint = checkpoint_interval

    def _on_step(self):
        for info in self.locals["infos"]:
            if "experiment_episode" not in info:
                continue
            record = {
                "timesteps": self.num_timesteps, "score": info["score"], "kills": info["kills"],
                "survival_seconds": info["frames"] / 60,
                **info["experiment_episode"],
            }
            append_json(self.run_directory / "episodes.jsonl", record)
            for key, value in record.items():
                if key != "timesteps":
                    self.logger.record_mean(f"game/{key}", value)
        if self.num_timesteps >= self.next_checkpoint:
            path = self.run_directory / "checkpoints" / f"step_{self.num_timesteps:09d}.zip"
            save_model(self.model, path)
            self.next_checkpoint = self.num_timesteps + self.checkpoint_interval
        return True


def collect_demonstrations(environments, steps, run_directory, model, teacher_rollin_probability):
    observations, actions = [], []
    observation = environments.reset()
    transitions = 0
    while transitions < steps:
        selected = np.stack([teacher_action(value) for value in observation])
        observations.append(observation.copy())
        actions.append(selected)
        executed = selected.copy()
        if teacher_rollin_probability < 1:
            learner_actions = model.predict(observation, deterministic=True)[0]
            use_learner = np.random.random(environments.num_envs) >= teacher_rollin_probability
            executed[use_learner] = learner_actions[use_learner]
        observation, _, _, _ = environments.step(executed)
        transitions += environments.num_envs
    observation_array = np.concatenate(observations)[:steps]
    action_array = np.concatenate(actions)[:steps]
    np.savez_compressed(run_directory / "demonstrations.npz", observations=observation_array, actions=action_array)
    return observation_array, action_array


def imitate(model, observations, actions, epochs, writer):
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.001)
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    for epoch in range(epochs):
        losses = []
        for indices in torch.randperm(len(observations), device=model.device).split(256):
            _, log_probabilities, _ = model.policy.evaluate_actions(observation_tensor[indices], action_tensor[indices])
            loss = -log_probabilities.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            distribution = model.policy.get_distribution(observation_tensor)
            predictions = distribution.mode()
            accuracies = (predictions == action_tensor).float().mean(dim=0).cpu().tolist()
        writer.add_scalar("imitation/loss", float(np.mean(losses)), epoch)
        for branch, accuracy in zip(["move", "turn", "fire"], accuracies):
            writer.add_scalar(f"imitation/{branch}_accuracy", accuracy, epoch)
        print(f"imitation epoch={epoch+1} loss={np.mean(losses):.3f} accuracies={accuracies}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--training-method", choices=["pure_ppo", "imitation_ppo"], default="pure_ppo")
    parser.add_argument("--stage", choices=["harvest", "combat", "full"], default="full")
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--demo-steps", type=int, default=None)
    parser.add_argument("--imitation-epochs", type=int, default=20)
    parser.add_argument("--teacher-rollin-probability", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    pure_rl = args.training_method == "pure_ppo"
    if args.demo_steps is None:
        args.demo_steps = 0 if pure_rl else 16384
    if pure_rl and args.demo_steps:
        parser.error("Pure PPO does not accept demonstrations; use --demo-steps 0")
    if not 1 <= args.num_envs <= 6:
        parser.error("Use at most 6 workers; Windows launcher + coordinator + workers <= 8")
    if min(args.steps, args.demo_steps) < 0 or min(args.eval_steps, args.episode_steps, args.imitation_epochs) <= 0:
        parser.error("Invalid training/evaluation lengths")
    if not 0 <= args.teacher_rollin_probability <= 1:
        parser.error("Teacher roll-in probability must be in [0, 1]")
    if Path(args.name).name != args.name or not args.name.strip():
        parser.error("Experiment name must be a single directory name")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_directory = ROOT / "experiments" / args.name
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "checkpoints").mkdir()
    manifest = {
        "name": args.name, "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_game": asdict(game_configuration(args.stage)),
        "evaluation_game": asdict(game_configuration("full")),
        "environment": asdict(EnvironmentSettings(args.stage, args.episode_steps, pure_rl=pure_rl)),
        "training_method": args.training_method,
        "growth_controller": "PPO stat/evolution action branches" if pure_rl else "fixed observation-based stat/evolution rule",
        "observations": "unmodified VectorObservation, 5139 features" if pure_rl else "164 tactical features",
        "actions": "MultiDiscrete([9, 9, 2, 6, 5])" if pure_rl else "MultiDiscrete([9, 9, 2])",
        "initialization": "resume" if args.resume else "random",
        "process_limit": 8, "environment_workers": args.num_envs,
        "versions": {"torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__},
        "best_selection_rule": "time-limit completion fraction, then mean game score, then mean survival",
    }
    with ZipFile(run_directory / "source_code.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.glob("*.py")):
            archive.write(path, path.name)
    manifest["source_sha256"] = hashlib.sha256((run_directory / "source_code.zip").read_bytes()).hexdigest()
    write_json(run_directory / "experiment.json", manifest)

    def record_evaluation(label, selector, writer):
        report = evaluate(selector, args.eval_seeds, args.eval_steps, pure_rl=pure_rl)
        record = {"label": label, "timesteps": model.num_timesteps, **report}
        append_json(run_directory / "evaluations.jsonl", record)
        for key, statistics in report["summary"].items():
            writer.add_scalar(f"evaluation/{label}/{key}", statistics["mean"], model.num_timesteps)
        print(f"EVALUATION {label}: {json.dumps(report['summary'])}", flush=True)
        if label != "teacher":
            completion_fraction = sum(episode["truncated"] for episode in report["episodes"]) / len(report["episodes"])
            ranking = [completion_fraction, report["summary"]["score"]["mean"], report["summary"]["survival_seconds"]["mean"]]
            if ranking > manifest.get("best_ranking", [-1, -1, -1]):
                save_model(model, run_directory / "checkpoints/best.zip")
                manifest["best_ranking"] = ranking
                manifest["best_label"] = label
                write_json(run_directory / "experiment.json", manifest)
        return report

    environments = None
    model = None
    started = time.monotonic()
    try:
        factories = [partial(monitored_environment, args.stage, args.episode_steps, pure_rl) for _ in range(args.num_envs)]
        environments = SubprocVecEnv(factories, start_method="spawn") if args.num_envs > 1 else DummyVecEnv(factories)
        environments.seed(args.seed)
        if args.resume:
            model = PPO.load(args.resume.resolve(), env=environments, device="cpu", tensorboard_log=str(run_directory / "tensorboard"))
            if pure_rl and getattr(model, "training_method", None) != "pure_ppo":
                raise ValueError("Pure PPO can only resume checkpoints with recorded pure-PPO provenance")
            manifest["parent_checkpoint_sha256"] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
        else:
            model = PPO(
                "MlpPolicy", environments, seed=args.seed, device="cpu", verbose=1,
                n_steps=256, batch_size=256, n_epochs=4, learning_rate=0.0003 if pure_rl else 0.0001,
                gamma=0.995, gae_lambda=0.95, ent_coef=0.02 if pure_rl else 0.005, clip_range=0.15,
                target_kl=0.02, tensorboard_log=str(run_directory / "tensorboard"),
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            )
        model.training_method = args.training_method
        model.decision_frame_skip = 4
        manifest["ppo"] = {key: getattr(model, key) for key in ["n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "ent_coef", "target_kl"]}
        manifest["policy_architecture"] = model.policy.net_arch
        write_json(run_directory / "experiment.json", manifest)
        with SummaryWriter(str(run_directory / "tensorboard/manual")) as writer:
            record_evaluation("before", partial(policy_action, model), writer)
            if not pure_rl:
                record_evaluation("teacher", teacher_action, writer)
            if args.demo_steps:
                observations, actions = collect_demonstrations(
                    environments, args.demo_steps, run_directory, model, args.teacher_rollin_probability
                )
                imitate(model, observations, actions, args.imitation_epochs, writer)
                save_model(model, run_directory / "checkpoints/imitation.zip")
                record_evaluation("imitation", partial(policy_action, model), writer)
            if args.steps:
                callback = ExperimentLog(run_directory, checkpoint_interval=16384)
                callback.next_checkpoint = model.num_timesteps + 16384
                model.learn(args.steps, callback=callback, reset_num_timesteps=False, tb_log_name="ppo")
            save_model(model, run_directory / "checkpoints/final.zip")
            record_evaluation("final", partial(policy_action, model), writer)
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        if model is not None:
            save_model(model, run_directory / "checkpoints/interrupted.zip")
        print("Interrupted; policy/optimizer saved. Environment/rollout state restarts on resume.", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        if environments is not None:
            environments.close()
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["timesteps"] = model.num_timesteps if model else 0
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_directory / "experiment.json", manifest)
        append_json(ROOT / "experiments.jsonl", {"name": args.name, "status": manifest["status"], "timesteps": manifest["timesteps"]})


if __name__ == "__main__":
    main()
