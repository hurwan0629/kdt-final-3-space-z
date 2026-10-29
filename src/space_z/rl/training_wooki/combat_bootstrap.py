"""Observation-only combat demonstrations and supervised PPO warm start.

Uses the main game's sensors/actions/rewards. No privileged world queries,
automatic aiming at inference, or reward modifications are applied.
"""
import argparse
from pathlib import Path
import json
import math

import numpy as np
import torch

from space_z.config import ObsConfig, GameConfig
from space_z.game.actions import MOVE_VECTORS
from space_z.rl.observations.vector_observation import VectorObservation
from .runtime import (make_vec_env, normalizer, evaluate,
                      publish_checkpoint, atomic_json, load_checkpoint, read_json)
from .settings import TrainSettings
from .policy import CombatFeatures, TacticalCombatFeatures
from stable_baselines3 import PPO

_SENSORS = ObsConfig()
_SLICES = VectorObservation(_SENSORS).slices
_GAME = GameConfig()


def demonstration_action(observation, tick=0):
    """Aim at visible sensor slots, keep distance and steer away from walls."""
    cfg = _SENSORS
    slices = _SLICES
    v = observation['vector'][-cfg.vector_size:]
    enemies = v[slices['enemies']].reshape(cfg.n_enemies, cfg.ENEMY_DIM)
    shapes = v[slices['shapes']].reshape(cfg.n_shapes, cfg.SHAPE_DIM)
    targets = enemies[enemies[:, 0] > .5]
    enemy = len(targets) > 0
    if not enemy:
        targets = shapes[shapes[:, 0] > .5]
    direction = np.array([math.cos(tick / 90), math.sin(tick / 90)])
    rotation, fire = 2, 0
    if len(targets):
        t = targets[0]
        dx, dy = t[1] * cfg.grid_width * 24, t[2] * cfg.grid_height * 24
        distance = math.hypot(dx, dy)
        # Lead moving targets using their observed velocity.
        dx += t[3] * 4 * distance / 12
        dy += t[4] * 4 * distance / 12
        angle = math.atan2(dy, dx) - math.atan2(v[3], v[4])
        angle = math.atan2(math.sin(angle), math.cos(angle))
        rotation = 0 if abs(angle) < .10 else (2 if angle > 0 else 1)
        fire = int(abs(angle) < .22)
        toward = np.array([dx, dy]) / max(1., math.hypot(dx, dy))
        radial = np.clip((distance - (370 if enemy else 230)) / 200, -1, 1)
        direction = radial * toward
        if enemy:
            direction += .8 * np.array([-toward[1], toward[0]])
    grid = observation['grid'][-cfg.grid_shape()[0]]
    cy, cx = grid.shape[0] // 2, grid.shape[1] // 2
    scores = []
    for mx, my in MOVE_VECTORS:
        penalty = 0.
        for reach in (1, 2, 3):
            r, c = cy + round(my * reach), cx + round(mx * reach)
            penalty += float(grid[r, c]) * (4 - reach)
        scores.append(mx * direction[0] + my * direction[1] - penalty * 3)
    return (int(np.argmax(scores)) * 3 + rotation) * 2 + fire


def evasive_action(observation, tick=0):
    """Kite groups and avoid predicted projectile paths using main sensors only."""
    original = demonstration_action(observation, tick)
    cfg = _SENSORS
    v = observation['vector'][-cfg.vector_size:]
    enemies = v[_SLICES['enemies']].reshape(cfg.n_enemies, cfg.ENEMY_DIM)
    enemies = enemies[enemies[:, 0] > .5]
    bullets = v[_SLICES['bullets']].reshape(cfg.n_bullets, cfg.BULLET_DIM)
    bullets = bullets[bullets[:, 0] > .5][:16]
    scale = np.array([cfg.grid_width, cfg.grid_height]) * _GAME.terrain.tile_size / 2
    positions = enemies[:, 1:3] * scale
    distances = np.linalg.norm(positions, axis=1)
    direction = np.array(MOVE_VECTORS[original//6]) * .45
    crowded = int(np.sum(distances < 480)) >= 2
    if len(enemies):
        away = -positions / np.maximum(distances[:, None], 1)
        pressure = np.clip((650-distances)/350, 0, 2)
        direction += (away * pressure[:, None]).sum(0) * (2 if crowded else 1)
        if distances[0] > 600 and not crowded:
            direction -= away[0]
    direction *= min(1., 1.5/max(float(np.linalg.norm(direction)), .001))
    grid = observation['grid'][-cfg.grid_shape()[0]]
    cy, cx = grid.shape[0]//2, grid.shape[1]//2
    scores = []
    for mx, my in MOVE_VECTORS:
        movement = np.array([mx, my])
        score = float(movement @ direction)
        for reach in (1, 2, 3, 4):
            score -= float(grid[cy+round(my*reach), cx+round(mx*reach)]) * (5-reach)*2
        if len(bullets):
            pos = bullets[:, 1:3] * scale
            velocity = bullets[:, 3:5] * _GAME.bullet_speed - movement * _GAME.max_speed
            approach = np.clip(-np.sum(pos*velocity, axis=1) /
                               np.maximum(np.sum(velocity*velocity,axis=1), 1), 0, 24)
            separation = np.linalg.norm(pos+velocity*approach[:, None],axis=1)
            risk = np.exp(-np.square(separation/85)) * (1-approach/32)
            score -= float(risk.sum()) * 6
        scores.append(score)
    return int(np.argmax(scores))*6 + original%6


def bootstrap(directory, samples=4096, epochs=12):
    directory = Path(directory)
    cfg = TrainSettings(workers=1, torch_threads=2, curriculum=False, learning_rate=.0001)
    torch.set_num_threads(cfg.torch_threads)
    env = make_vec_env(cfg, evaluation=True)
    normalizer(env).norm_obs = False
    model = PPO('MultiInputPolicy', env, n_steps=512, batch_size=256,
                learning_rate=.0001, ent_coef=.003, n_epochs=4,
                gamma=cfg.gamma, seed=cfg.seed, device='cpu',
                policy_kwargs=dict(features_extractor_class=CombatFeatures,
                                   net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                   normalize_images=False))
    obs = env.reset()
    inputs = {k: [] for k in obs}
    labels = []
    rng = np.random.default_rng(cfg.seed)
    for tick in range(samples):
        raw = env.get_original_obs()
        action = demonstration_action({k: x[0] for k, x in raw.items()}, tick)
        for key in raw:
            inputs[key].append(raw[key][0].astype(np.float16))
        labels.append(action)
        # Visit off-demonstration states too, to reduce imitation drift.
        executed = int(rng.integers(54)) if rng.random() < .15 else action
        obs, _, _, _ = env.step([executed])
        if tick % 240 == 239:
            env.reset()
    data = {k: np.stack(v) for k, v in inputs.items()}
    labels = np.asarray(labels)
    norm = normalizer(env)
    norm.training = False
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.0003)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        losses = []
        for indices in np.array_split(rng.permutation(samples), max(1, samples // 128)):
            batch = norm.normalize_obs({k: v[indices].astype(np.float32) for k, v in data.items()})
            tensors, _ = model.policy.obs_to_tensor(batch)
            target = torch.as_tensor(labels[indices], device=model.device)
            distribution = model.policy.get_distribution(tensors)
            loss = -distribution.log_prob(target).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            optimizer.step()
            losses.append(loss.item())
        print(f'imitation epoch {epoch+1}: loss={np.mean(losses):.3f}', flush=True)
    result = evaluate(model, env, cfg)
    atomic_json(directory/'bootstrap-evaluation.json', result)
    atomic_json(directory/'config.json', cfg.to_dict())
    publish_checkpoint(model, env, directory, cfg, best=True, evaluation=result)
    atomic_json(directory/'bootstrap.json', {
        'method': 'observation-only behavioral cloning before PPO',
        'samples': samples, 'epochs': epochs, 'evaluation': result,
    })
    env.close()
    print(json.dumps(result), flush=True)
    return result


def refine(source, directory, rounds=3, samples=4096, epochs=60, tactical=False):
    """DAgger: label policy-visited states, then learn accumulated corrections.

    Cache only deterministic analytic features, not the large image history.
    The source run and its best checkpoint remain untouched.
    """
    directory, source = Path(directory), Path(source)
    entry = read_json(source/'best.json') or read_json(source/'latest.json')
    cfg = TrainSettings(**entry['settings'])
    cfg.workers, cfg.curriculum, cfg.total_steps = 1, False, 2_000_000
    cfg.evaluation_seed = 930001
    if tactical:
        cfg.evaluation_seed, cfg.eval_episodes = 960001, 6
    torch.set_num_threads(2)
    model, env = load_checkpoint(source, entry, cfg=cfg, training=False, evaluation=True)
    if not isinstance(model.policy.features_extractor, CombatFeatures):
        env.close()
        raise ValueError('Refinement requires the observation-only combat feature policy')
    if tactical:
        previous = model
        model = PPO('MultiInputPolicy', env, n_steps=512, batch_size=256,
                    learning_rate=.00005, ent_coef=.002, n_epochs=4,
                    gamma=cfg.gamma, seed=cfg.seed, device='cpu',
                    policy_kwargs=dict(features_extractor_class=TacticalCombatFeatures,
                                       net_arch=dict(pi=[128,128],vf=[128,128]),
                                       normalize_images=False))
        state = model.policy.state_dict()
        for name, old in previous.policy.state_dict().items():
            if state[name].shape == old.shape:
                state[name] = old
            elif name.endswith('.0.weight') and old.ndim == 2:
                state[name].zero_()
                state[name][:, :old.shape[1]] = old
            else:
                raise ValueError(f'Unsupported policy transfer: {name}')
        model.policy.load_state_dict(state)
        model.num_timesteps = previous.num_timesteps
    rng = np.random.default_rng(20261029)
    env.seed(73157)
    obs = env.reset()
    features, labels, weights = [], [], []
    evaluations = []
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.0001)
    best_score = -float('inf')
    try:
        for iteration in range(rounds):
            for tick in range(samples):
                raw = env.get_original_obs()
                expert = (evasive_action if tactical else demonstration_action)(
                    {k: x[0] for k, x in raw.items()}, tick)
                with torch.no_grad():
                    tensors, _ = model.policy.obs_to_tensor(obs)
                    features.append(model.policy.extract_features(tensors)[0].cpu().numpy())
                labels.append(expert)
                raw_v=raw['vector'][0][-_SENSORS.vector_size:]
                es=raw_v[_SLICES['enemies']].reshape(_SENSORS.n_enemies,_SENSORS.ENEMY_DIM)
                xy=es[:,1:3]*np.array([_SENSORS.grid_width,_SENSORS.grid_height])*24
                pressured=np.sum((es[:,0]>.5)&(np.linalg.norm(xy,axis=1)<480))>=2
                weights.append(3. if tactical and pressured else 1.)
                predicted = int(model.predict(obs, deterministic=True)[0][0])
                action = expert if rng.random() < .5 / (iteration + 1) else predicted
                if rng.random() < .05:
                    action = int(rng.integers(54))
                obs, _, _, _ = env.step([action])
                if tick % 600 == 599:
                    obs = env.reset()
            x = torch.as_tensor(np.asarray(features), device=model.device)
            y = torch.as_tensor(np.asarray(labels), device=model.device)
            w = torch.as_tensor(np.asarray(weights),dtype=torch.float32,device=model.device)
            model.policy.set_training_mode(True)
            for epoch in range(epochs):
                for indices in np.array_split(rng.permutation(len(labels)), max(1, len(labels)//256)):
                    latent = model.policy.mlp_extractor.forward_actor(x[indices])
                    distribution = model.policy.action_dist.proba_distribution(
                        action_logits=model.policy.action_net(latent))
                    loss = -(distribution.log_prob(y[indices])*w[indices]).sum()/w[indices].sum()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
                    optimizer.step()
            result = evaluate(model, env, cfg)
            evaluations.append(result)
            improved = result['mean'] > best_score
            if improved:
                best_score = result['mean']
            atomic_json(directory/'config.json', cfg.to_dict())
            # Demonstrations are separate from PPO environment steps.
            publish_checkpoint(model, env, directory, cfg, best=improved, evaluation=result)
            atomic_json(directory/'refinement.json', {
                'method': 'Threat-aware DAgger' if tactical else 'DAgger on policy-visited states', 'source': str(source),
                'demonstrations': len(labels), 'rounds': iteration+1,
                'epochs_per_round': epochs, 'validation': evaluations,
            })
            print('REFINEMENT', iteration+1, json.dumps({k: result[k] for k in
                  ('mean', 'kills', 'seconds')}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--refine-from')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--tactical', action='store_true')
    args = parser.parse_args()
    if args.refine_from:
        refine(args.refine_from, args.run_dir, args.rounds, args.samples, args.epochs, args.tactical)
    else:
        bootstrap(args.run_dir, args.samples, args.epochs)
