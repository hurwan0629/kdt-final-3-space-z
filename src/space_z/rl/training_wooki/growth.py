"""Factorized growth decisions mapped to main's unchanged action decoder."""

# 성장 확장: 행동 인코딩 → 가능한 선택 마스크 → 성장 정책 → 기존 전투 모델 이식.

import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Discrete
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from space_z.config import ObsConfig
from space_z.config.progression_definitions import SHIPS, STAT_UPGRADES
from space_z.rl.observations.vector_observation import VectorObservation
from .policy import TacticalCombatFeatures, RecoveryPolicy, PathfindingPolicy

STAT_COUNT = len(STAT_UPGRADES) + 1
EVOLUTION_COUNT = ObsConfig.EVOLUTION_SLOTS + 1


def encode_growth_action(action):
    # 전투·스탯·전직 세 정수의 범위를 검사하고, 기본 decoder가 받는 정수 하나로 합친다.
    value = np.asarray(action)
    if value.shape != (3,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError('Growth action must contain three integer choices')
    combat, stat, evolution = (int(x) for x in value)
    if not (
        0 <= combat < 54 and 0 <= stat < STAT_COUNT and 0 <= evolution < EVOLUTION_COUNT
    ):
        raise ValueError('Growth action choice outside main action space')
    return (combat * STAT_COUNT + stat) * EVOLUTION_COUNT + evolution


def legal_growth_masks(vector):
    """Read main progression flags; never query or mutate game state."""
    # 최신 성장 관찰에서 스탯·전직 선택 가능 여부를 읽어 불리언 마스크를 반환한다.
    cfg = ObsConfig()
    v = vector[:, -cfg.vector_size :]
    start = cfg.SELF_DIM + len(SHIPS) + 4 + len(STAT_UPGRADES)

    # 1) 스탯별 강화 가능 표시를 읽고, 스탯 미선택 항목은 항상 허용한다.
    available = v[:, start : start + len(STAT_UPGRADES)] > .5
    stat = torch.cat((torch.ones_like(available[:, :1]), available), dim=1)
    slot_start = start + len(STAT_UPGRADES)

    # 2) 전직 선택지가 하나라도 있으면 전직 미선택 항목은 마스크로 제외한다.
    choices = torch.stack(
        [
            v[:, slot_start + i * (1 + len(SHIPS))] > .5
            for i in range(cfg.EVOLUTION_SLOTS)
        ],
        dim=1,
    )
    evolution = torch.cat((~choices.any(1, keepdim=True), choices), dim=1)
    return stat, evolution


class GrowthFeatures(TacticalCombatFeatures):
    def __init__(self, observation_space):
        # 전술 특징 115개에 기본 성장 관찰 길이를 더해 출력 크기를 설정한다.
        super().__init__(observation_space)
        self._features_dim = 115 + ObsConfig.PROGRESSION_DIM

    def forward(self, obs):
        # 전술 특징 뒤에 최신 성장 관찰 구간을 이어 붙인다.
        base = super().forward(obs)
        cfg = ObsConfig()
        v = obs['vector'][:, -cfg.vector_size :]
        return torch.cat(
            (base, v[:, cfg.SELF_DIM : cfg.SELF_DIM + cfg.PROGRESSION_DIM]), dim=1
        )


class FullGrowthPolicy(MultiInputActorCriticPolicy):
    """Frozen trained combat prior plus trainable combat/stat/evolution heads."""

    def __init__(self, *args, combat_recovery=True, **kwargs):
        # 전투 하위 정책이 모서리 탈출 기능을 사용할지 보관한다.
        self.combat_recovery = combat_recovery
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        # 고정 전투 정책을 만들고, 추가 학습 분기의 초기 행동 점수를 0으로 둔다.
        super()._build(lr_schedule)
        combat_class = RecoveryPolicy if self.combat_recovery else PathfindingPolicy
        self.combat = combat_class(
            self.observation_space,
            Discrete(54),
            lr_schedule,
            features_extractor_class=TacticalCombatFeatures,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            normalize_images=False,
        )
        self.combat.requires_grad_(False)
        # 전투 이식 직후 행동을 보존하도록 추가 학습 분기의 초기 보정값을 0으로 둔다.
        nn.init.zeros_(self.action_net.weight)
        nn.init.zeros_(self.action_net.bias)

    def _distribution_and_value(self, obs):
        # 기존 전투 점수에 학습 점수를 더하고, 가능한 성장 선택만 분포에 남긴다.
        features = self.extract_features(obs)
        pi, vf = self.mlp_extractor(features)
        output = self.action_net(pi)

        # 1) 고정된 전투 정책 점수에 새 정책이 학습하는 보정 점수를 더한다.
        with torch.no_grad():
            prior = self.combat.get_distribution(obs).distribution.logits
        combat = prior + output[:, :54]
        # 추가 학습 점수가 기존 안전·탈출 제한을 다시 허용하지 못하게 한다.
        combat = combat.masked_fill(prior < -1000, -1e4)

        # 2) 스탯·전직은 기본 관찰의 가능 여부로 제한해 각각의 분포를 만든다.
        stat = output[:, 54 : 54 + STAT_COUNT]
        evolution = output[:, 54 + STAT_COUNT :]
        stat_mask, evolution_mask = legal_growth_masks(obs['vector'])
        # 초기에 가능한 강화를 선호하게 하되, 미선택 행동도 학습할 수 있게 남긴다.
        wait_penalty = torch.zeros_like(stat)
        wait_penalty[:, 0] = stat_mask[:, 1:].any(1) * 2.
        stat = (stat - wait_penalty).masked_fill(~stat_mask, -1e4)
        evolution = evolution.masked_fill(~evolution_mask, -1e4)
        distribution = self.action_dist.proba_distribution(
            action_logits=torch.cat((combat, stat, evolution), dim=1)
        )
        return distribution, self.value_net(vf)

    def get_distribution(self, obs):
        # 전투·스탯·전직의 행동 분포를 반환한다.
        return self._distribution_and_value(obs)[0]

    def forward(self, obs, deterministic=False):
        # 세 행동을 선택하고 상태 가치 및 결합된 로그 확률을 반환한다.
        distribution, values = self._distribution_and_value(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        # 전달받은 성장 행동을 PPO가 학습할 로그 확률·엔트로피로 평가한다.
        distribution, values = self._distribution_and_value(obs)
        return values, distribution.log_prob(actions), distribution.entropy()


def migrate_combat(source, directory):
    """Start a new growth run with the best combat prior, preserving its source."""
    # 기존 전투 모델을 새 성장 모델에 이식하고, 최초 전투 행동이 보존되는지 확인한다.
    from pathlib import Path
    from dataclasses import replace
    from stable_baselines3 import PPO
    from .runtime import (
        read_json,
        load_checkpoint,
        make_vec_env,
        make_model,
        publish_checkpoint,
        atomic_json,
    )
    from .settings import TrainSettings

    source, directory = Path(source), Path(directory)

    # 1) 이식 가능한 전투 모델인지 확인하고 원본 체크포인트를 읽는다.
    entry = read_json(source / 'best.json') or read_json(source / 'latest.json')
    if not entry:
        raise ValueError('No source combat checkpoint')
    old_cfg = replace(TrainSettings(**entry['settings']), workers=1)
    if old_cfg.full_growth:
        raise ValueError('Use --resume for an existing full-growth model')
    if not old_cfg.corner_recovery:
        raise ValueError('Transfer expects the latest corner-recovery combat model')
    if (directory / 'latest.json').exists():
        raise ValueError(
            'Destination already contains a checkpoint; use a new run directory'
        )
    old, old_env = load_checkpoint(source, entry, cfg=old_cfg, evaluation=True)

    # 2) 원본 설정을 바탕으로 성장 전용 환경과 학습 조건을 구성한다.
    cfg = replace(
        old_cfg,
        full_growth=True,
        corner_recovery=True,
        curriculum=False,
        objective='score100',
        workers=1,
        max_steps=9000,
        eval_steps=1500,
        eval_episodes=6,
        evaluation_seed=1100001,
        gamma=.9995,
        learning_rate=.0001,
    )
    env = make_vec_env(cfg, evaluation=True)
    try:
        model = make_model(cfg, env)

        # 3) 전투 가중치를 복사하고 같은 관찰에서 전투 선택이 보존되는지 검사한다.
        model.policy.combat.load_state_dict(old.policy.state_dict())
        model.ent_coef = .003
        model.gae_lambda = .97
        model.rollout_buffer.gae_lambda = .97
        obs = env.reset()
        before = old.predict(obs, deterministic=True)[0]
        after = model.predict(obs, deterministic=True)[0]
        if not np.array_equal(before, after[:, 0]):
            raise AssertionError('Combat prior changed during growth transfer')

        # 4) 새 성장 실행의 설정·체크포인트·이식 출처를 저장한다.
        atomic_json(directory / 'config.json', cfg.to_dict())
        publish_checkpoint(model, env, directory, cfg)
        atomic_json(
            directory / 'transfer.json',
            {
                'source_run': str(source.resolve()),
                'source_checkpoint': entry['directory'],
                'inherited_combat_steps': entry['step'],
                'growth_steps': 0,
                'combat_action_preserved': True,
                'main_action_combinations': 54 * STAT_COUNT * EVOLUTION_COUNT,
                'training_horizon_seconds': 600,
                'standard_evaluation_seconds': 100,
                'starts_at_level': 1,
                'fixture_xp_in_training': False,
            },
        )
        return cfg
    finally:
        env.close()
        old_env.close()
