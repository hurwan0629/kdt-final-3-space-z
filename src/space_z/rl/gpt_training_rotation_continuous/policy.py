"""이산 선택과 tanh Gaussian 회전의 결합 확률을 PPO에 제공한다."""
import math
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from space_z.rl.actions.rotation_continuous_action_decoder import RotationContinuousActionDecoder
from space_z.rl.gpt_training.networks import SpatialFeaturesExtractor


class RotationHybridDistribution(Distribution):
    """독립 선택들의 log probability를 합치고 회전의 tanh 보정을 적용한다."""

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, 22), nn.Parameter(torch.full((1,), -1.2))

    def proba_distribution(self, parameters, log_std):
        self.move = Categorical(logits=parameters[:, :9])
        self.rotation = Normal(parameters[:, 9], log_std.clamp(-4, 0).exp().expand_as(parameters[:, 9]))
        self.fire = Bernoulli(logits=parameters[:, 10])
        self.stat = Categorical(logits=parameters[:, 11:17])
        self.evolution = Categorical(logits=parameters[:, 17:22])
        return self

    def log_prob(self, actions):
        rotation = actions[:, 1].clamp(-1 + 1e-6, 1 - 1e-6)
        unsquashed = torch.atanh(rotation)
        log_jacobian = 2 * (math.log(2) - unsquashed - nn.functional.softplus(-2 * unsquashed))
        return (self.move.log_prob(actions[:, 0].long())
                + self.rotation.log_prob(unsquashed) - log_jacobian
                + self.fire.log_prob(actions[:, 2])
                + self.stat.log_prob(actions[:, 3].long())
                + self.evolution.log_prob(actions[:, 4].long()))

    def entropy(self):
        # tanh Gaussian에는 닫힌 형태가 없으므로 SB3의 -log_prob 추정치를 사용한다.
        return None

    def sample(self):
        return torch.stack([
            self.move.sample().float(), self.rotation.sample().tanh().clamp(-1 + 1e-6, 1 - 1e-6),
            self.fire.sample(), self.stat.sample().float(), self.evolution.sample().float(),
        ], dim=1)

    def mode(self):
        return torch.stack([
            self.move.probs.argmax(dim=1).float(), self.rotation.mean.tanh(),
            (self.fire.logits > 0).float(), self.stat.probs.argmax(dim=1).float(),
            self.evolution.probs.argmax(dim=1).float(),
        ], dim=1)

    def actions_from_params(self, parameters, log_std, deterministic=False):
        return self.proba_distribution(parameters, log_std).get_actions(deterministic)

    def log_prob_from_params(self, parameters, log_std):
        actions = self.actions_from_params(parameters, log_std)
        return actions, self.log_prob(actions)


class RotationContinuousPolicy(ActorCriticPolicy):
    """장면망은 이동·성장, 독립 조준망은 회전·발사를 선택한다."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', SpatialFeaturesExtractor)
        kwargs.setdefault('net_arch', dict(pi=[128, 128], vf=[128, 128]))
        super().__init__(*args, **kwargs)
        if self.action_space != RotationContinuousActionDecoder().action_space or not self.share_features_extractor:
            raise ValueError('Hybrid policy requires the rotation decoder action space and shared features')

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = RotationHybridDistribution()
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.aim_net = nn.Sequential(nn.Linear(187, 64), nn.Tanh(), nn.Linear(64, 2))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        for module, gain in [(self.features_extractor, np.sqrt(2)), (self.mlp_extractor, np.sqrt(2)),
                             (self.action_net, 0.01), (self.aim_net, np.sqrt(2)), (self.value_net, 1)]:
            module.apply(partial(self.init_weights, gain=gain))
        self.aim_net[-1].apply(partial(self.init_weights, gain=0.01))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def aim_features(self, observation):
        """기존 조준망과 같은 자기 기준 좌표·가시 표적 정보를 만든다."""
        context = torch.zeros_like(observation[:, :133])
        context[:, 5:7] = observation[:, 5:7]
        context[:, 10:31] = observation[:, 10:31]
        context[:, 36:40] = observation[:, 36:40]
        sine, cosine = observation[:, None, 3:4], observation[:, None, 4:5]
        groups = [observation[:, 133:412].reshape(-1, 9, 31), observation[:, 412:712].reshape(-1, 30, 10)]
        geometries = []
        for entities, radius_index in zip(groups, [8, 6]):
            x, y = entities[:, :, 1:2], entities[:, :, 2:3] * (696 / 1080)
            forward, sideways = x * cosine + y * sine, y * cosine - x * sine
            distance = (x.square() + y.square() + 1e-8).sqrt()
            geometry = torch.cat([
                entities[:, :, :1], forward, sideways, torch.atan2(sideways, forward) / np.pi, distance,
                torch.ones_like(distance), entities[:, :, 3:4] * cosine + entities[:, :, 4:5] * sine,
                entities[:, :, 4:5] * cosine - entities[:, :, 3:4] * sine,
                entities[:, :, radius_index:radius_index + 1],
            ], dim=-1) * entities[:, :, :1]
            geometries.append(geometry)
        geometry = torch.cat(geometries, dim=1)
        position = torch.cat([group[:, :, 1:3] for group in groups], dim=1)
        fractions = torch.linspace(0.05, 0.95, 20, device=observation.device)
        columns = (22 + position[:, :, 0:1] * 1080 / 48 * fractions).round().long().clamp(0, 44)
        rows = (14 + position[:, :, 1:2] * 696 / 48 * fractions).round().long().clamp(0, 28)
        indices = (rows * 45 + columns).flatten(start_dim=1)
        solid = observation[:, 1224:2529].gather(1, indices).reshape(-1, 39, 20)
        visible = (solid.amax(dim=-1) < 0.5) & (geometry[:, :, 0] > 0.5)
        priority = geometry[:, :, 4].clone()
        priority[:, :9] *= 0.8
        chosen = priority.masked_fill(~visible, 1e4).argmin(dim=1)
        selected = geometry[torch.arange(len(observation), device=observation.device), chosen]
        aligned = observation.new_zeros((len(observation), 54))
        aligned[:, 27:36] = selected * visible.any(dim=1, keepdim=True)
        return torch.cat([context, aligned], dim=1)

    def _distribution(self, observation, actor_latent):
        parameters = self.action_net(actor_latent)
        aim = self.aim_net(self.aim_features(observation))
        parameters = torch.cat([parameters[:, :9], aim, parameters[:, 11:]], dim=1)
        valid = torch.ones_like(parameters, dtype=torch.bool)
        valid[:, 12:17] = observation[:, 40:45] > 0.5
        valid[:, 18:22] = observation[:, [45, 67, 89, 111]] > 0.5
        return self.action_dist.proba_distribution(parameters.masked_fill(~valid, -1e9), self.log_std)

    def forward(self, observation, deterministic=False):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(observation))
        distribution = self._distribution(observation, actor_latent)
        actions = distribution.get_actions(deterministic)
        return actions, self.value_net(critic_latent), distribution.log_prob(actions)

    def get_distribution(self, observation):
        actor_latent = self.mlp_extractor.forward_actor(self.extract_features(observation))
        return self._distribution(observation, actor_latent)

    def evaluate_actions(self, observation, actions):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(observation))
        distribution = self._distribution(observation, actor_latent)
        return self.value_net(critic_latent), distribution.log_prob(actions), distribution.entropy()

    def initialize_from_discrete(self, previous_policy):
        """기존 순수 PPO 가중치를 옮긴다. 회전 출력은 새로 학습한다."""
        self.features_extractor.load_state_dict(previous_policy.features_extractor.state_dict())
        self.mlp_extractor.load_state_dict(previous_policy.mlp_extractor.state_dict())
        self.value_net.load_state_dict(previous_policy.value_net.state_dict())
        with torch.no_grad():
            for name in ['weight', 'bias']:
                current, previous = getattr(self.action_net, name), getattr(previous_policy.action_net, name)
                current[:9].copy_(previous[:9])
                current[11:].copy_(previous[20:])
            self.aim_net[0].weight.copy_(previous_policy.aim_residual[0].weight[:, :187])
            self.aim_net[0].bias.copy_(previous_policy.aim_residual[0].bias)
            self.aim_net[-1].weight[0].zero_()
            self.aim_net[-1].bias[0].zero_()
            self.aim_net[-1].weight[1].copy_(previous_policy.aim_residual[-1].weight[10] - previous_policy.aim_residual[-1].weight[9])
            self.aim_net[-1].bias[1].copy_(previous_policy.aim_residual[-1].bias[10] - previous_policy.aim_residual[-1].bias[9])
