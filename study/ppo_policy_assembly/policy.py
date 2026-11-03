"""실제 SB3 ActorCriticPolicy/Distribution API를 이용한 최소 하이브리드 정책."""
import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=8)
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.network(observations)


class SmallHybridDistribution(Distribution):
    """이동 Categorical + 회전 tanh Normal + 발사 Bernoulli."""

    def proba_distribution_net(self, latent_dim):
        # 다섯 파라미터: 이동 Logits 3, 회전 평균 1, 발사 Logit 1
        return nn.Linear(latent_dim, 5), nn.Parameter(torch.tensor([-1.2]))

    def proba_distribution(self, parameters, log_std):
        self.move = Categorical(logits=parameters[:, :3])
        scale = log_std.clamp(-4.0, 0.0).exp().expand_as(parameters[:, 3])
        self.rotation = Normal(loc=parameters[:, 3], scale=scale)
        self.fire = Bernoulli(logits=parameters[:, 4])
        return self

    def sample(self):
        return torch.stack(
            (
                self.move.sample().float(),
                self.rotation.sample().tanh().clamp(-1 + 1e-6, 1 - 1e-6),
                self.fire.sample(),
            ),
            dim=1,
        )

    def mode(self):
        return torch.stack(
            (
                self.move.probs.argmax(dim=1).float(),
                self.rotation.mean.tanh(),
                (self.fire.logits > 0).float(),
            ),
            dim=1,
        )

    def log_prob(self, actions):
        a = actions[:, 1].clamp(-1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(a)
        log_jacobian = 2 * (math.log(2) - z - nn.functional.softplus(-2 * z))
        return (
            self.move.log_prob(actions[:, 0].long())
            + self.rotation.log_prob(z) - log_jacobian
            + self.fire.log_prob(actions[:, 2])
        )

    def entropy(self):
        # tanh 변환된 정규분포의 entropy는 PPO가 -log_prob로 근사한다.
        return None

    def actions_from_params(self, parameters, log_std, deterministic=False):
        return self.proba_distribution(parameters, log_std).get_actions(deterministic)

    def log_prob_from_params(self, parameters, log_std):
        actions = self.actions_from_params(parameters, log_std)
        return actions, self.log_prob(actions)


class SmallHybridPolicy(ActorCriticPolicy):
    """SB3가 소유할 Policy: 신경망은 모두 이 객체 안에 등록된다."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", SmallFeaturesExtractor)
        kwargs.setdefault("net_arch", dict(pi=[16], vf=[16]))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        # 부모의 __init__에서 호출되는 자식 _build (학습 시마다 호출되지 않음)
        self._build_mlp_extractor()
        self.action_dist = SmallHybridDistribution()
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi
        )
        self.aim_net = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _distribution(self, observations, actor_latent):
        parameters = self.action_net(actor_latent)  # (batch, 5)
        aim = self.aim_net(observations)            # (batch, 2)
        # 회전 평균과 발사 Logit은 독립 조준망 출력으로 교체한다.
        parameters = torch.cat([parameters[:, :3], aim], dim=1)
        return self.action_dist.proba_distribution(parameters, self.log_std)

    def forward(self, obs, deterministic=False):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(critic_latent), distribution.log_prob(actions)

    def get_distribution(self, obs):
        actor_latent = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self._distribution(obs, actor_latent)

    def evaluate_actions(self, obs, actions):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor_latent)
        return self.value_net(critic_latent), distribution.log_prob(actions), distribution.entropy()
