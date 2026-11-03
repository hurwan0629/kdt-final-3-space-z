"""원본 벡터에서 개체·지형 특징을 학습하는 PPO 정책."""
from functools import partial
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from space_z.rl.gpt_training_rotation_continuous.policy import RotationHybridDistribution


def entity_encoder(input_size):
    return nn.Sequential(nn.Linear(input_size, 64), nn.GELU(), nn.Linear(64, 64), nn.LayerNorm(64))


class VectorFeaturesExtractor(BaseFeaturesExtractor):
    """5139개 관찰만 받아 개체와 지형을 플레이어 질의로 통합한다."""

    def __init__(self, observation_space):
        if observation_space.shape != (5139,):
            raise ValueError('Vector-only policy requires a (5139,) observation')
        super().__init__(observation_space, features_dim=64)
        self.player_encoder = entity_encoder(133)
        self.enemy_encoder = entity_encoder(31)
        self.shape_encoder = entity_encoder(10)
        self.bullet_encoder = entity_encoder(8)
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.terrain_projection = nn.Linear(16, 64)
        self.terrain_position_projection = nn.Linear(2, 64)
        self.type_embedding = nn.Embedding(4, 64)
        rows, columns = torch.meshgrid(torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 8), indexing='ij')
        self.register_buffer('terrain_positions', torch.stack([rows, columns], dim=-1).reshape(40, 2))
        self.query_normalization = nn.LayerNorm(64)
        self.context_normalization = nn.LayerNorm(64)
        self.cross_attention = nn.MultiheadAttention(64, 4, dropout=0, batch_first=True)
        self.feedforward = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def encode(self, observation, return_attention=False):
        player = self.player_encoder(observation[:, :133]).unsqueeze(1)
        entities = [observation[:, 133:412].reshape(-1, 9, 31),
                    observation[:, 412:712].reshape(-1, 30, 10),
                    observation[:, 712:1224].reshape(-1, 64, 8)]
        tokens = [encoder(values) + self.type_embedding.weight[index]
                  for index, (encoder, values) 
                  in enumerate(zip(
                      [self.enemy_encoder, self.shape_encoder, self.bullet_encoder],
                        entities
                  ))]
        terrain = self.terrain_cnn(observation[:, 1224:].reshape(-1, 3, 29, 45))
        terrain = terrain.flatten(2).transpose(1, 2)
        terrain = (self.terrain_projection(terrain) + self.terrain_position_projection(self.terrain_positions)
                   + self.type_embedding.weight[3])
        tokens.append(terrain)
        context = self.context_normalization(torch.cat(tokens, dim=1))
        empty = [values[:, :, 0] < 0.5 for values in entities]
        empty.append(torch.zeros((observation.shape[0], 40), dtype=torch.bool, device=observation.device))
        attended, weights = self.cross_attention(
            self.query_normalization(player), context, context,
            key_padding_mask=torch.cat(empty, dim=1),
            need_weights=return_attention, average_attn_weights=False,
        )
        combined = player + attended
        features = (combined + self.feedforward(combined)).squeeze(1)
        return (features, weights) if return_attention else features

    def forward(self, observations):
        return self.encode(observations)


class VectorOnlyPolicy(ActorCriticPolicy):
    """목표 선택 전처리 없이 관찰에서 혼합 행동과 가치를 출력한다."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', VectorFeaturesExtractor)
        kwargs.setdefault('net_arch', dict(pi=[128], vf=[128]))
        kwargs.setdefault('activation_fn', nn.GELU)
        super().__init__(*args, **kwargs)
        if not self.share_features_extractor:
            raise ValueError('This policy requires a shared features extractor')

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = RotationHybridDistribution()
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        for module, gain in [(self.features_extractor, np.sqrt(2)), (self.mlp_extractor, np.sqrt(2)),
                             (self.action_net, 0.01), (self.value_net, 1)]:
            module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def distribution(self, observation, actor_features):
        parameters = self.action_net(actor_features)
        valid = torch.ones_like(parameters, dtype=torch.bool)
        valid[:, 12:17] = observation[:, 40:45] > 0.5
        valid[:, 18:22] = observation[:, [45, 67, 89, 111]] > 0.5
        return self.action_dist.proba_distribution(parameters.masked_fill(~valid, -1e9), self.log_std)

    def forward(self, observation, deterministic=False):
        actor, critic = self.mlp_extractor(self.extract_features(observation))
        distribution = self.distribution(observation, actor)
        actions = distribution.get_actions(deterministic)
        return actions, self.value_net(critic), distribution.log_prob(actions)

    def get_distribution(self, observation):
        actor = self.mlp_extractor.forward_actor(self.extract_features(observation))
        return self.distribution(observation, actor)

    def evaluate_actions(self, observation, actions):
        actor, critic = self.mlp_extractor(self.extract_features(observation))
        distribution = self.distribution(observation, actor)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()
