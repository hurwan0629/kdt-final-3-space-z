"""기존 5,139개 관찰을 개체별 신경망과 지형 CNN으로 처리한다."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EntityEncoder(nn.Module):
    """동일 종류 개체를 공유 가중치로 인코딩하고 주의집중·최댓값으로 모은다."""

    def __init__(self, entity_size, embedding_size):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(entity_size + 6, 32), nn.Tanh(), nn.Linear(32, embedding_size), nn.Tanh())
        self.attention = nn.Linear(embedding_size, 1)

    def forward(self, entities, player_direction):
        present = entities[:, :, :1] > 0.5
        x = entities[:, :, 1:2]
        y = entities[:, :, 2:3] * (696 / 1080)
        sine, cosine = player_direction[:, None, :1], player_direction[:, None, 1:2]
        forward = x * cosine + y * sine
        sideways = y * cosine - x * sine
        distance = (x.square() + y.square() + 1e-8).sqrt()
        geometry = torch.cat([forward, sideways, distance, forward / distance, sideways / distance, 1 / (1 + distance)], dim=-1)
        encoded = self.embedding(torch.cat([entities, geometry], dim=-1)) * present
        scores = self.attention(encoded).masked_fill(~present, -1e4)
        attention = scores.softmax(dim=1) * present
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = (encoded * attention).sum(dim=1)
        maximum = encoded.masked_fill(~present, -1e4).amax(dim=1)
        maximum = torch.where(present.any(dim=1), maximum, torch.zeros_like(maximum))
        return torch.cat([encoded[:, 0], attended, maximum], dim=-1)


class SpatialFeaturesExtractor(BaseFeaturesExtractor):
    """자기·성장, 적, 천체, 탄환, 지형을 분리해 256차원으로 합친다."""

    def __init__(self, observation_space):
        if observation_space.shape != (5139,):
            raise ValueError('SpatialFeaturesExtractor requires the original 5139-dimensional observation')
        super().__init__(observation_space, features_dim=256)
        self.player = nn.Sequential(nn.Linear(133, 64), nn.Tanh())
        self.enemies = EntityEncoder(31, 32)
        self.shapes = EntityEncoder(10, 32)
        self.bullets = EntityEncoder(8, 16)
        self.terrain = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(240, 32), nn.Tanh(),
        )
        self.combined = nn.Sequential(nn.Linear(64 + 96 + 96 + 48 + 32, 256), nn.Tanh())

    def forward(self, observation):
        direction = observation[:, 3:5]
        features = [
            self.player(observation[:, :133]),
            self.enemies(observation[:, 133:412].reshape(-1, 9, 31), direction),
            self.shapes(observation[:, 412:712].reshape(-1, 30, 10), direction),
            self.bullets(observation[:, 712:1224].reshape(-1, 64, 8), direction),
            self.terrain(observation[:, 1224:].reshape(-1, 3, 29, 45)),
        ]
        return self.combined(torch.cat(features, dim=-1))


class GrowthMaskedPolicy(ActorCriticPolicy):
    """현재 불가능한 성장 요청만 제외한다. 가능한 행동의 우선순위는 PPO가 학습한다."""

    def __init__(self, *args, aim_residual=False, freeze_base_actor=False, unified_aim_input=False,
                 independent_aim=False, freeze_aim=False, ignore_aim_health=False, **kwargs):
        self.use_aim_residual = aim_residual or independent_aim
        self.base_actor_frozen = freeze_base_actor
        self.unified_aim_input = unified_aim_input or independent_aim
        self.independent_aim = independent_aim
        self.aim_frozen = freeze_aim
        self.ignore_aim_health = ignore_aim_health
        super().__init__(*args, **kwargs)
        if not np.array_equal(self.action_space.nvec, [9, 9, 2, 6, 5]) or not self.share_features_extractor:
            raise ValueError('GrowthMaskedPolicy requires the shared extractor and [9,9,2,6,5] actions')

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.use_aim_residual:
            self.enable_aim_residual()
        self.set_actor_frozen(self.base_actor_frozen)
        self.set_aim_frozen(self.aim_frozen)

    def set_aim_frozen(self, frozen):
        self.aim_frozen = frozen
        if hasattr(self, 'aim_residual'):
            for parameter in self.aim_residual.parameters():
                parameter.requires_grad_(not frozen)
        elif frozen:
            raise ValueError('Cannot freeze an absent aim network')

    def enable_independent_aim(self):
        """장면 신경망과 분리한 조준 head를 중립 출력에서 다시 PPO로 학습한다."""
        if self.independent_aim:
            return
        self.enable_aim_residual()
        self.independent_aim = self.unified_aim_input = True
        nn.init.zeros_(self.aim_residual[-1].weight)
        nn.init.zeros_(self.aim_residual[-1].bias)
        for parameter in self.aim_residual.parameters():
            self.optimizer.state.pop(parameter, None)

    def set_actor_frozen(self, frozen):
        """사격 연습에서는 기존 제어·인코더를 보존하고 추가 조준 신경망만 학습한다."""
        self.base_actor_frozen = frozen
        for module in [self.features_extractor, self.mlp_extractor.policy_net, self.action_net]:
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)

    def enable_aim_residual(self):
        """기존 행동을 유지하는 0출력 신경망을 추가하고 PPO optimizer에 등록한다."""
        if hasattr(self, 'aim_residual'):
            return
        self.use_aim_residual = True
        self.aim_residual = nn.Sequential(
            nn.Linear(187 + self.mlp_extractor.latent_dim_pi, 64), nn.Tanh(), nn.Linear(64, 11),
        ).to(self.device)
        nn.init.zeros_(self.aim_residual[-1].weight)
        nn.init.zeros_(self.aim_residual[-1].bias)
        self.optimizer.add_param_group({'params': self.aim_residual.parameters()})

    def _aim_geometry(self, observation):
        sine, cosine = observation[:, None, 3:4], observation[:, None, 4:5]
        groups = [observation[:, 133:412].reshape(-1, 9, 31),
                  observation[:, 412:712].reshape(-1, 30, 10)]
        context = observation[:, :133]
        if self.independent_aim:
            context = torch.zeros_like(context)
            context[:, 5:7] = observation[:, 5:7]
            context[:, 10:31] = observation[:, 10:31]
            context[:, 35:40] = observation[:, 35:40]
            if self.ignore_aim_health:
                context[:, 35] = 0
        features = [context]
        all_geometry = []
        for entities, radius_index in zip(groups, [8, 6]):
            if not self.unified_aim_input:
                entities = entities[:, :3]
            x, y = entities[:, :, 1:2], entities[:, :, 2:3] * (696/1080)
            forward, sideways = x*cosine + y*sine, y*cosine - x*sine
            distance = (x.square()+y.square()+1e-8).sqrt()
            bearing = torch.atan2(sideways, forward) / np.pi
            velocity_forward = entities[:, :, 3:4]*cosine + entities[:, :, 4:5]*sine
            velocity_sideways = entities[:, :, 4:5]*cosine - entities[:, :, 3:4]*sine
            health = torch.ones_like(entities[:, :, 5:6]) if self.ignore_aim_health else entities[:, :, 5:6]
            geometry = torch.cat([entities[:, :, :1], forward, sideways, bearing, distance,
                                  health, velocity_forward, velocity_sideways,
                                  entities[:, :, radius_index:radius_index+1]], dim=-1)
            geometry = geometry * entities[:, :, :1]
            if self.unified_aim_input:
                all_geometry.append(geometry)
            else:
                features.append(geometry.flatten(start_dim=1))
        if self.unified_aim_input:
            geometry = torch.cat(all_geometry, dim=1)
            position = torch.cat([group[:, :, 1:3] for group in groups], dim=1)
            fractions = torch.linspace(0.05, 0.95, 20, device=observation.device)
            columns = (22 + position[:, :, 0:1]*1080/48*fractions).round().long().clamp(0, 44)
            rows = (14 + position[:, :, 1:2]*696/48*fractions).round().long().clamp(0, 28)
            indices = (rows*45 + columns).flatten(start_dim=1)
            solid = observation[:, 1224:2529].gather(1, indices).reshape(-1, 39, 20)
            visible = (solid.amax(dim=-1) < 0.5) & (geometry[:, :, 0] > 0.5)
            priority = geometry[:, :, 4].clone()
            priority[:, :9] *= 0.8
            chosen = priority.masked_fill(~visible, 1e4).argmin(dim=1)
            selected = geometry[torch.arange(len(observation), device=observation.device), chosen]
            selected = selected * visible.any(dim=1, keepdim=True)
            # 같은 물리적 조준 관계를 종류에 상관없이 기존 첫 천체 입력 위치로 정렬한다.
            aligned_geometry = observation.new_zeros((len(observation), 54))
            aligned_geometry[:, 27:36] = selected
            features.append(aligned_geometry)
        return torch.cat(features, dim=-1)

    def _masked_distribution(self, observation, actor_latent):
        logits = self.action_net(actor_latent)
        if self.use_aim_residual:
            scene_context = torch.zeros_like(actor_latent) if self.independent_aim else actor_latent
            correction = self.aim_residual(torch.cat([self._aim_geometry(observation), scene_context], dim=-1))
            if self.independent_aim:
                logits = torch.cat([logits[:, :9], correction, logits[:, 20:]], dim=-1)
            else:
                logits = logits + torch.nn.functional.pad(correction, (9, 11))
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, 21:26] = observation[:, 40:45] > 0.5
        valid[:, 27:31] = observation[:, [45, 67, 89, 111]] > 0.5
        return self.action_dist.proba_distribution(action_logits=logits.masked_fill(~valid, -1e9))

    def forward(self, observation, deterministic=False):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(observation))
        distribution = self._masked_distribution(observation, actor_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(critic_latent), distribution.log_prob(actions)

    def get_distribution(self, observation):
        actor_latent = self.mlp_extractor.forward_actor(self.extract_features(observation))
        return self._masked_distribution(observation, actor_latent)

    def evaluate_actions(self, observation, actions):
        actor_latent, critic_latent = self.mlp_extractor(self.extract_features(observation))
        distribution = self._masked_distribution(observation, actor_latent)
        return self.value_net(critic_latent), distribution.log_prob(actions), distribution.entropy()
