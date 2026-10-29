"""Factorized growth decisions mapped to main's unchanged action decoder."""
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Discrete
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from space_z.config import ObsConfig
from space_z.config.progression_definitions import SHIPS,STAT_UPGRADES
from space_z.rl.observations.vector_observation import VectorObservation
from .policy import TacticalCombatFeatures,RecoveryPolicy,PathfindingPolicy

STAT_COUNT=len(STAT_UPGRADES)+1
EVOLUTION_COUNT=ObsConfig.EVOLUTION_SLOTS+1


def encode_growth_action(action):
    value=np.asarray(action)
    if value.shape!=(3,) or not np.issubdtype(value.dtype,np.integer):
        raise ValueError('Growth action must contain three integer choices')
    combat,stat,evolution=(int(x) for x in value)
    if not (0<=combat<54 and 0<=stat<STAT_COUNT and 0<=evolution<EVOLUTION_COUNT):
        raise ValueError('Growth action choice outside main action space')
    return (combat*STAT_COUNT+stat)*EVOLUTION_COUNT+evolution


def legal_growth_masks(vector):
    """Read main progression flags; never query or mutate game state."""
    cfg=ObsConfig();v=vector[:,-cfg.vector_size:]
    start=cfg.SELF_DIM+len(SHIPS)+4+len(STAT_UPGRADES)
    available=v[:,start:start+len(STAT_UPGRADES)]>.5
    stat=torch.cat((torch.ones_like(available[:,:1]),available),dim=1)
    slot_start=start+len(STAT_UPGRADES)
    choices=torch.stack([v[:,slot_start+i*(1+len(SHIPS))]>.5
                         for i in range(cfg.EVOLUTION_SLOTS)],dim=1)
    evolution=torch.cat((~choices.any(1,keepdim=True),choices),dim=1)
    return stat,evolution


class GrowthFeatures(TacticalCombatFeatures):
    def __init__(self,observation_space):
        super().__init__(observation_space)
        self._features_dim=115+ObsConfig.PROGRESSION_DIM

    def forward(self,obs):
        base=super().forward(obs);cfg=ObsConfig()
        v=obs['vector'][:,-cfg.vector_size:]
        return torch.cat((base,v[:,cfg.SELF_DIM:cfg.SELF_DIM+cfg.PROGRESSION_DIM]),dim=1)


class FullGrowthPolicy(MultiInputActorCriticPolicy):
    """Frozen trained combat prior plus trainable combat/stat/evolution heads."""
    def __init__(self,*args,combat_recovery=True,**kwargs):
        self.combat_recovery=combat_recovery
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        combat_class=RecoveryPolicy if self.combat_recovery else PathfindingPolicy
        self.combat=combat_class(self.observation_space,Discrete(54),lr_schedule,
                                features_extractor_class=TacticalCombatFeatures,
                                net_arch=dict(pi=[128,128],vf=[128,128]),normalize_images=False)
        self.combat.requires_grad_(False)
        # Preserve combat at transfer; growth heads start without arbitrary bias.
        nn.init.zeros_(self.action_net.weight);nn.init.zeros_(self.action_net.bias)

    def _distribution_and_value(self,obs):
        features=self.extract_features(obs)
        pi,vf=self.mlp_extractor(features)
        output=self.action_net(pi)
        with torch.no_grad():
            prior=self.combat.get_distribution(obs).distribution.logits
        combat=prior+output[:,:54]
        # Safety/recovery restrictions cannot be undone by a residual logit.
        combat=combat.masked_fill(prior<-1000,-1e4)
        stat=output[:,54:54+STAT_COUNT]
        evolution=output[:,54+STAT_COUNT:]
        stat_mask,evolution_mask=legal_growth_masks(obs['vector'])
        # Prefer spending available points initially; waiting remains learnable.
        wait_penalty=torch.zeros_like(stat)
        wait_penalty[:,0]=stat_mask[:,1:].any(1)*2.
        stat=(stat-wait_penalty).masked_fill(~stat_mask,-1e4)
        evolution=evolution.masked_fill(~evolution_mask,-1e4)
        distribution=self.action_dist.proba_distribution(action_logits=torch.cat((combat,stat,evolution),dim=1))
        return distribution,self.value_net(vf)

    def get_distribution(self,obs):
        return self._distribution_and_value(obs)[0]

    def forward(self,obs,deterministic=False):
        distribution,values=self._distribution_and_value(obs)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions,values,distribution.log_prob(actions)

    def evaluate_actions(self,obs,actions):
        distribution,values=self._distribution_and_value(obs)
        return values,distribution.log_prob(actions),distribution.entropy()


def migrate_combat(source,directory):
    """Start a new growth run with the best combat prior, preserving its source."""
    from pathlib import Path
    from dataclasses import replace
    from stable_baselines3 import PPO
    from .runtime import read_json,load_checkpoint,make_vec_env,make_model,publish_checkpoint,atomic_json
    from .settings import TrainSettings
    source,directory=Path(source),Path(directory)
    entry=read_json(source/'best.json') or read_json(source/'latest.json')
    if not entry:
        raise ValueError('No source combat checkpoint')
    old_cfg=replace(TrainSettings(**entry['settings']),workers=1)
    if old_cfg.full_growth:
        raise ValueError('Use --resume for an existing full-growth model')
    if not old_cfg.corner_recovery:
        raise ValueError('Transfer expects the latest corner-recovery combat model')
    if (directory/'latest.json').exists():
        raise ValueError('Destination already contains a checkpoint; use a new run directory')
    old,old_env=load_checkpoint(source,entry,cfg=old_cfg,evaluation=True)
    cfg=replace(old_cfg,full_growth=True,corner_recovery=True,curriculum=False,
                objective='score100',workers=1,max_steps=9000,eval_steps=1500,
                eval_episodes=6,evaluation_seed=1100001,gamma=.9995,learning_rate=.0001)
    env=make_vec_env(cfg,evaluation=True)
    try:
        model=make_model(cfg,env)
        model.policy.combat.load_state_dict(old.policy.state_dict())
        model.ent_coef=.003
        model.gae_lambda=.97;model.rollout_buffer.gae_lambda=.97
        obs=env.reset()
        before=old.predict(obs,deterministic=True)[0]
        after=model.predict(obs,deterministic=True)[0]
        if not np.array_equal(before,after[:,0]):
            raise AssertionError('Combat prior changed during growth transfer')
        atomic_json(directory/'config.json',cfg.to_dict())
        publish_checkpoint(model,env,directory,cfg)
        atomic_json(directory/'transfer.json',{
            'source_run':str(source.resolve()),'source_checkpoint':entry['directory'],
            'inherited_combat_steps':entry['step'],'growth_steps':0,
            'combat_action_preserved':True,'main_action_combinations':54*STAT_COUNT*EVOLUTION_COUNT,
            'training_horizon_seconds':600,'standard_evaluation_seconds':100,
            'starts_at_level':1,'fixture_xp_in_training':False,
        })
        return cfg
    finally:
        env.close();old_env.close()
