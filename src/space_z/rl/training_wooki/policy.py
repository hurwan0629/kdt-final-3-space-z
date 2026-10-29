"""Small, independent encoders for tactical vectors and the local wall map."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from space_z.config import ObsConfig
from space_z.config import GameConfig
from space_z.game.actions import MOVE_VECTORS
from space_z.rl.observations.vector_observation import VectorObservation
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class SpaceFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        vector_size = observation_space['vector'].shape[0]
        channels = observation_space['grid'].shape[0]
        self.vector = nn.Sequential(nn.Linear(vector_size, 256), nn.LayerNorm(256), nn.Tanh(),
                                    nn.Linear(256, 128), nn.Tanh())
        self.grid = nn.Sequential(nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(32*6*6, 128), nn.ReLU())

    def forward(self, observations):
        return torch.cat((self.vector(observations['vector']), self.grid(observations['grid'])), dim=1)


class CombatFeatures(BaseFeaturesExtractor):
    """Expose relative target geometry from main sensors to the learned policy.

    Inputs must retain main's [-1, 1] scale (no running observation whitening).
    Features contain no actions, world queries or hidden target information.
    """
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=32)
        cfg = ObsConfig()
        self.vector_size = cfg.vector_size
        slices = VectorObservation(cfg).slices
        self.enemy_start = slices['enemies'].start
        self.shape_start = slices['shapes'].start
        self.aspect = cfg.grid_height / cfg.grid_width

    def forward(self, observations):
        v = observations['vector'][:, -self.vector_size:]
        enemy = v[:, self.enemy_start:self.enemy_start+10]
        shape = v[:, self.shape_start:self.shape_start+10]
        target = torch.where(enemy[:, :1] > .5, enemy, shape)
        dx, dy = target[:, 1:2], target[:, 2:3] * self.aspect
        distance = torch.sqrt(dx.square()+dy.square()).clamp_min(.001)
        ux, uy = dx/distance, dy/distance
        sine, cosine = v[:, 3:4], v[:, 4:5]
        cross, dot = cosine*uy-sine*ux, cosine*ux+sine*uy
        grid = observations['grid'][:, -3:-2]
        cy, cx = grid.shape[2]//2, grid.shape[3]//2
        wall = grid[:, :, cy-2:cy+3:2, cx-2:cx+3:2].flatten(1)
        return torch.cat((v[:, :10], target[:, :1], enemy[:, :1], dx, dy,
                          distance, ux, uy, cross, dot, target[:, 3:5],
                          target[:, 5:6], v[:, 2:3], wall), dim=1)


class ThreatCombatFeatures(CombatFeatures):
    """Retain learned combat features and expose nearby threats from main sensors."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self._features_dim = 88
        cfg = ObsConfig()
        slices = VectorObservation(cfg).slices
        self.enemy_slice = slices['enemies']
        self.bullet_slice = slices['bullets']
        self.enemy_dim, self.bullet_dim = cfg.ENEMY_DIM, cfg.BULLET_DIM

    def forward(self, observations):
        base = super().forward(observations)
        v = observations['vector'][:, -self.vector_size:]
        enemies = v[:, self.enemy_slice].reshape(-1, 9, self.enemy_dim)[:, :4]
        bullets = v[:, self.bullet_slice].reshape(-1, 64, self.bullet_dim)[:, :4]
        # Presence, relative position, velocity, health/radius; padding stays zero.
        enemy_data = enemies[:, :, :6].flatten(1)
        bullet_data = bullets[:, :, :6].flatten(1)
        dx, dy = enemies[:, :, 1], enemies[:, :, 2] * self.aspect
        d = torch.sqrt(dx.square()+dy.square()).clamp_min(.001)
        weights = enemies[:, :, 0] * torch.clamp(1-d/.6, min=0)
        pressure = torch.stack((weights.sum(1), (weights*dx/d).sum(1),
                                (weights*dy/d).sum(1),
                                ((d < .42)*enemies[:, :, 0]).sum(1)/4), dim=1)
        grid = observations['grid'][:, -3:-2]
        cy, cx = grid.shape[2]//2, grid.shape[3]//2
        exits = torch.stack((grid[:, 0, cy-4, cx], grid[:, 0, cy+4, cx],
                            grid[:, 0, cy, cx-4], grid[:, 0, cy, cx+4]),dim=1)
        return torch.cat((base, enemy_data, bullet_data, pressure, exits), dim=1)


class TacticalCombatFeatures(ThreatCombatFeatures):
    """Projected collision/clearance features for all nine legal movement directions."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self._features_dim = 115
        cfg, game = ObsConfig(), GameConfig()
        self.register_buffer('moves', torch.tensor(MOVE_VECTORS,dtype=torch.float32))
        self.register_buffer('scale', torch.tensor([cfg.grid_width,cfg.grid_height],dtype=torch.float32)*game.terrain.tile_size/2)
        self.speed, self.shot_speed = game.max_speed, game.bullet_speed

    def forward(self, observations):
        base = super().forward(observations)
        v = observations['vector'][:, -self.vector_size:]
        enemies = v[:, self.enemy_slice].reshape(-1,9,self.enemy_dim)
        bullets = v[:, self.bullet_slice].reshape(-1,64,self.bullet_dim)[:, :16]
        pos = bullets[:, None, :, 1:3]*self.scale
        velocity = bullets[:, None, :, 3:5]*self.shot_speed-self.moves[None,:,None,:]*self.speed
        time = (-(pos*velocity).sum(-1)/velocity.square().sum(-1).clamp_min(1)).clamp(0,24)
        separation = (pos+velocity*time[:,:,:,None]).square().sum(-1)
        risk = (torch.exp(-separation/85**2)*(1-time/32)*bullets[:,None,:,0]).sum(-1)
        future = enemies[:,None,:,1:3]*self.scale-self.moves[None,:,None,:]*self.speed*12
        distances = future.square().sum(-1).sqrt()
        clearance = torch.where(enemies[:,None,:,0]>.5,distances,torch.full_like(distances,1080)).min(-1).values/1080
        grid = observations['grid'][:, -3]
        cy,cx=grid.shape[1]//2,grid.shape[2]//2
        walls=[]
        for mx,my in MOVE_VECTORS:
            walls.append(torch.stack([grid[:,cy+round(my*r),cx+round(mx*r)] for r in (1,2,3,4)],dim=1).max(1).values)
        return torch.cat((base,risk,clearance,torch.stack(walls,dim=1)),dim=1)


class KitingPolicy(MultiInputActorCriticPolicy):
    """Preserve a frozen attack policy; learn defensive movement in close groups.

    Outside group pressure, actions exactly follow the preserved policy. In
    groups, movement uses the learned evasion branch and aim/fire use the
    preserved attack branch. The router reads only main observation slots.
    """
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.base_features = CombatFeatures(self.observation_space)
        self.base_actor = nn.Sequential(nn.Linear(32,128),nn.Tanh(),
                                        nn.Linear(128,128),nn.Tanh(),nn.Linear(128,54))
        self.base_actor.requires_grad_(False)

    def _distribution_and_value(self, obs):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        evasion = self.action_net(latent_pi).reshape(-1,9,6)
        base = self.base_actor(self.base_features(obs)).reshape(-1,9,6)
        movement = evasion.logsumexp(dim=2)
        aim_fire = base.logsumexp(dim=1)
        combined = movement[:,:,None]+aim_fire[:,None,:]
        cfg=ObsConfig()
        v=obs['vector'][:,-cfg.vector_size:]
        start=VectorObservation(cfg).slices['enemies'].start
        enemies=v[:,start:start+cfg.n_enemies*cfg.ENEMY_DIM].reshape(-1,cfg.n_enemies,cfg.ENEMY_DIM)
        distance=(enemies[:,:,1].square()+(enemies[:,:,2]*cfg.grid_height/cfg.grid_width).square()).sqrt()
        crowded=((distance < 480/(cfg.grid_width*24)) & (enemies[:,:,0]>.5)).sum(1)>=2
        logits=torch.where(crowded[:,None,None],combined,base).flatten(1)
        return self.action_dist.proba_distribution(action_logits=logits), self.value_net(latent_vf)

    def get_distribution(self, obs):
        return self._distribution_and_value(obs)[0]

    def forward(self, obs, deterministic=False):
        distribution, values=self._distribution_and_value(obs)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1,*self.action_space.shape)), values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution, values=self._distribution_and_value(obs)
        return values, distribution.log_prob(actions), distribution.entropy()


class CrowdPolicy(KitingPolicy):
    """Route whole learned actions, preserving movement/aim coordination."""
    def _distribution_and_value(self, obs):
        features=self.extract_features(obs)
        latent_pi,latent_vf=self.mlp_extractor(features)
        evasion=self.action_net(latent_pi)
        base=self.base_actor(self.base_features(obs))
        cfg=ObsConfig()
        v=obs['vector'][:,-cfg.vector_size:]
        start=self.base_features.enemy_start
        enemies=v[:,start:start+cfg.n_enemies*cfg.ENEMY_DIM].reshape(-1,cfg.n_enemies,cfg.ENEMY_DIM)
        distance=(enemies[:,:,1].square()+(enemies[:,:,2]*cfg.grid_height/cfg.grid_width).square()).sqrt()
        crowded=((distance<480/(cfg.grid_width*24))&(enemies[:,:,0]>.5)).sum(1)>=2
        logits=torch.where(crowded[:,None],evasion,base)
        return self.action_dist.proba_distribution(action_logits=logits),self.value_net(latent_vf)


class GuardedPolicy(CrowdPolicy):
    """Learned combat policy with a transparent predictive movement safety filter.

    Only observed group pressure or imminent projectile risk activates the
    filter. It excludes dangerous movement directions; the preserved learned
    policy still chooses the complete legal move/aim/fire action.
    """
    def _distribution_and_value(self, obs):
        features=self.extract_features(obs)
        latent_pi,latent_vf=self.mlp_extractor(features)
        base=self.base_actor(self.base_features(obs))+self.action_net(latent_pi)
        risk,clearance,walls=features[:,88:97],features[:,97:106],features[:,106:115]
        wall_free=walls<.5
        safest=risk.masked_fill(~wall_free,1e6).min(1).values
        safe=wall_free & (risk<=safest[:,None]+.12)
        farthest=clearance.masked_fill(~safe,-1e6).max(1).values
        retreat=safe & (clearance>=farthest[:,None]-.02)
        cfg=ObsConfig();v=obs['vector'][:,-cfg.vector_size:]
        start=self.base_features.enemy_start
        enemies=v[:,start:start+cfg.n_enemies*cfg.ENEMY_DIM].reshape(-1,cfg.n_enemies,cfg.ENEMY_DIM)
        distance=(enemies[:,:,1].square()+(enemies[:,:,2]*cfg.grid_height/cfg.grid_width).square()).sqrt()
        crowded=((distance<480/(cfg.grid_width*24))&(enemies[:,:,0]>.5)).sum(1)>=2
        chosen=base.argmax(1)//6
        chosen_risk=risk.gather(1,chosen[:,None]).squeeze(1)
        imminent=chosen_risk>safest+.25
        allowed=torch.where(crowded[:,None],retreat,safe)
        activate=(crowded | imminent) & allowed.any(1)
        mask=(~allowed).repeat_interleave(6,dim=1) & activate[:,None]
        logits=base.masked_fill(mask,-1e4)
        return self.action_dist.proba_distribution(action_logits=logits),self.value_net(latent_vf)


class ExploringPolicy(GuardedPolicy):
    """Continue through open space instead of idling/spinning without targets.

    Local wall rays and existing velocity provide direction persistence without
    privileged map coordinates or cross-episode mutable state.
    """
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        cfg=ObsConfig();cy,cx=cfg.grid_height//2,cfg.grid_width//2
        self.register_buffer('nav_rows',torch.tensor([[cy+round(my*r) for r in range(1,13)]
                             for mx,my in MOVE_VECTORS]),persistent=False)
        self.register_buffer('nav_cols',torch.tensor([[cx+round(mx*r) for r in range(1,13)]
                             for mx,my in MOVE_VECTORS]),persistent=False)

    def _distribution_and_value(self, obs):
        distribution, values=super()._distribution_and_value(obs)
        logits=distribution.distribution.logits.clone()
        cfg=ObsConfig();v=obs['vector'][:,-cfg.vector_size:]
        slices=VectorObservation(cfg).slices
        enemies=v[:,slices['enemies']].reshape(-1,cfg.n_enemies,cfg.ENEMY_DIM)
        shapes=v[:,slices['shapes']].reshape(-1,cfg.n_shapes,cfg.SHAPE_DIM)
        empty=(enemies[:,:,0].sum(1)+shapes[:,:,0].sum(1))<.5
        grid=obs['grid'][:,-3]
        free=(1-grid[:,self.nav_rows,self.nav_cols]).cumprod(dim=2)
        scores=free.sum(dim=2)/12+.35*(v[:,:2]@self.features_extractor.moves.T)
        scores[:,0]=-1e4
        viable=scores>=scores.max(1,keepdim=True).values-.15
        nav=scores.repeat_interleave(6,dim=1)+.05*logits
        # Keep the turret still and do not waste shots while looking for targets.
        controls=(torch.arange(54,device=v.device)%6)==0
        nav_allowed=(viable.repeat_interleave(6,dim=1)&controls[None,:]) & (logits>-1000)
        nav=nav.masked_fill(~nav_allowed,-1e4)
        combined=torch.where((empty & nav_allowed.any(1))[:,None],nav,logits)
        return self.action_dist.proba_distribution(action_logits=combined),values


class PathfindingPolicy(ExploringPolicy):
    """Attack visible slots; route around occlusion using only observed walls."""
    def _distribution_and_value(self, obs):
        from .local_navigation import route_move
        cfg=ObsConfig();slices=VectorObservation(cfg).slices
        v=obs['vector'][:,-cfg.vector_size:]
        grid=obs['grid'][:,-3]
        filtered=dict(obs)
        filtered['vector']=obs['vector'].clone()
        newest=filtered['vector'][:,-cfg.vector_size:]
        raw_targets=[]
        visibility=[]
        visible_counts=torch.zeros(v.shape[0],device=v.device)
        times=torch.arange(1,33,device=v.device,dtype=v.dtype)/33
        batch=torch.arange(v.shape[0],device=v.device)[:,None,None]
        for name,count,dimension in (('enemies',cfg.n_enemies,cfg.ENEMY_DIM),
                                     ('shapes',cfg.n_shapes,cfg.SHAPE_DIM)):
            slots=v[:,slices[name]].reshape(-1,count,dimension)
            raw_targets.append(slots)
            rows=(slots[:,:,2,None]*(cfg.grid_height/2)*times).round().long()+cfg.grid_height//2
            cols=(slots[:,:,1,None]*(cfg.grid_width/2)*times).round().long()+cfg.grid_width//2
            rows=rows.clamp(0,cfg.grid_height-1);cols=cols.clamp(0,cfg.grid_width-1)
            visible=(slots[:,:,0]>.5)&(grid[batch,rows,cols].amax(2)<.5)
            visibility.append(visible)
            visible_counts+=visible.sum(1)
            order=(~visible).long()*count+torch.arange(count,device=v.device)[None,:]
            order=order.argsort(dim=1)
            compact=(slots*visible[:,:,None]).gather(1,order[:,:,None].expand(-1,-1,dimension))
            newest[:,slices[name]]=compact.flatten(1)
        enemy_present=raw_targets[0][:,0,0]>.5
        preferred=torch.where(enemy_present[:,None],raw_targets[0][:,0,1:3],raw_targets[1][:,0,1:3])
        preferred_visible=torch.where(enemy_present,visibility[0][:,0],visibility[1][:,0])
        any_raw=sum(slots[:,:,0].sum(1) for slots in raw_targets)>.5
        direction=preferred*torch.tensor([cfg.grid_width,cfg.grid_height],device=v.device)
        direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(.001)
        reaches=torch.arange(1,3,device=v.device)[None,:]
        near_rows=(direction[:,1,None]*reaches).round().long()+cfg.grid_height//2
        near_cols=(direction[:,0,None]*reaches).round().long()+cfg.grid_width//2
        wall_near=grid[torch.arange(v.shape[0],device=v.device)[:,None],near_rows,near_cols].amax(1)>.5
        history=obs['vector'].reshape(v.shape[0],-1,cfg.vector_size)
        stalled=(history[:,:,:2].norm(dim=2).amax(1)<.15)&(history[:,:,2]>.0).all(1)
        reroute=any_raw&~preferred_visible&(wall_near|stalled)
        # Preserve ordinary combat observations until the hidden target actually
        # causes an obstruction; avoid changing distant, working combat behavior.
        filtered['vector']=torch.where(reroute[:,None],filtered['vector'],obs['vector'])
        distribution,values=super()._distribution_and_value(filtered)
        logits=distribution.distribution.logits.clone()
        occluded=(visible_counts==0)&reroute
        if occluded.any():
            features=self.extract_features(obs)
            for index in occluded.nonzero(as_tuple=False).flatten().tolist():
                target=next(slots[index,0,1:3] for slots in raw_targets if slots[index,:,0].sum()>.5)
                move=route_move(grid[index].detach().cpu().numpy(),target.detach().cpu().numpy())
                if move is None:
                    # Follow the wall tangentially until its end enters view.
                    free=(1-grid[index,self.nav_rows,self.nav_cols]).cumprod(dim=1).sum(1)
                    direction=target*torch.tensor([cfg.grid_width,cfg.grid_height],device=v.device)
                    direction=direction/direction.norm().clamp_min(.001)
                    moves=self.features_extractor.moves
                    score=free/12+.4*(moves@v[index,:2])-.7*(moves@direction).abs()
                    score[0]=-1e4
                    score=score.masked_fill(free<2,-1e4)
                    if score.max()<=-1000:
                        continue
                    move=int(score.argmax())
                risk=features[index,88:97]
                if risk[move]>.35 and risk[move]>risk.min()+.15:
                    continue
                logits[index]=-1e4
                # Do not keep firing or spinning at an occluded target while routing.
                logits[index,move*6]=0
        return self.action_dist.proba_distribution(action_logits=logits),values


class RecoveryPolicy(PathfindingPolicy):
    """Use episode-local actual-motion feedback to commit to a corner escape."""
    def _distribution_and_value(self, obs):
        distribution,values=super()._distribution_and_value(obs)
        state=obs['recovery'][:,-20:]
        active=state[:,9]>0
        logits=distribution.distribution.logits.clone()
        avoid=(state[:,11:20]<.5).repeat_interleave(6,dim=1)
        restricted=logits.masked_fill(avoid,-1e4)
        conflicts=(restricted.max(1).values<=-1000)&avoid.any(1)
        if conflicts.any():
            fallback,_=GuardedPolicy._distribution_and_value(self,obs)
            alternatives=fallback.distribution.logits.masked_fill(avoid,-1e4)
            use=conflicts&(alternatives.max(1).values>-1000)
            restricted=torch.where(use[:,None],alternatives,restricted)
        logits=torch.where((restricted.max(1).values>-1000)[:,None],restricted,logits)
        forced=torch.full_like(logits,-1e4)
        forced.scatter_(1,(state[:,:9].argmax(1)*6)[:,None],0)
        return self.action_dist.proba_distribution(action_logits=torch.where(active[:,None],forced,logits)),values
