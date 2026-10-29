"""Physical progress tracking and short committed escapes for corner contacts."""
from collections import deque
import numpy as np
from space_z.game.actions import MOVE_VECTORS


def escape_direction(walls, offset, velocity, radius, tile, friction, accel, failures):
    """Predict local circle/wall motion for each legal movement direction.

    Uses observed wall tiles and sub-tile offset, never the unseen world map.
    Collision response removes velocity into walls, matching main's sliding.
    """
    rows,cols=np.nonzero(np.asarray(walls)>.5)
    cy,cx=np.array(walls.shape)//2
    lower=np.stack(((cols-cx-.5)*tile,(rows-cy-.5)*tile),axis=1)
    upper=lower+tile
    origin=np.asarray(offset,dtype=float)
    scores=np.full(9,-1e6)
    for move,(mx,my) in enumerate(MOVE_VECTORS[1:],1):
        p=origin.copy();v=np.asarray(velocity,dtype=float).copy()
        for _ in range(48):
            v=v*friction+np.array([mx,my])*accel
            p+=v
            for _ in range(2):
                near=(lower[:,0]<p[0]+radius)&(upper[:,0]>p[0]-radius)&(lower[:,1]<p[1]+radius)&(upper[:,1]>p[1]-radius)
                for lo,hi in zip(lower[near],upper[near]):
                    delta=p-np.clip(p,lo,hi);distance=float(np.linalg.norm(delta))
                    if distance>=radius:
                        continue
                    if distance<1e-8:
                        choices=np.array([[lo[0]-radius,p[1]],[hi[0]+radius,p[1]],
                                          [p[0],lo[1]-radius],[p[0],hi[1]+radius]])
                        dest=choices[np.linalg.norm(choices-p,axis=1).argmin()]
                        normal=(dest-p)/max(np.linalg.norm(dest-p),1e-8);p=dest
                    else:
                        normal=delta/distance;p+=normal*(radius-distance)
                    vn=float(v@normal)
                    if vn<0: v-=normal*vn
        clearance=float(np.linalg.norm(p-np.clip(p,lower,upper),axis=1).min())-radius if len(lower) else tile*4
        scores[move]=np.linalg.norm(p-origin)+.7*min(clearance,tile*4)-float(failures[move])*80
    return int(scores.argmax())


class CornerRecovery:
    """Episode-local state exposed to the policy, so PPO records executed actions."""
    def __init__(self, position):
        self.positions=deque([np.asarray(position,dtype=float)],maxlen=16)
        self.intents=deque(maxlen=16);self.productivity=deque(maxlen=16)
        self.remaining=0;self.move=0;self.count=0;self.failures=np.zeros(9)
        self.origin=np.asarray(position,dtype=float)
        self.stalled=0
        self.avoid_center=None

    def observation(self):
        result=np.zeros(20,np.float32)
        if self.remaining:
            result[self.move]=1
        result[9]=self.remaining/12
        result[10]=min(1.,self.stalled/16)
        result[11:]=1
        if self.avoid_center is not None:
            position=self.positions[-1];distance=np.linalg.norm(position-self.avoid_center)
            if distance<220:
                destinations=position+np.asarray(MOVE_VECTORS)*48
                result[11:]=(np.linalg.norm(destinations-self.avoid_center,axis=1)>=distance-4)
        return result

    def update(self, game, walls, attempted=True, productive=False):
        p=game.player;position=np.array([p.x,p.y])
        self.positions.append(position)
        self.intents.append(attempted);self.productivity.append(productive)
        if self.avoid_center is not None and np.linalg.norm(position-self.avoid_center)>400:
            self.avoid_center=None
        if self.remaining:
            self.remaining-=1
            if self.remaining==0:
                if np.linalg.norm(position-self.origin)<24:
                    self.failures[self.move]+=1
                else:
                    self.failures*=0
                    self.positions.clear();self.positions.append(position)
                    self.intents.clear();self.productivity.clear()
            return
        cy,cx=np.array(walls.shape)//2
        near_wall=np.asarray(walls)[cy-2:cy+3,cx-2:cx+3].any()
        span=np.linalg.norm(np.ptp(np.asarray(self.positions),axis=0))
        self.stalled=len(self.positions) if near_wall and span<18 else 0
        if self.stalled<12:
            return
        if not any(self.intents) and any(self.productivity):
            return  # Intentional stationary fire is still making combat progress.
        cfg=game.config;tile=game.terrain.tile
        self.move=escape_direction(walls,position%tile-tile/2,
                                   (p.velocity_x,p.velocity_y),p.radius,tile,
                                   cfg.friction,cfg.accel*p.accel_scale,self.failures)
        self.origin=position.copy();self.remaining=12;self.count+=1
        if self.avoid_center is None:
            self.avoid_center=position.copy()
