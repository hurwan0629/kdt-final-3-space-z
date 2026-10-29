"""Local routes from the main observation wall grid, without world-map access."""
from collections import deque
import numpy as np

from space_z.game.actions import MOVE_VECTORS


def route_move(walls, target):
    """Return the first safe move toward a reachable target, or None to explore.

    Inflate walls for the ship's body and disallow diagonal corner cutting.
    If a wall spans the observed map, fall back to exploration instead of
    repeatedly pushing against the closest wall cell.
    """
    walls=np.asarray(walls)>.5
    height,width=walls.shape
    cy,cx=height//2,width//2
    padded=np.pad(walls,1,constant_values=True)
    blocked=np.zeros_like(walls)
    for dy in range(3):
        for dx in range(3):
            blocked |= padded[dy:dy+height,dx:dx+width]
    blocked[cy,cx]=False
    goal=(int(np.clip(round(cy+target[1]*height/2),0,height-1)),
          int(np.clip(round(cx+target[0]*width/2),0,width-1)))
    queue=deque([(cy,cx)])
    first={(cy,cx):None}
    while queue:
        y,x=queue.popleft()
        close=(y-goal[0])**2+(x-goal[1])**2<=4
        rows=np.rint(np.linspace(y,goal[0],7)).astype(int) if close else None
        cols=np.rint(np.linspace(x,goal[1],7)).astype(int) if close else None
        if (y,x)==goal or (close and not walls[rows,cols].any()):
            return first[(y,x)]
        for index,(mx,my) in enumerate(MOVE_VECTORS[1:],1):
            dx,dy=int(np.sign(mx)),int(np.sign(my))
            ny,nx=y+dy,x+dx
            if not (0<=ny<height and 0<=nx<width):
                continue
            if blocked[ny,nx] or (ny,nx) in first:
                continue
            if dx and dy and (blocked[y,nx] or blocked[ny,x]):
                continue
            first[(ny,nx)]=index if (y,x)==(cy,cx) else first[(y,x)]
            queue.append((ny,nx))
    # Do not select an unreachable wall face as a permanent destination.
    return None
