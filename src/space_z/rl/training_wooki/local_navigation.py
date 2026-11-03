"""Local routes from the main observation wall grid, without world-map access."""

# 관찰된 벽 격자 안에서만 경로를 찾는다. 전체 월드 지형은 조회하지 않는다.

from collections import deque
import numpy as np

from space_z.game.actions import MOVE_VECTORS


def route_move(walls, target):
    """Return the first safe move toward a reachable target, or None to explore.

    Inflate walls for the ship's body and disallow diagonal corner cutting.
    If a wall spans the observed map, fall back to exploration instead of
    repeatedly pushing against the closest wall cell.
    """
    # 벽 여유 공간을 확보한 뒤 BFS로 탐색하여, 목표로 향하는 첫 이동 번호를 반환한다.
    walls = np.asarray(walls) > .5
    height, width = walls.shape
    cy, cx = height // 2, width // 2

    # 1) 벽 주위 한 칸도 막힌 것으로 취급해 기체가 지나갈 여유를 확보한다.
    padded = np.pad(walls, 1, constant_values=True)
    blocked = np.zeros_like(walls)
    for dy in range(3):
        for dx in range(3):
            blocked |= padded[dy : dy + height, dx : dx + width]
    blocked[cy, cx] = False
    goal = (
        int(np.clip(round(cy + target[1] * height / 2), 0, height - 1)),
        int(np.clip(round(cx + target[0] * width / 2), 0, width - 1)),
    )

    # 2) 현재 격자 중심부터 탐색하며 각 칸으로 향하는 최초 이동 번호를 전파한다.
    queue = deque([(cy, cx)])
    first = {(cy, cx): None}
    while queue:
        y, x = queue.popleft()
        close = (y - goal[0]) ** 2 + (x - goal[1]) ** 2 <= 4
        rows = np.rint(np.linspace(y, goal[0], 7)).astype(int) if close else None
        cols = np.rint(np.linspace(x, goal[1], 7)).astype(int) if close else None
        if (y, x) == goal or (close and not walls[rows, cols].any()):
            return first[(y, x)]
        for index, (mx, my) in enumerate(MOVE_VECTORS[1:], 1):
            dx, dy = int(np.sign(mx)), int(np.sign(my))
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if blocked[ny, nx] or (ny, nx) in first:
                continue

            # 대각선 이동 시 양옆 벽 사이를 가로질러 모서리를 통과하지 못하게 한다.
            if dx and dy and (blocked[y, nx] or blocked[ny, x]):
                continue
            first[(ny, nx)] = index if (y, x) == (cy, cx) else first[(y, x)]
            queue.append((ny, nx))
    # 도달할 수 없는 벽 앞을 고정 목표로 삼지 않고 탐색으로 넘긴다.
    return None
