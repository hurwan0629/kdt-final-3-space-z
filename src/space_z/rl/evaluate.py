"""학습 알고리즘과 독립적으로 고정 시드의 에피소드를 평가한다."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from numbers import Integral
from statistics import fmean, pstdev

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from space_z.rl.environment import TankEnv
    from space_z.rl.resource_monitor import ResourceMonitor


class PolicyEvaluator:
    """공통 평가 조건에서 모델의 보상·점수·처치·생존 시간을 측정할 객체."""

    def __init__(
        self,
        environment: TankEnv,
        action_selector: Callable[[Any], int | NDArray[np.float32]],
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        """평가 환경·행동 선택 함수·선택적인 자원 측정기를 받는다.

        인자:
            environment: 학습과 분리해 평가에 사용할 환경. 종료·해제는 호출자가 맡는다.
            action_selector: 관찰을 받아 이산 행동 번호 또는 연속 행동 배열을 반환하는 함수.
                관찰은 관찰 생성기가 정한 벡터·이미지·딕셔너리 형식을 그대로 전달한다.
                평가 중에는 모델을 학습시키지 않는 함수를 전달해야 한다.
            resource_monitor: 평가 전체의 자원 사용량을 측정할 객체. None이면 측정하지 않는다.

        반환:
            None. 전달받은 객체와 함수를 속성으로 보관한다.
        """
        if not callable(action_selector):
            raise TypeError("action_selector는 관찰을 받아 행동을 반환하는 함수여야 합니다.")
        self.environment = environment
        self.action_selector = action_selector
        self.resource_monitor = resource_monitor

    def evaluate(self, seeds: Sequence[int]) -> dict[str, Any]:
        """지정한 시드마다 에피소드를 평가하고 결과를 모은다.

        인자:
            seeds: 비교할 모델에 공통으로 적용할 비어 있지 않은 평가 시드 목록.

        처리:
            1) 시드마다 evaluate_episode()를 호출한다.
            2) 에피소드별 결과와 각 수치의 평균·표준편차를 정리한다.
            3) 측정기가 있으면 평가 전체의 자원 사용량을 함께 기록한다.

        반환:
            episodes: 시드별 결과 목록.
            summary: 보상 합계·점수·처치·생존 시간·step 수·프레임 수의 평균과 표준편차.
            resources: 자원 측정 결과. 측정기가 없으면 None.
            위 항목을 키로 갖는 딕셔너리.
        """
        seeds = tuple(seeds)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
            raise ValueError("평가 시드는 0 이상의 정수로 된 비어 있지 않은 목록이어야 합니다.")
        resources = None
        if self.resource_monitor is not None:
            self.resource_monitor.start()
        try:
            episodes = [self.evaluate_episode(int(seed)) for seed in seeds]
        finally:
            if self.resource_monitor is not None:
                resources = self.resource_monitor.stop()
        summary = {}
        for key in ("total_reward", "score", "kills", "survival_seconds", "environment_steps", "game_frames"):
            values = [episode[key] for episode in episodes]
            summary[key] = {"mean": fmean(values), "std": pstdev(values)}
        return {"episodes": episodes, "summary": summary, "resources": resources}

    def evaluate_episode(self, seed: int) -> dict[str, Any]:
        """하나의 시드로 종료 또는 시간 제한까지 평가한다.

        인자:
            seed: environment.reset()에 전달할 게임 초기화 시드.

        처리:
            1) 환경을 초기화하고 관찰을 action_selector에 전달한다.
            2) 선택한 행동으로 step()을 반복하며 보상을 누적한다.
            3) terminated 또는 truncated가 참이면 최종 지표를 모은다.

        반환:
            seed, total_reward, score, kills, survival_seconds, environment_steps,
            game_frames, terminated, truncated를 담은 딕셔너리.
            생존 시간은 게임 프레임과 게임 fps로 계산하며 실제 평가 소요 시간과 구분한다.
        """
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("평가 시드는 0 이상의 정수여야 합니다.")
        observation, info = self.environment.reset(seed=int(seed))
        total_reward, steps = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = self.action_selector(observation)
            observation, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward
            steps += 1
        return {"seed": int(seed), "total_reward": total_reward, "score": info["score"],
                "kills": info["kills"], "survival_seconds": info["frames"] / self.environment.game_config.fps,
                "environment_steps": steps, "game_frames": info["frames"],
                "terminated": terminated, "truncated": truncated}
