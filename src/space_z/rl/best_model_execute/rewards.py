"""한 환경 step의 전투 결과와 실제 발사 탄환 수를 학습 보상으로 변환한다."""

import math
from numbers import Integral

from space_z.config import RewardConfig
from space_z.game.types import CombatEvents


class RewardCalculator:
    """보상 계수만 보관하고, 전달받은 이번 step의 결과로 보상을 계산한다.

    게임 상태와 이벤트는 변경하지 않고, 이전 step의 보상이나 발사 수를 누적하지 않는다.
    여러 게임 프레임의 이벤트를 모으는 작업은 TankEnv가 담당하며, 실제 발사 수도 이벤트에 포함한다.
    """

    def __init__(self, reward_config: RewardConfig) -> None:
        """발사 비용 설정을 검증하고 보상 계산에 사용할 설정을 보관한다.

        인자:
            reward_config: 피해·처치·피격·사망·환경 step 보상 계수와 탄환 1발당 발사 비용.

        반환:
            None. 전달받은 설정 객체를 인스턴스에 저장한다.

        예외:
            ValueError: shot_cost가 음수이거나 NaN·무한대인 경우.
        """
        # if not math.isfinite(reward_config.shot_cost) or reward_config.shot_cost < 0:
        #     raise ValueError("shot_cost는 0 이상의 유한한 값이어야 합니다.")
        self.reward_config = reward_config  # 보상 항목별 계수와 실제 발사 1발당 차감 비용

    def calculate(self, events: CombatEvents) -> float:
        """이번 환경 step의 전투 보상에서 실제 발사 비용을 차감해 반환한다.

        인자:
            events: 한 환경 step에 포함된 모든 게임 프레임의 누적 전투 결과.
                천체 가중 피해·천체 파괴 점수·적 피해·적 처치 수·피격 피해·사망 여부를 사용한다.
                player_shots_fired는 같은 기간 플레이어가 실제 발사한 탄환 수(0 이상의 정수).
                발사 요청 횟수가 아니며, 포신별 탄환과 예약 점사도 실제 발사 시점에 기록된다.

        처리:
            1) 실제 발사 수가 유효한지 확인한다.
            2) 전투 결과에 보상 계수를 곱하고 환경 step 보상을 한 번 더한다.
            3) 실제 발사 탄환 수에 따른 비용을 차감한다.

        반환:
            float: 이번 환경 step에 대한 학습 보상. 게임 점수·경험치와는 별개다.

        예외:
            TypeError: events의 player_shots_fired가 정수가 아니거나 bool인 경우.
            ValueError: events의 player_shots_fired가 음수인 경우.
        """
        # 1) 발사 수를 검증해 잘못된 입력으로 비용이 보상으로 바뀌는 것을 막는다.
        shots_fired = events["player_shots_fired"]
        if isinstance(shots_fired, bool) or not isinstance(shots_fired, Integral):
            raise TypeError("player_shots_fired는 실제 발사 탄환 수를 나타내는 정수여야 합니다.")
        if shots_fired < 0:
            raise ValueError("player_shots_fired는 0 이상이어야 합니다.")

        # 2) 전투 보상을 합산한다. 시간 보상은 게임 프레임 수와 무관하게 환경 step당 한 번 적용한다.
        config = self.reward_config
        reward = (
            config.shape_damage * events["shape_damage_weighted"]
            + config.shape_destroy * events["shape_destroyed_score"]
            + config.enemy_damage * events["enemy_damage"]
            + config.enemy_kill * events["enemies_killed"]
            + config.damage_taken * events["damage_taken"]
            + (config.death if events["died"] else 0.0)
            + config.time_step
        )

        # 3) 실제 탄환마다 비용을 차감한다. 쿨다운 등으로 발사되지 않은 요청은 세지 않는다.
        return float(reward - config.shot_cost * shots_fired)
