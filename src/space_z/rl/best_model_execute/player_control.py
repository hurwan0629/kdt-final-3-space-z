"""이산·연속 행동 해석기가 공통으로 반환할 게임 제어 입력."""

from dataclasses import dataclass


@dataclass
class PlayerControl:
    """게임에 전달할 이동·회전·발사 입력과 스탯 강화·전직 선택.

    이 객체는 값만 보관한다. 범위 검증과 이동 벡터 길이 제한은 ActionDecoder가 담당한다.
    실제 위치·속도·각도 갱신과 발사 처리는 게임 코어가 담당한다.
    스탯 강화·전직 요청은 환경 step당 한 번만 적용하고, 가능 여부는 성장 담당 객체가 검사한다.
    """

    move_x: float  # 정지·8방향의 정규화한 가로 성분: 음수는 왼쪽, 양수는 오른쪽
    move_y: float  # 정지·8방향의 정규화한 세로 성분: 음수는 위쪽, 양수는 아래쪽
    rotation: float  # 회전 입력 세기(-1~1): 음수는 반시계, 양수는 시계 방향
    want_fire: bool  # 새 발사 묶음의 시작 요청 여부; 이미 예약된 발사는 별도 처리
    stat_upgrade_key: str | None = None  # 강화할 스탯 키; None이면 강화 요청 없음
    evolution_choice: int | None = None  # 현재 전직·숙련 선택지의 0부터 시작하는 번호; None이면 요청 없음
