"""관찰·행동·보상을 공통 게임 코어에 연결하는 Gymnasium 환경."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
from gymnasium.spaces import Space

from space_z.config import EnvConfig, GameConfig
from space_z.game.actions import ROT_DIR, encode_action, move_from_vector
from space_z.game.game_core import GameCore
from space_z.game.types import CombatEvents
from space_z.rl.actions.action_decoder import ActionDecoder
from space_z.rl.observations.observation_builder import Observation, ObservationBuilder
from space_z.rl.rewards import RewardCalculator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from space_z.rendering.renderer import Renderer


class TankEnv(gym.Env[Observation, "int | NDArray[np.float32]"]):
    """관찰 생성·행동 해석·보상 계산을 연결하고 환경 진행을 조율할 객체.

    구현 방향:
        DQN·PPO 구분 없이 전달받은 관찰 생성기와 행동 해석기를 사용한다.
        모델 학습·모델 저장·평가·자원 사용량 측정은 환경 밖에서 담당한다.
        담당 객체는 호출자가 만들어 전달하며, 환경의 close()가 관찰 생성기도 정리한다.
    """

    def __init__(
        self,
        game_config: GameConfig,
        observation_builder: ObservationBuilder,
        action_decoder: ActionDecoder,
        reward_calculator: RewardCalculator,
        env_config: EnvConfig | None = None,
        render_mode: Literal["human"] | None = None,
    ) -> None:
        """게임 설정과 담당 객체를 받아 학습 환경을 준비한다.

        인자:
            game_config: 적 수·체력·지형·성장·탄약 제한 등 게임 규칙.
            observation_builder: 벡터 또는 픽셀 관찰을 생성할 객체.
            action_decoder: 이산 번호 또는 연속 배열을 게임 제어 입력으로 해석할 객체.
            reward_calculator: 실제 발사 수를 포함한 누적 전투 이벤트로 보상을 계산할 객체.
            env_config: 행동 반복 프레임 수와 최대 환경 step 수. None이면 기본 EnvConfig 사용.
            render_mode: None이면 사람용 화면 없이 실행, "human"이면 게임 창 표시.
                픽셀 관찰 생성 여부는 observation_builder가 결정한다.

        반환:
            None. 환경 객체에 설정·담당 객체·게임 상태를 보관한다.
        """
        # 1) 설정을 준비하고 frame_skip·max_steps가 양의 정수인지 검증한다.
        # 2) 받은 설정과 담당 객체를 인스턴스 속성으로 저장한다.
        # 3) 관찰 생성기·행동 해석기의 정의에 맞춰 observation_space·action_space를 구성한다.
        # 4) GameCore를 생성하고 환경 step 수·종료 상태·화면 자원을 초기화한다.
        # 5) 지원하는 render_modes와 fps // frame_skip 기준의 render_fps를 정의한다.
        self.environment_config = env_config or EnvConfig()
        for name in ("frame_skip", "max_steps"):
            value = getattr(self.environment_config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}는 양의 정수여야 합니다.")
        if render_mode not in (None, "human"):
            raise ValueError('render_mode는 None 또는 "human"이어야 합니다.')
        if not isinstance(observation_builder, ObservationBuilder) or not isinstance(action_decoder, ActionDecoder):
            raise TypeError("관찰 생성기와 행동 해석기는 공통 인터페이스를 구현해야 합니다.")
        if not callable(getattr(reward_calculator, "calculate", None)):
            raise TypeError("보상 계산기는 calculate(events)를 제공해야 합니다.")
        self.observation_space = observation_builder.observation_space
        self.action_space = action_decoder.action_space
        if not isinstance(self.observation_space, Space) or not isinstance(self.action_space, Space):
            raise TypeError("관찰·행동 공간은 Gymnasium Space여야 합니다.")
        self.game_config = game_config
        self.observation_builder = observation_builder
        self.action_decoder = action_decoder
        self.reward_calculator = reward_calculator
        self.render_mode = render_mode
        self.game = GameCore(game_config)
        self.steps = 0  # reset 이후 진행한 환경 step 수
        self._needs_reset = True  # 생성 직후·종료 이후에는 reset이 필요하다.
        self._closed = False  # close 이후 재사용을 막는다.
        self._renderer: Renderer | None = None  # 사람용 렌더러는 필요할 때 생성한다.
        self.metadata = {"render_modes": ["human"],
                         "render_fps": max(1, game_config.fps // self.environment_config.frame_skip)}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """새 에피소드를 시작하고 첫 관찰과 부가 정보를 반환한다.

        인자:
            seed: 게임 난수 초기화에 사용할 시드. 같은 시드·입력으로 결과를 재현한다.
            options: 에피소드 초기화 옵션을 위한 인자. 현재 별도 옵션은 정의하지 않는다.

        반환:
            (observation, info). 픽셀 관찰은 image·progression 딕셔너리다.
            관찰 크기는 observation_builder가 정의한 observation_space와 일치해야 한다.
        """
        # 1) Gymnasium 기반 클래스의 난수 초기화와 GameCore.reset(seed)를 연결한다.
        # 2) 환경 step 수·종료 상태·이전 판의 전투 및 발사 기록을 초기화한다.
        # 3) 관찰 생성기의 지형 캐시·누적 이미지와 이전 판의 화면 연출을 초기화한다.
        # 4) 첫 관찰과 _info() 결과를 만들고, human 모드이면 화면도 출력한다.
        # 5) (observation, info)를 반환한다.
        if self._closed:
            raise RuntimeError("종료한 환경은 다시 사용할 수 없습니다.")
        if options:
            raise ValueError("현재 reset에서 지원하는 추가 옵션은 없습니다.")
        super().reset(seed=seed)
        # seed=None도 Gymnasium 난수열에서 시드를 뽑아 core의 별도 난수열에 연결한다.
        core_seed = seed if seed is not None else int(self.np_random.integers(0, 2**32))
        self.game.reset(core_seed)
        self.steps = 0
        self._needs_reset = False
        self.observation_builder.reset()
        if self._renderer is not None:
            self._renderer.reset_score_popups()
        observation = self.observation_builder.build(self.game)
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self,
        action: int | NDArray[np.float32],
    ) -> tuple[
        Observation,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """한 행동을 게임에 적용하고 다음 관찰·보상·종료 정보 등을 반환한다.

        인자:
            action: 이산 모드에서는 정수 행동 번호, 연속 모드에서는 실수 배열.
                이동·회전·발사 여부의 해석과 입력 범위 검증은 action_decoder가 담당한다.

        반환:
            (observation, reward, terminated, truncated, info).
            observation: 행동 이후의 벡터 또는 픽셀 관찰.
            reward: 이번 환경 step의 누적 전투 결과와 실제 발사 비용을 반영한 보상.
            terminated: 플레이어 사망 등 게임 규칙에 따른 종료 여부.
            truncated: 게임이 종료되지 않은 상태에서 최대 환경 step 수에 도달했는지 여부.
            info: 누적 전적·진행 상태·이번 step의 전투 결과 등 부가 정보.
        """
        # 1) 에피소드 진행 가능 여부를 확인하고 action_decoder로 게임 제어 입력을 얻는다.
        # 2) 같은 제어 입력을 frame_skip회 적용하고, 중간에 사망하면 반복을 중단한다.
        #    각 게임 프레임의 피해·처치·사망·player_shots_fired 이벤트를 함께 누적한다.
        #    발사 수는 요청 횟수나 현재 탄환 목록 길이의 차이로 계산하지 않는다.
        #    쿨다운으로 발사되지 않은 요청에는 비용을 부과하지 않는다.
        # 3) 환경 step 수를 한 번 증가시키고 reward_calculator로 누적 보상을 계산한다.
        # 4) terminated와 truncated를 판단하고 행동 이후의 관찰 및 _info() 결과를 만든다.
        # 5) human 모드이면 화면을 출력하고, 화면 없는 실행에서는 대기 연출 기록을 정리한다.
        # 6) (observation, reward, terminated, truncated, info)를 반환한다.
        if self._closed or self._needs_reset:
            raise RuntimeError("step 전에 reset()으로 진행 가능한 에피소드를 시작해야 합니다.")
        control = self.action_decoder.decode(action)

        # 1) 성장 요청은 환경 step당 한 번만 적용한다.
        if self.game_config.progression_enabled:
            if control.stat_upgrade_key is not None:
                self.game.progression.upgrade_stat(control.stat_upgrade_key)
            if control.evolution_choice is not None:
                self.game.progression.choose_reward(control.evolution_choice)

        # 2) 이동 세기는 고정하고, 연속 회전만 코어의 선택 인자로 전달한다.
        move = move_from_vector(control.move_x, control.move_y)
        rotation = ROT_DIR.index(control.rotation) if control.rotation in ROT_DIR else 0
        core_action = encode_action(move, rotation)
        for _ in range(self.environment_config.frame_skip):
            if control.rotation in ROT_DIR:
                self.game.step(core_action, want_fire=control.want_fire)
            else:
                self.game.step(core_action, want_fire=control.want_fire, rotation_input=control.rotation)
            if self.game.game_over:
                break

        # 3) 코어가 여러 프레임 동안 누적한 이벤트를 한 번 꺼내 모든 소비자에 공유한다.
        self.steps += 1
        events = self.game.pop_events()
        reward = self.reward_calculator.calculate(events)
        terminated = self.game.game_over
        truncated = not terminated and self.steps >= self.environment_config.max_steps
        self._needs_reset = terminated or truncated
        observation = self.observation_builder.build(self.game)
        info = self._info(events)
        if self.render_mode == "human":
            self.render()
        else:
            self.game.pop_score_award_events()
        return observation, reward, terminated, truncated, info

    def _info(self, events: CombatEvents | None = None) -> dict[str, Any]:
        """환경 결과에 덧붙일 부가 정보를 모은다.

        인자:
            events: 이번 환경 step 동안 누적한 전투 결과. reset() 직후에는 None.

        반환:
            점수·처치 수·게임 프레임·환경 step·맵·성장·무장 상태와 events를 담은 딕셔너리.
            모델 입력 관찰이나 보상을 대신하지 않으며, 기록·평가·디버깅에 사용한다.
        """
        # 1) 코어와 각 담당 객체에서 누적 전적·게임 진행·성장·무장 상태를 읽는다.
        # 2) 받은 events를 포함해 reset/step에서 공통으로 사용할 키를 정리한다.
        # 3) 변경 가능한 내부 자료를 외부에서 바꾸지 못하도록 필요한 값을 복사해 반환한다.
        #    게임 상태를 변경하거나 다른 메서드가 사용할 이벤트를 여기서 소비하지 않는다.
        progression = self.game.progression
        return {
            "score": progression.score, "kills": progression.kills,
            "map": self.game.terrain.name, "frames": self.game.frame, "steps": self.steps,
            "events": dict(events) if events is not None else None,
            "level": progression.level, "xp": progression.xp,
            "pending_rewards": progression.pending_rewards, "job": self.game.player.job,
            "ammo": self.game.player.ammo if self.game_config.limited_ammo else None,
            "reload_seconds_left": self.game.combat.reload_frames_left / self.game_config.fps,
            "upgrade_points": progression.upgrade_points, "family": progression.family,
            "stat_choices": [card.key for card in progression.stat_choices],
            "evolution_choices": [card.key for card in progression.reward_choices],
            "special_reward_levels": list(progression.special_reward_levels),
            "stat_levels": dict(progression.stat_levels), "special_levels": dict(progression.special_levels),
        }

    def render(self) -> None:
        """human 모드에서 사람이 확인할 게임 화면을 출력한다.

        인자:
            없음. 환경에 저장한 render_mode와 현재 게임 상태를 사용한다.

        반환:
            None. 픽셀 관찰 배열은 이 메서드의 반환값이 아니라 관찰 생성기에서 준비한다.
        """
        # 1) human 모드인지 확인하고 화면이 처음 필요할 때 렌더러와 창을 준비한다.
        # 2) 창 이벤트와 점수 연출을 처리하고 현재 게임 상태를 화면에 그린다.
        # 3) 화면을 갱신하고 render_fps에 맞춰 출력 속도를 제한한다.
        #    게임 프레임·학습 보상은 변경하지 않고, 픽셀 학습용 이미지 생성과 구분한다.
        if self.render_mode != "human" or self._closed:
            return
        import pygame
        from space_z.rendering.renderer import Renderer

        if self._renderer is None:
            pygame.display.init()
            pygame.font.init()
            self._renderer = Renderer(self.game_config)
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.VIDEORESIZE:
                self._renderer.resize(event.size)
        self._renderer.add_score_popups(self.game.pop_score_award_events())
        self._renderer.draw(self.game, show_controls=False)
        pygame.display.flip()
        self._renderer.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """환경이 사용한 화면과 관찰 생성 관련 자원을 정리한다.

        인자:
            없음. 환경과 담당 객체가 보유한 자원을 사용한다.

        반환:
            None. 화면이 없거나 이미 정리한 상태에서 다시 호출해도 안전해야 한다.
        """
        # 1) 사람용 화면과 픽셀 관찰 생성에 사용한 자원이 있는지 확인한다.
        # 2) 환경이 생성한 자원을 해제하고 관련 참조를 정리한다.
        if self._closed:
            return
        try:
            self.observation_builder.close()
        finally:
            if self._renderer is not None:
                import pygame

                if pygame.display.get_surface() is self._renderer.screen:
                    pygame.display.quit()
                self._renderer = None
            self._closed = True
