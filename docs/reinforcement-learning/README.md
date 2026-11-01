# 강화학습 안내

공통 게임 환경과 학습 실행 코드가 함께 있습니다. 게임 규칙은 `GameCore`, 학습과의 연결은 `TankEnv`, 정책 갱신은 학습 실행 코드가 담당합니다. 실행 경로는 [RL 안내](../../src/space_z/rl/README.md)를 참고합니다.

- [환경 인터페이스·관찰·보상](environment.md): `TankEnv.reset()/step()`, 벡터·픽셀 관찰, 이산·연속 행동, 평가·자원 측정.
- [[development/classes/GameCore|화면 없는 공통 게임 코어]]: 실제 규칙과 프레임 진행.
- [[development/configuration|학습 설정 위치]]: `ObsConfig`·`RewardConfig`·`EnvConfig`.
- [[development/testing-and-changes|무작위 행동·seed 재현성 검증]]: 개발 도구 실행.

학습의 한 step과 게임의 한 프레임은 다릅니다. `TankEnv`는 성장 요청을 한 번 적용하고 설정된 프레임 수만큼 코어를 진행한 뒤, 전투 이벤트를 한 번 소비해 보상을 계산합니다. 관찰·행동·보상 형식과 기존 import 경로는 구조 정리에서 유지합니다. 개인 폴더는 별도 유지하며 수정하거나 이동하지 않습니다.

[[getting-started|설치와 환경 사용 예시]] · [[README|문서 목차]]
