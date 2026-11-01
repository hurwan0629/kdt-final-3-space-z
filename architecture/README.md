# Space Z 게임 구조도

브라우저에서 `space-z.html`을 열면 됩니다. 설치나 서버 실행이 필요 없습니다.

- `space-z.html`: 게임 아키텍처 다이어그램. 검색, 확대/축소, 테마, 내보내기 지원.
- `space-z.architecture.json`: 편집 가능한 Archify 원본.
- `delivery-receipt.json`: 원본·HTML 해시, 바이트 수, 코드 근거 검증 결과.
- `space-z.visual-check.json`: 자동 브라우저 검사 결과.
- `space-z.visual-check.html`: 밝은/어두운 테마 스크린샷 모음.

게임 실행·입력·화면·규칙·전투·적 AI·성장·지형·설정·리소스를 주요 역할 단위로 표현했습니다. 화살표는 선택한 주요 호출·소유 관계이며 전체 import 그래프는 아닙니다. RL 코드, 학습 설정, F1 관찰 디버그 경로는 제외했습니다. 실제 프로젝트에서 해당 코드를 삭제하거나 변경하지 않았습니다.

GameCore가 관리하는 Tank, Shape, Bullet, Beam, Explosion과 행동 인코딩 및 공통 타입은 별도 노드 대신 코어·전투 역할에 묶었습니다. Renderer는 GameApplication이 전달한 동일한 코어 상태를 읽습니다. Config는 초기화 시 여러 객체에 전달되며, 전체 설정 전달선은 생략했습니다.

내용은 한국어이며 Archify 고정 Viewer UI와 HTML lang은 영어로 폴백합니다.

## 검증 기록

diagram_type: architecture
validation: 9/9 showcase, 0 errors, 0 warnings
browser_evidence: passed
visual_review: passed
correction_rounds: 2

브라우저 자동 검사는 1440×900, 1600×1000, 1920×1080, 2048×1320에서 페이지 잘림·가독성 기준을 통과했습니다. 1440×900 및 2048×1320의 밝은/어두운 테마 스크린샷 4장을 직접 확인했으며 노드·라벨 겹침이나 잘림은 발견하지 못했습니다. 검색·포커스·내보내기 기능을 수동으로 조작해 보지는 않았습니다. 게임 실행 테스트는 수행하지 않았습니다.

specification_sha256: 66feb5f9bf30b9e567c29703bf7359742d0aa734afdeb67d51eb55588f2b181b
artifact_sha256: 82bf901022d5aa25834c81f1fba210731e4b469de5086beb10e4f97425f12518

근거 경로 18개는 커밋 cc3d8c06030397e0ddbad9cc7cedd712853377f1에서 존재 여부를 검증했습니다. 다이어그램 설명은 작업 당시 로컬 코드를 읽어 작성한 스냅샷으로, 코드 변경 시 자동으로 갱신되지 않습니다.
