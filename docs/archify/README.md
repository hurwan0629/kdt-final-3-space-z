# Space Z 프로젝트 구조

[space-z.html](space-z.html)을 브라우저에서 열면 됩니다. 서버나 추가 설치가 필요 없습니다. 확대·검색·테마 전환·내보내기를 지원합니다.

## 파일

| 파일 | 용도 |
|---|---|
| [space-z.html](space-z.html) | 현재 프로젝트의 주요 역할·호출·소유 관계 |
| [space-z.architecture.json](space-z.architecture.json) | 편집 가능한 Archify 원본 |
| [source-evidence.json](source-evidence.json) | 근거 파일·라인·작업 트리 파일 해시 |
| [delivery-receipt.json](delivery-receipt.json) | 원본·HTML 해시와 9개 검증 결과 |
| [space-z.visual-check.json](space-z.visual-check.json) | 브라우저 자동 검증 결과 |
| [space-z.visual-check.html](space-z.visual-check.html) | 밝은·어두운 테마 스크린샷 |

## 읽는 방법과 범위

가운데는 게임 실행 → GameApplication → GameCore → 개체와 지형입니다. 아래의 학습 실행 → TankEnv도 동일한 GameCore를 진행합니다. GameCore는 Combat·PlayerProgression·EnemyController 사이의 결과 적용을 동기적으로 조율합니다. 화면은 앱이 전달한 게임 상태를 읽고 카메라·팝업·캐시를 관리합니다.

화살표는 선택한 주요 호출·소유·데이터 전달 관계입니다. 전체 import 그래프나 모든 반환 경로는 아닙니다. 예를 들어 앱은 코어 상태를 렌더러에 전달하며, 렌더러가 앱에서 게임 상태를 소유권째 넘겨받는 것은 아닙니다. 설정은 여러 객체가 같은 인스턴스를 참조하지만 전달선을 모두 그리지는 않았습니다. 성장 객체와 전투 객체도 같은 가변 개체를 참조하므로 완전히 격리된 모듈을 뜻하지 않습니다.

공통 학습 환경에는 행동 해석·관찰 생성·보상 계산이 포함됩니다. 평가와 실행은 별도 코드입니다. 개인 폴더 내부는 펼치지 않았고 해당 파일을 수정하거나 이동하지 않았습니다.

테스트·빌드·개발 도구는 [tests/](../../tests/), [scripts/](../../scripts/), [Windows 배포 안내](../development/windows-release.md)에 있습니다. 메뉴·패널 레이아웃, 데이터 타입, 상세 프레임 순서는 [프로젝트 책임 문서](../development/project-structure.md)를 참고합니다.

## 근거와 갱신

2026-10-30의 현재 작업 트리를 기준으로 작성했습니다. 아직 커밋하지 않은 구조 변경을 포함하므로, 이전 커밋을 현재 코드의 검증 근거로 표시하지 않았습니다. `source-evidence.json`의 `base_revision`은 작업 기반을 식별할 뿐이며, 내용의 기준은 함께 기록된 작업 트리 파일 해시입니다. 그래프의 넓은 질의는 일부 결과가 잘렸으므로 관련 소스와 책임 문서를 직접 확인해 보완했습니다.

코드가 바뀌어도 이 HTML은 자동으로 갱신되지 않습니다. JSON을 수정한 뒤 Archify의 `validate` → `deliver` → `visual-check` 순서로 다시 검증해야 합니다.

작성 내용은 한국어입니다. Archify 고정 Viewer UI와 HTML의 `lang`은 영어로 폴백합니다.

## 검증 기록

```text
diagram_type: architecture
validation: 9/9 showcase, 0 errors, 0 warnings
browser_evidence: passed
visual_review: passed
correction_rounds: 2
specification_sha256: e8fd24aa2c5140c08606409cce9b7c0cbbb34a6a45bd081d750bd02a004fabe7
artifact_sha256: c38ef650a120412a39bc259c339668f623c1cc95bcf2843f5b79142a4af7f0d6
```

브라우저 자동 검사는 1440×900, 1600×1000, 1920×1080, 2048×1320에서 잘림 없이 통과했습니다. 1440×900과 2048×1320의 밝은·어두운 테마 스크린샷 4장을 직접 확인했고, 노드·라벨 겹침이나 경로 교차는 발견하지 못했습니다. 확대·검색·내보내기 동작은 수동으로 조작하지 않았습니다. 게임 실행·모델 로딩·학습은 이번 문서 생성 과정에서 시험하지 않았습니다.
