"""기존 게임 환경 위에서 실행하는 독립적인 학습 실험."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
