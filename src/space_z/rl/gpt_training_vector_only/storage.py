"""실험 원본과 현재 상태를 파일에 기록한다."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_replace(temporary, target, attempts=5, delay=0.3):
    """Windows에서 대상 파일이 다른 프로세스에 잠깐 잠겨도 재시도 후 교체한다."""
    temporary, target = Path(temporary), Path(target)
    for attempt in range(1, attempts + 1):
        try:
            temporary.replace(target)
            return
        except (PermissionError, OSError):
            if attempt == attempts:
                raise
            time.sleep(delay)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    atomic_replace(temporary, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(directory, category, message, **evidence):
    append_json(directory / 'logs/events.jsonl', dict(time=now(), category=category, message=message, **evidence))
