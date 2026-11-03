"""평가 원본을 복사하고 시드별 결과에서 점수·생존 시간 평균을 다시 계산한다."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter


OUTPUT = Path(__file__).resolve().parent
RL = OUTPUT.parent
GROUPS = {
    "training_hurwan": "runs/*/*/evaluations/results.jsonl",
    "gpt_training": "experiments/*/evaluations.jsonl",
    "gpt_training_rotation_continuous": "experiments/*/evaluations.jsonl",
    "gpt_training_vector_only": "experiments/*/logs/evaluations.jsonl",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluation_nodes(value, pointer=""):
    """집계 JSON의 배열·중첩 구조에서 에피소드가 있는 평가 묶음만 찾는다."""
    if isinstance(value, dict):
        episodes = value.get("episodes")
        if isinstance(episodes, list) and episodes and all(
            isinstance(e, dict) and {"seed", "score", "survival_seconds"} <= e.keys() for e in episodes
        ):
            yield pointer, value
            return
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from evaluation_nodes(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from evaluation_nodes(child, f"{pointer}/{index}")


def normalize(raw, source, locator, context, kind):
    episodes = raw["episodes"]
    assert episodes, f"Empty evaluation: {source} {locator}"
    metrics = {}
    for name in ("score", "survival_seconds"):
        values = [e[name] for e in episodes]
        assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in values)
        mean = statistics.mean(values)
        stored = raw.get("summary", {}).get(name, {}).get("mean")
        if stored is not None:
            assert math.isclose(mean, stored, rel_tol=1e-9, abs_tol=1e-7), (source, locator, name, mean, stored)
        metrics[name] = {"mean": mean, "std_population": statistics.pstdev(values), "min": min(values), "max": max(values)}
    env = context.get("evalEnv", {})
    frame_skip = raw.get("frame_skip", env.get("frame_skip"))
    max_seconds = raw.get("max_seconds", env.get("max_seconds"))
    max_steps = raw.get("max_steps", env.get("max_steps"))
    scenario = raw.get("scenario", raw.get("stage", "full"))
    game = context.get("evalGames", {}).get(scenario, {})
    fps = game.get("fps")
    if max_seconds is None and max_steps and frame_skip and fps:
        max_seconds = max_steps * frame_skip / fps
    mode = raw.get("execution_mode")
    if "deterministic" in raw:
        mode = "greedy" if raw["deterministic"] else "sampled"
    return {
        "family": source.parts[0], "experiment_id": context.get("id"),
        "experiment_folder": context.get("folder"), "kind": kind,
        "source": source.as_posix(), "copied_source": f"data/raw/{source.as_posix()}", "locator": locator,
        "label": raw.get("label"), "time": raw.get("time"),
        "training_step_counter": raw.get("global_step", raw.get("timesteps")),
        "checkpoint": raw.get("checkpoint", raw.get("policy")),
        "checkpoint_sha256": raw.get("checkpoint_sha256"),
        "scenario": scenario, "execution_mode": mode,
        "max_seconds": max_seconds, "max_steps": max_steps, "frame_skip": frame_skip,
        "seeds": [e["seed"] for e in episodes], "episode_count": len(episodes),
        "metrics": metrics,
        "episodes": [{k: e[k] for k in ("seed", "score", "survival_seconds", "terminated", "truncated") if k in e} for e in episodes],
        "decoder_status": ("corrected" if source.name.startswith("corrected_") else "historical_before_correction")
        if source.parts[0] == "gpt_training_rotation_continuous" else None,
        "included_in_plot": False,
    }


def collect():
    """원본 바이트를 스냅샷으로 보존하고 정기 평가와 별도 검증을 분리한다."""
    atlas = RL / "gpt_report/experiment_atlas.html"
    match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', atlas.read_text(encoding="utf-8"), re.S)
    records = json.loads(match[1])["records"]
    keys = ("id", "folder", "title", "created", "parents", "initial", "total", "trainEnv", "evalEnv", "evalGames", "reward", "evalReward", "flags")
    contexts = {r["folder"]: {k: r[k] for k in keys if k in r} for r in records}
    sources = {}
    regular, supplementary, duplicates = [], [], []

    def snapshot(path):
        relative = path.relative_to(RL)
        if relative.as_posix() not in sources:
            payload = path.read_bytes()
            target = OUTPUT / "data/raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == payload, f"Snapshot changed; keep this export intact and use a new output directory: {relative}"
            else:
                target.write_bytes(payload)
            sources[relative.as_posix()] = {"copy": target.relative_to(OUTPUT).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        return relative

    for family, pattern in GROUPS.items():
        for path in sorted((RL / family).glob(pattern)):
            folder = path.parent.parent if family in ("training_hurwan", "gpt_training_vector_only") else path.parent
            folder_key = folder.relative_to(RL).as_posix()
            if folder_key not in contexts:
                # shortcut: 보고서 미등록 실행은 임시 ID를 쓰며, 보고서 등록 시 공식 EXP 번호로 바꾼다.
                config = read_json(folder / "config.json")
                contexts[folder_key] = {"id": f"V{folder.name.split('_')[0]}", "folder": folder_key,
                    "title": folder.name, "evalEnv": config.get("evaluation", {}),
                    "evalGames": {"full": config.get("environment", {}).get("game", {})},
                    "report_registration": "not_registered"}
            context = contexts[folder_key]
            relative = snapshot(path)
            seen = set()
            copied_path = OUTPUT / "data/raw" / relative
            for line_number, line in enumerate(copied_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                fingerprint = json.dumps(raw, sort_keys=True)
                if fingerprint in seen:
                    duplicates.append({"source": relative.as_posix(), "line": line_number, "reason": "identical row in same source"})
                    continue
                seen.add(fingerprint)
                regular.append(normalize(raw, relative, f"line:{line_number}", context, "regular"))
            for name in ("config.json", "user_config.json", "experiment.json", "manifest.json", "state.json"):
                if (folder / name).exists():
                    snapshot(folder / name)
            if family == "training_hurwan" and (folder.parent / "model.json").exists():
                snapshot(folder.parent / "model.json")

    extra_paths = set()
    for family in GROUPS:
        if family != "training_hurwan":
            extra_paths.update((RL / family).glob("*.json"))
    extra_paths.update((RL / "gpt_training_vector_only/experiments").glob("*/confirmation.json"))
    extra_paths.update((RL / "gpt_training_vector_only/evaluations").glob("*/comparison.json"))
    for path in sorted(extra_paths):
        value = read_json(path)
        nodes = list(evaluation_nodes(value))
        if not nodes:
            if "comparison" in path.name and isinstance(value, dict) and "paired_changes" in value:
                snapshot(path)
            continue
        relative = snapshot(path)
        name = path.name.lower()
        kind = "heldout" if "heldout" in name or "unseen" in name else "confirmation" if name == "confirmation.json" else "supplementary"
        for pointer, raw in nodes:
            supplementary.append(normalize(raw, relative, f"json_pointer:{pointer}", {}, kind))

    # 서로 다른 평가 조건의 마지막 체크포인트를 각각 선택한다.
    latest = {}
    for row in regular:
        condition = (row["experiment_folder"], row["scenario"], tuple(row["seeds"]), row["max_seconds"], row["frame_skip"], row["execution_mode"])
        previous = latest.get(condition)
        if previous is None or (row["training_step_counter"] or 0) >= (previous["training_step_counter"] or 0):
            latest[condition] = row
    for row in latest.values():
        row["included_in_plot"] = True
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_definition": {"score": "arithmetic mean of episode score", "survival_seconds": "arithmetic mean of observed episode duration, seconds; time-limited episodes are right-censored"},
        "plot_selection": "last regular evaluation by checkpoint counter within each experiment and evaluation condition; grey dots show all regular checkpoint means",
        "notes": ["Means are recomputed from episodes and checked against stored means.",
            "No pooling across seed sets, scenarios, limits, or experiments; no cross-file deduplication of supplementary batches/aggregates.",
            "Missing evaluation metadata stays null; source defaults were not guessed.",
            "Training counters are local checkpoint counters, not cumulative compute across branches.",
            "Continuous-rotation regular evaluations predate the decoder correction; corrected heldout records are supplementary only.",
            "EXP019-EXP021 include imitation-learning ancestry; V008 is not yet registered in the report."],
        "experiments": list(contexts.values()), "sources": sources,
        "duplicates_removed": duplicates, "regular_evaluations": regular, "supplementary_evaluations": supplementary,
    }


def plot(data):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#98a2b3",
        "text.color": "#243047", "axes.labelcolor": "#344054", "xtick.color": "#475467", "ytick.color": "#475467"})
    fig, axes = plt.subplots(4, 2, figsize=(20, 17))
    fig.subplots_adjust(left=.07, right=.98, bottom=.105, top=.865, hspace=.68, wspace=.16)
    fig.suptitle("실험별 평균 점수와 평균 생존 시간", x=.07, y=.975, ha="left", fontsize=25, fontweight="bold")
    fig.text(.07, .942, "색 점: 실행별 마지막 정기 평가  ·  회색 점: 중간 체크포인트 평가 평균  ·  한 점은 여러 시드의 평균", fontsize=12)
    fig.text(.07, .916, "실험 순서는 분기를 포함합니다. 시드·환경·평가 시간 제한이 달라 전체 성능 순위로 해석하지 않습니다.", fontsize=11, color="#667085")
    titles = ["EXP001–018 · 기존 학습 실행", "EXP019–039 · 정책 구조 확장", "EXP041–045 · 연속 회전", "EXP047–052 + V008 · 원본 벡터 입력"]
    subtitles = ["시드 42·43·44 / 천체 전용과 전투 평가를 분리", "평가 시드 3–6개 / 초기 3개 실행은 모방학습 계열 포함", "4개 시드 · 200초 제한 / 당시 디코더로 평가", "3개 또는 20개 시드 · 120초 제한 / V008은 보고서 미등록 실행"]
    for row_index, family in enumerate(GROUPS):
        rows = [r for r in data["regular_evaluations"] if r["family"] == family]
        identifiers = list(dict.fromkeys(r["experiment_id"] for r in rows))
        for col, (metric, ylabel) in enumerate((("score", "평균 점수"), ("survival_seconds", "평균 생존 시간 (초)"))):
            ax = axes[row_index, col]
            for position, identifier in enumerate(identifiers):
                entries = [r for r in rows if r["experiment_id"] == identifier]
                for r in entries:
                    x = position
                    asteroid = r["scenario"] == "asteroid_only"
                    if family == "training_hurwan":
                        x += -.15 if asteroid else .15
                    is_last = r["included_in_plot"]
                    color = "#aeb7c5" if not is_last else "#bd701d" if asteroid else "#275d9f"
                    ax.scatter(x, r["metrics"][metric]["mean"], s=48 if is_last else 12,
                        c=color, marker="^" if asteroid else "o", alpha=1 if is_last else .38,
                        edgecolors="white" if is_last else "none", linewidths=.6, zorder=3 if is_last else 2)
            ax.set_title(titles[row_index] + "  |  " + ylabel, loc="left", fontsize=12, pad=37, fontweight="bold")
            ax.text(0, 1.065, subtitles[row_index], transform=ax.transAxes, fontsize=9, color="#667085")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(identifiers)), [s.replace("EXP", "") for s in identifiers], rotation=0)
            ax.set_xlabel("실험 번호 (EXP 접두사 생략)", fontsize=9)
            ax.set_xlim(-.65, len(identifiers)-.35)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.grid(axis="y", color="#e5e9ef", linewidth=.8, zorder=0)
            if family == "training_hurwan":
                ax.legend(handles=[Line2D([], [], marker="^", linestyle="", color="#bd701d", label="천체 전용"),
                    Line2D([], [], marker="o", linestyle="", color="#275d9f", label="전투")], loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, fontsize=9, frameon=False)
    fig.text(.07, .060, "각 행은 독립적인 세로축 범위입니다. 점수·생존 평균은 원본 에피소드에서 재계산했습니다. 회색 점은 오차범위가 아닙니다.", fontsize=10, color="#667085")
    fig.text(.07, .039, "시간 제한까지 생존한 경우 실제 최종 생존 시간은 더 길 수 있습니다. 수정 후 연속 회전 검증·별도 heldout 평가는 data/에 따로 보관했습니다.", fontsize=10, color="#667085")
    fig.savefig(OUTPUT / "evaluation_means.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    data_path = OUTPUT / "data/evaluation_data.json"
    data = read_json(data_path) if data_path.exists() else collect()
    assert data["regular_evaluations"] and all(r["episode_count"] == len(r["seeds"]) for r in data["regular_evaluations"])
    for info in data["sources"].values():
        assert hashlib.sha256((OUTPUT / info["copy"]).read_bytes()).hexdigest() == info["sha256"]
    if not data_path.exists():
        write_json(data_path, data)
    plot(data)
    print(json.dumps({"source_files": len(data["sources"]), "regular_evaluations": len(data["regular_evaluations"]),
        "supplementary_evaluations": len(data["supplementary_evaluations"]), "duplicates_removed": len(data["duplicates_removed"]),
        "plot_points": sum(r["included_in_plot"] for r in data["regular_evaluations"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
