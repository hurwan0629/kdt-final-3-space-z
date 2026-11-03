"""원본 게임 시간 구간별 배속 편집본을 만들고 보고서에 연결한다."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

from record_milestones import EXPERIMENTS, VIDEOS, read_json, save_manifest, publish

SPEED_SEGMENTS = (
    ((0, 5, 1), (5, None, 10)),
    ((0, 5, 1), (5, None, 10)),
    ((0, 5, 1), (5, None, 10)),
    # 기존 구간별 배속에 추가 3배속을 곱한다.
    ((0, 45, 9), (45, None, 15)),
    ((0, 75, 9), (75, 165, 30), (165, None, 9)),
)
ENDING_SECONDS = 3
CANDIDATE_SPEED_SEGMENTS = {
    4: ((0, 45, 3), (45, None, 5)),
    5: ((0, 75, 3), (75, 165, 10), (165, None, 3)),
}


def edit(clip, segments, ffmpeg, destination=None):
    source = VIDEOS / clip['file']
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != clip['video_sha256']:
        raise ValueError(f'Original video changed: {source.name}')
    destination = destination or VIDEOS / f'{source.stem}_edited.mp4'
    if not destination.resolve().is_relative_to(VIDEOS.resolve()):
        raise ValueError('Edited videos must stay inside videos/')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.mp4')
    ending = VIDEOS / 'endings' / f"{clip['number']:03d}.png"
    if not ending.is_file():
        raise ValueError(f'Ending image missing: {ending}')
    overlay = VIDEOS / 'overlays' / f"{clip['number']:03d}.png"
    if not overlay.is_file() or 'total_training_steps' not in clip:
        raise ValueError('Prepare the cumulative-step diagram overlay before editing')
    filters = [f'[0:v]split={len(segments)}' + ''.join(f'[s{i}]' for i in range(len(segments)))]
    duration = 0
    for i, (start, end, speed) in enumerate(segments):
        end = clip['video_duration'] if end is None else end
        if not 0 <= start < end <= clip['video_duration'] or speed <= 0:
            raise ValueError('Invalid source-time segment')
        filters.append(f'[s{i}]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/{speed}[v{i}]')
        duration += (end - start) / speed
    filters.append(''.join(f'[v{i}]' for i in range(len(segments))) + f'concat=n={len(segments)}:v=1:a=0,fps=30,setsar=1,format=yuv420p[body]')
    filters.append(f'[1:v]trim=duration={ENDING_SECONDS},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[ending]')
    filters.append('[body][ending]concat=n=2:v=1:a=0[presentation]')
    filters.append('[presentation][2:v]overlay=0:0:eof_action=repeat:format=auto,format=yuv420p[out]')
    duration += ENDING_SECONDS
    process = subprocess.run([str(ffmpeg), '-y', '-v', 'error', '-i', str(source),
                              '-loop', '1', '-framerate', '30', '-t', str(ENDING_SECONDS), '-i', str(ending),
                              '-i', str(overlay),
                              '-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '1',
                              '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-progress', 'pipe:1', str(partial)],
                             capture_output=True, text=True, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(process.stderr)
    frames = int([line.split('=')[1] for line in process.stdout.splitlines() if line.startswith('frame=')][-1])
    if abs(frames / 30 - duration) > 0.12:
        raise ValueError(f'Unexpected edited duration: {frames/30} vs {duration}')
    partial.replace(destination)
    return dict(file=destination.relative_to(VIDEOS).as_posix(), source_sha256=source_hash,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                segments=[dict(start=start, end=end, speed=speed) for start, end, speed in segments],
                expected_seconds=duration, duration_seconds=frames/30, frames=frames,
                ending_seconds=ENDING_SECONDS, ending_image=ending.relative_to(VIDEOS).as_posix(),
                ending_sha256=hashlib.sha256(ending.read_bytes()).hexdigest(),
                ending_reason='game_over' if clip['terminated'] else 'time_limit',
                diagram_overlay=True, overlay_image=overlay.relative_to(VIDEOS).as_posix(),
                overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest(),
                displayed_training_steps=clip['total_training_steps'],
                time_reference='original video seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    if not args.ffmpeg.is_file():
        parser.error('Provide an existing ffmpeg executable')
    manifest = read_json(VIDEOS / 'manifest.json')
    if [c['experiment'] for c in manifest['clips']] != list(EXPERIMENTS):
        raise ValueError('Expected the agreed milestone order')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(edit, clip, segments, args.ffmpeg) for clip, segments in zip(manifest['clips'], SPEED_SEGMENTS)]
        for clip, job in zip(manifest['clips'], jobs):
            clip['edited'] = job.result()
            save_manifest(manifest)
            print(f"{clip['edited']['file']}: {clip['edited']['duration_seconds']:.2f}s", flush=True)
    for clip in manifest['clips']:
        if clip['number'] not in CANDIDATE_SPEED_SEGMENTS:
            continue
        destination = VIDEOS / '후보' / f"{Path(clip['file']).stem}_before_extra_3x.mp4"
        clip['candidate_before_extra_3x'] = edit(
            clip, CANDIDATE_SPEED_SEGMENTS[clip['number']], args.ffmpeg, destination,
        )
        save_manifest(manifest)
        print(f"{clip['candidate_before_extra_3x']['file']}: candidate saved", flush=True)
    publish(manifest)


if __name__ == '__main__':
    main()
