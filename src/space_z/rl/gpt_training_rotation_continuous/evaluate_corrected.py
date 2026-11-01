"""수정된 디코더에서 고정 후보와 기준 모델을 새 seed로 병렬 검증한다."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

from .train import ROOT, evaluate, write_json


def configure_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def evaluate_checkpoint(checkpoint_path, seeds, output_name):
    from stable_baselines3 import PPO
    from .policy import RotationContinuousPolicy
    checkpoint = Path(checkpoint_path)
    output = ROOT / output_name
    if output.exists():
        raise FileExistsError(output)
    model = PPO.load(checkpoint, device='cpu')
    decoder_source = (ROOT.parent / 'actions/rotation_continuous_action_decoder.py'
                      if isinstance(model.policy, RotationContinuousPolicy)
                      else ROOT.parent / 'gpt_training/environment.py')
    report = dict(checkpoint=str(checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  decoder_source=str(decoder_source),
                  decoder_source_sha256=hashlib.sha256(decoder_source.read_bytes()).hexdigest(),
                  **evaluate(model, seeds, 400, deterministic=True))
    write_json(output, [report])
    return dict(output=output_name, summary=report['summary'])


def main():
    protocol = json.loads((ROOT / 'decoder_correction_protocol.json').read_text(encoding='utf-8'))
    selection = json.loads((ROOT / 'corrected_candidate_selection.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT / 'selected_model.zip').read_bytes()).hexdigest() == selection['shipped_sha256']
    assert len(protocol['heldout_seeds']) == 20
    jobs = [
        (ROOT.parent / 'gpt_training/pure_ppo_model.zip', 'corrected_heldout_baseline.json'),
        (ROOT / 'selected_model.zip', 'corrected_heldout_candidate.json'),
    ]
    # Coordinator 2개 + worker 최대 4개를 예약하며 추출기나 학습 worker를 중첩하지 않는다.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=configure_worker) as pool:
        futures = [pool.submit(evaluate_checkpoint, str(path), protocol['heldout_seeds'], output)
                   for path, output in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()
