"""보관한 이전 보상 코드와 지연 계산의 결과·환경 step 시간을 비교한다."""
import sys
import time
import types
from zipfile import ZipFile

import numpy as np

from .environment import make_environment
from .score_reward import ScoreReward, reward_settings
from .train import ROOT, write_json


def main():
    module = types.ModuleType('archived_score_reward')
    sys.modules[module.__name__] = module
    with ZipFile(ROOT / 'experiments/008_full_score_v3/source_code.zip') as archive:
        source = archive.read('score_reward.py').decode('utf-8')
    exec(compile(source, 'archived_score_reward.py', 'exec'), module.__dict__)
    actions = np.random.default_rng(42).integers(0, [9, 9, 2, 6, 5], size=(300, 5))

    def measure(reward):
        environment = make_environment('full', 300, pure_rl=True, reward_profile='score_v3')
        try:
            environment.score_reward = reward
            observation, _ = environment.reset(seed=42)
            records = []
            started = time.perf_counter()
            for action in actions:
                observation, value, terminated, truncated, _ = environment.step(action)
                records.append((observation, value, terminated, truncated))
                if terminated or truncated:
                    break
            return time.perf_counter()-started, records
        finally:
            environment.close()

    runs = []
    for repeat in range(3):
        old_seconds, old_records = measure(module.ScoreReward(module.reward_settings('score_v3')))
        new_seconds, new_records = measure(ScoreReward(reward_settings('score_v3')))
        assert len(old_records) == len(new_records)
        for old, new in zip(old_records, new_records):
            np.testing.assert_array_equal(old[0], new[0])
            assert old[2:] == new[2:]
        errors = [abs(old[1]-new[1]) for old, new in zip(old_records, new_records)]
        assert max(errors) < 1e-6
        runs.append({'previous_seconds': old_seconds, 'lazy_seconds': new_seconds,
                     'maximum_reward_difference': max(errors)})
    report = {'steps_per_run': len(old_records), 'runs': runs,
              'median_speedup': float(np.median([run['previous_seconds']/run['lazy_seconds'] for run in runs])),
              'maximum_reward_difference': max(run['maximum_reward_difference'] for run in runs),
              'scope': 'same seed/actions, default static full-game terrain; measures one environment, not total PPO throughput'}
    write_json(ROOT / 'reward_benchmark.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
