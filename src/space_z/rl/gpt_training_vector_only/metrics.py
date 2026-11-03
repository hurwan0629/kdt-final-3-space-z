"""학습·평가 에피소드의 원본 수치와 TensorBoard 지표를 연결한다."""


def episode_metrics(episode):
    """수량·보상·생존 시간당 비율을 서로 구분한다."""
    metrics = {
        'outcomes/score': episode['score'],
        'outcomes/survival_seconds': episode['survival_seconds'],
        'outcomes/enemies_killed': episode['kills'],
        'outcomes/shots_fired': episode['shots_fired'],
        'reward_episode/total': episode['episode_reward'],
        'termination/death': float(episode.get('terminated', not episode['truncated'])),
        'termination/time_limit': float(episode['truncated']),
    }
    minutes = max(episode['survival_seconds'], 1 / 60) / 60
    for name, value in episode['reward_components'].items():
        metrics['reward_components/' + name] = value
        metrics['reward_per_minute/' + name] = value / minutes
    for name, value in episode['reward_inputs'].items():
        metrics['reward_inputs/' + name] = value
    events = episode['telemetry']['events']
    quantities = {
        'score': episode['score'],
        'shape_damage_score_weighted': events['shape_damage_weighted'],
        'enemy_damage_hp': events['enemy_damage'],
        'damage_received_hp': events['damage_taken'],
        'shapes_destroyed': events['shapes_destroyed'],
        'enemies_killed': events['enemies_killed'],
        'shots_fired': events['player_shots_fired'],
    }
    for name, value in quantities.items():
        metrics['combat/' + name] = value
        metrics['per_minute/' + name] = value / minutes
    for name, value in episode['telemetry']['damage_steps'].items():
        metrics['damage_steps/' + name] = value
    steps = episode['steps'] if 'steps' in episode else episode['episode_steps']
    for name, value in episode['telemetry']['actions'].items():
        metrics['actions/' + name] = value / steps
    for name, value in episode['telemetry']['player'].items():
        metrics['player/' + name] = value
    return metrics


def empty_telemetry():
    return dict(events={}, damage_steps=dict(shape=0, enemy=0, received=0),
                actions=dict(move_requested_fraction=0, fire_requested_fraction=0,
                             absolute_rotation_mean=0), player={})


def collect_telemetry(telemetry, info, action):
    """이미 전달된 이벤트를 합산하며 게임 상태와 난수열은 변경하지 않는다."""
    for name, value in info['events'].items():
        telemetry['events'][name] = telemetry['events'].get(name, 0) + value
    for name, event in [('shape', 'shape_damage_weighted'), ('enemy', 'enemy_damage'), ('received', 'damage_taken')]:
        telemetry['damage_steps'][name] += int(info['events'][event] > 0)
    telemetry['actions']['move_requested_fraction'] += int(action[0] != 0)
    telemetry['actions']['fire_requested_fraction'] += int(action[2] != 0)
    telemetry['actions']['absolute_rotation_mean'] += abs(float(action[1]))
    telemetry['player'] = dict(health_end=info['player_health'], max_health_end=info['player_max_health'],
                               level_end=info['level'], upgrade_points_end=info['upgrade_points'])
