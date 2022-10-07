import numpy as np

from mdp_config import mdp_configs


def average_speed(signals):
    rewards = dict()
    for signal_id, signal in signals.items():
        cars = 0
        speed = 0
        for i, lane in enumerate(signal.lanes):
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                speed += vehicle['speed']
                cars += 1
            if cars == 0:
                rewards[signal_id] = 0
            else:
                rewards[signal_id] = speed / cars
    return rewards

def average_speed_norm(signals):
    rewards = dict()
    for signal_id, signal in signals.items():
        cars = 0
        speed = 0
        mspeed = 0
        for i, lane in enumerate(signal.lanes):
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                speed += vehicle['speed']
                cars += 1
                if mspeed > vehicle['speed']:
                    mspeed = vehicle['speed']
        if cars == 0:
            rewards[signal_id] = 0
        elif mspeed == 0 :
            rewards[signal_id] = ( speed / cars )
        else:
            rewards[signal_id] = ( speed / cars ) / mspeed
    return rewards

def mwait(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['max_wait']
        rewards[signal_id] = -total_wait
    return rewards


def wait(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']
        rewards[signal_id] = -total_wait
    return rewards


def wait_norm(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']
        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards


def pressure(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += signals[signal_id].full_observation[lane]['queue']

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= signals[signal_id].signals[dwn_signal].full_observation[lane]['queue']

        rewards[signal_id] = -queue_length
    return rewards

def queue(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += (signals[signal_id].full_observation[lane]['queue'])
        rewards[signal_id] = -queue_length
    return rewards

def queue_sq(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += (signals[signal_id].full_observation[lane]['queue'] ** 2)
        rewards[signal_id] = -queue_length
    return rewards

def pressure_sq(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += (signals[signal_id].full_observation[lane]['queue'] ** 2)

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= (signals[signal_id].signals[dwn_signal].full_observation[lane]['queue'] ** 2)

        rewards[signal_id] = -queue_length
    return rewards


def queue_maxwait(signals):
    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * 0.9)
        rewards[signal_id] = -reward
    return rewards


def queue_maxwait_neighborhood(signals):
    rewards = queue_maxwait(signals)
    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                sum_reward += (0.9 * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    return neighborhood_rewards


def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2C']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards


def fma2c_full(signals):
    fma2c_config = mdp_configs['FMA2CFull']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2CFull']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards
