import numpy as np
import math
from mdp_config import mdp_configs



def get_lane_obs(signal, lane, act_index, i):
    lane_obs = []
    if i == act_index:
        lane_obs.append(1)
    else:
        lane_obs.append(0)

    lane_obs.append(signal.full_observation[lane]['approach'] / 28)
    lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
    lane_obs.append(signal.full_observation[lane]['queue'] / 28)

    total_speed = 0
    vehicles = signal.full_observation[lane]['vehicles']
    for vehicle in vehicles:
        total_speed += (vehicle['speed'] / 20 / 28)
    lane_obs.append(total_speed)

    return lane_obs


def graph(signals):
    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        # The node making the decisions
        for i, lane in enumerate(signal.lanes):
            # Information for the current lane
            obs.append(get_lane_obs(signal, lane, act_index, i))

        # Include information about surrounding nodes
        for neighbours in set(signal.out_lane_to_signalid.values()):
            # Select phase, collecting the relevant information from neighbours
            neighbour_obs = []
            nsignal = signals[neighbours]
            for i, lane in enumerate(nsignal.lanes):
                neighbour_obs.append(get_lane_obs(nsignal, lane, nsignal.phase, i))
            # Reduce Phase
            # _NO_ reduction happens in this variant
            # Connect Phase
            [obs.append(x) for x in neighbour_obs]
        
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)

    return observations

def graph_pooled(signals):
    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        # The node making the decisions
        for i, lane in enumerate(signal.lanes):
            # Information for the current lane
            obs.append(get_lane_obs(signal, lane, act_index, i))

        # Include information about surrounding nodes
        set_neighbours = set(signal.out_lane_to_signalid.values())
        for neighbours in set_neighbours:
            # Select phase, collecting the relevant information from neighbours
            neighbour_obs = []
            nsignal = signals[neighbours]
            for i, lane in enumerate(nsignal.lanes):
                lane_obs = [float(x)/len(set_neighbours) for x in get_lane_obs(nsignal, lane, nsignal.phase, i)]
                neighbour_obs.append(lane_obs)
            # Reduce Phase
            # We just normalise by multiplying by 1/root(2)

            # Connect Phase
            [obs.append(x) for x in neighbour_obs]

        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)

    return observations


def graph_pool_norm(signals):
    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        # The node making the decisions
        for i, lane in enumerate(signal.lanes):
            # Information for the current lane
            obs.append(get_lane_obs(signal, lane, act_index, i))

        # Include information about surrounding nodes
        for neighbours in set(signal.out_lane_to_signalid.values()):
            # Select phase, collecting the relevant information from neighbours
            neighbour_obs = []
            nsignal = signals[neighbours]
            for i, lane in enumerate(nsignal.lanes):
                neighbour_obs.append(get_lane_obs(nsignal, lane, nsignal.phase, i))
            # Reduce Phase
            # _NO_ reduction happens in this variant
            # Connect Phase
            [obs.append(x) for x in neighbour_obs]

        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)

    return observations


def graph_redux(signals, adjs=False):
    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        # The node making the decisions
        for i, lane in enumerate(signal.lanes):
            # Information for the current lane
            obs.append(get_lane_obs(signal, lane, act_index, i))

        # Include information about surrounding nodes
        for neighbours in set(signal.out_lane_to_signalid.values()):
            # Select phase, collecting the relevant information from neighbours
            neighbour_obs = []
            nsignal = signals[neighbours]
            for i, lane in enumerate(nsignal.lanes):
                neighbour_obs.append(get_lane_obs(nsignal, lane, nsignal.phase, i))
            # Reduce Phase
            # TODO: implement different options here
            # Connect Phase
            [obs.append(x) for x in neighbour_obs]
        if adjs:
            observations[signal_id+"_adj"] = set(signal.out_lane_to_signalid.values())
        
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)

    return observations


def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'])
            lane_obs.append(signal.full_observation[lane]['total_wait'])
            lane_obs.append(signal.full_observation[lane]['queue'])

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += vehicle['speed']
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'] / 28)
            lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
            lane_obs.append(signal.full_observation[lane]['queue'] / 28)
            # import pdb
            # pdb.set_trace()
            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    import pdb
    pdb.set_trace()
    return observations


def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations

def mplight_advanced(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        junctions = {x:y for x,y in signal.full_observation.items() if x not in ('num_vehicles', 'arrivals', 'departures')}
        vehicles = []
        for id, details in junctions.items():
            vehicles.append([s['speed'] for s in details['vehicles']])
            # speeds = [ob['speed'] for ob in vehicles]
        # print(vehicles)
        fastest_vehicles = [max(l, default=0) for l in vehicles]
        effective_range = [x*200 for x in fastest_vehicles]
        effective_running_vehicles = sum(effective_range)
        # print(effective_running_vehicles)
        obs = [signal.phase, effective_running_vehicles]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            upcount = 0
            for lane in signal.lane_sets[direction]:
                upcount += 1
                queue_length += signal.full_observation[lane]['queue']

            # Subtract downstream
            downstream_length = 0
            downcount = 0
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    downcount += 1
                    downstream_length += signal.signals[dwn_signal].full_observation[lane]['queue']
            if downcount == 0 and upcount == 0:
                queue_length = 0

            elif upcount == 0:
                queue_length = 0 - (downstream_length/downcount)

            elif downcount == 0:
                queue_length = (queue_length/upcount)
                
            else:
                queue_length = (queue_length/upcount) - (downstream_length/downcount)

            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'] / 28)
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                tot_approach += (signal.full_observation[lane]['approach'] / 28)

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations


def wave(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = 0
            for lane in signal.lane_sets[direction]:
                wave_sum += signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations


def ma2c(signals):
    ma2c_config = mdp_configs['MA2C']

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / ma2c_config['norm_wave'], 0, ma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(ma2c_config['coop_gamma'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / ma2c_config['norm_wait'], 0, ma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    fma2c_config = mdp_configs['FMA2CFull']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)

            waves.append(signal.full_observation[lane]['total_wait'] / 28)
            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations
