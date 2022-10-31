import pathlib
import os
import multiprocessing as mp


from multi_signal import MultiSignal
import argparse
from agent_config import agent_configs
from map_config import map_configs
from mdp_config import mdp_configs

import numpy as np

from collections import deque

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default='STOCHASTIC',
                    choices=[name for name in agent_configs])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--map", type=str, default='grid4x4',
                    choices=[name for name in map_configs])
    ap.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute())+os.sep)
    ap.add_argument("--log_dir", type=str, default=str(pathlib.Path().absolute())+os.sep+'logs'+os.sep)
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  # Can't multi-thread with libsumo, provide a trial number
    ap.add_argument("--out_name", type=str, default="")
    args = ap.parse_args()
    args.map = 'ingolstadt21'
    # args.out_name = args.agent + "-" + str(args.tr) + "-" + args.map
    args.trials = 1
    args.libsumo = True
    args.eps = 100


    if args.procs == 1 or args.libsumo:
        print("running trials")
        run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials+1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        print("running pool")
        pool.join()


def run_trial(args, trial):
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    agt_config = agent_configs[args.agent]
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:    # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors

    map_config = map_configs[args.map]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None: route = args.pwd + route


    env = MultiSignal(alg.__name__+'-tr'+str(trial),
                      args.map,
                      args.pwd + map_config['net'],
                      agt_config['state'],
                      agt_config['reward'],
                      route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                      step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                      max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args.gui,
                      log_dir=args.log_dir, libsumo=args.libsumo, warmup=map_config['warmup'], connection_name=args.out_name)

    agt_config['episodes'] = int(args.eps * 0.8)    # schedulers decay over 80% of steps
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = args.log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    agent = alg(agt_config, obs_act, args.map, trial)

    # import pdb
    # pdb.set_trace()

    for _ in range(args.eps):
        # create and initialise memory store
        if "GRAPH_Rec" in args.agent:
            agent_memories = {}
            deque(maxlen=10)
            for key in obs_act:
                agent_memories[key] = deque([0 for _ in range(10)], maxlen=10)

        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            # if "GRAPH_Rec" in args.agent:
            #     # update the memories
            #     for agt, action in act.items():
            #         agent_memories[agt].append(action)
            #     new_agt_obs = {}
            #     # adding to observation array
            #     for agt, memory in agent_memories.items():
            #         # store old obs
            #         agt_obs = obs[agt]
            #         # make memory into an np array
            #         memory = np.reshape(memory, (2, 5))
            #         # the append operation reshapes the array so reshape it back
            #         # import pdb
            #         # pdb.set_trace()
            #
            #         agt_obs_unshaped = np.append(agt_obs, memory)
            #
            #         agt_obs = np.reshape(
            #                 agt_obs_unshaped,
            #                 (len(agt_obs[0])+2 , 5)
            #             )
            #         # agt_obs = np.append(agt_obs, memory)
            #         new_agt_obs[agt] = agt_obs
            #     obs = new_agt_obs
            obs, rew, done, info = env.step(act)
            # print(info)
            agent.observe(obs, rew, done, info)
    env.close()


if __name__ == '__main__':
    main()
