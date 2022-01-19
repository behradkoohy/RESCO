from typing import Any, Sequence
import numpy as np
# pytorch
import torch
import torch.nn as nn
# reinforcement learning library
import pfrl
from pfrl import explorers, replay_buffers
from pfrl.explorer import Explorer
from pfrl.agents import DQN
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.initializers import init_chainer_default
from pfrl.utils.contexts import evaluating

# local imports
from agents.agent import IndependentAgent, Agent
from agents.models.gRNN_concat import gRNNConc
from agents.graph import SharedDQN

# debug imports
import pdb

# needed deque from collections
from collections import deque

# adding memory to our agents

class Graph_RIDQN(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)

        deque_len=10

        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            k_size = 4
            def conv2d_size_out(size, kernel_size=k_size, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1
            # pdb.set_trace()
            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])
            
            # model = nn.Sequential(
            #     gRNNConc(h, w, obs_space, act_space, k_size),
            #     init_chainer_default(nn.Linear(64, act_space)),
            #     DiscreteActionValueHead()
            #     )
            model = pfrl.nn.RecurrentSequential(
                nn.Conv2d(obs_space[0], 64, kernel_size=(k_size, k_size)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(h * w * 64, 64),
                nn.ReLU(),
                nn.LSTM(input_size=3136, hidden_size=512),
                nn.Linear(512, act_space),
                DiscreteActionValueHead(),
            )
            self.agents[key] = DQNAgent(config, act_space, model)

class DQNAgent(Agent):
    def __init__(self, config, act_space, model, num_agents=0):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        replay_buffer = replay_buffers.EpisodicReplayBuffer(10000)

        if num_agents > 0:
            explorer = SharedEpsGreedy(
                config['EPS_START'],
                config['EPS_END'],
                num_agents*config['steps'],
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = explorers.LinearDecayEpsilonGreedy(
                config['EPS_START'],
                config['EPS_END'],
                config['steps'],
                lambda: np.random.randint(act_space),
            )

        if num_agents > 0:
            print('USING SHAREDDQN')
            self.agent = SharedDQN(self.model, self.optimizer, replay_buffer,
                                   config['GAMMA'], explorer, gpu=self.device.index,
                                   minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                   phi=lambda x: np.asarray(x, dtype=np.float32),
                                   target_update_interval=config['TARGET_UPDATE']*num_agents, update_interval=num_agents)
        else:
            self.agent = DQN(self.model, self.optimizer, replay_buffer, config['GAMMA'], explorer,
                             gpu=self.device.index,
                             minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                             phi=lambda x: np.asarray(x, dtype=np.float32),
                             target_update_interval=config['TARGET_UPDATE'],
                             recurrent=True)

    def act(self, observation, valid_acts=None, reverse_valid=None):
        if isinstance(self.agent, SharedDQN):
            return self.agent.act(observation, valid_acts=valid_acts, reverse_valid=reverse_valid)
        else:
            return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        if isinstance(self.agent, SharedDQN):
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')