import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from collections import deque
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym import Space

from copy import deepcopy
from typing import List, Tuple, Dict

import random

from deeprl.common.utils import get_gym_space_shape
from deeprl.common.base import Network
import multiprocessing as mp
from torch.distributions import Categorical, Normal
from deeprl.common.base import CategoricalPolicy, GaussianPolicy



class PPO:
    def __init__(self, params):
        self.device = params['device']
        self.gamma = params['gamma']       
        self.env = params['env']
        self.step_lim = params['step_lim']

        # Critic
        self.critic = params['critic'].to(self.device)
        self.critic_lr = params['critic_lr']
        self.critic_optimiser = params['critic_optimiser'](self.critic.parameters(), self.critic_lr)
        self.critic_criterion = params['critic_criterion']

        # Policy
        self.policy = params['policy'].to(self.device)
        print(self.policy)
        self.policy_lr = params['policy_lr'] 
        self.policy_optimiser = params['policy_optimiser'](self.policy.parameters(), self.policy_lr)
        self.entropy_coef = params['entropy_coef']

    
    def choose_action(self, state):
        """Calls the policy network to sample an action for a given state. The log_prob of the action and the entropy of the distribution are also recorded for updates.

        Args:
            state (numpy array): current state of the environment

        Returns:
            action (numpy array, (gym action_dim))
            torch float tensor (1, 1): log probability of policy distribution used to sample the action
            torch float tensor (1, 1): entropy of policy distribution used to sample the action
        """
        # defensive programming inputs
        assert self.env.observation_space.contains(state), state

        state = torch.tensor([state], dtype=torch.float, device=self.device)
        action, log_prob, entropy = self.policy.sample(state)

        # Convert to gym action and squeeze to deal with discrete action spaces
        action = action.cpu().detach().numpy().squeeze()

        # defensive programming outputs
        assert self.env.action_space.contains(action), action
        assert log_prob.shape == (1,), log_prob
        assert entropy.shape == (1,), entropy

        return action, log_prob, entropy

  
# Run a simulation
if __name__ == '__main__':
    
    num_agents = 1
    num_epi = 500
    r = []    
    env = gym.make('CartPole-v1')
    policy_layers = [
                    (nn.Linear, {'in_features': get_gym_space_shape(env.observation_space), 'out_features': 128}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 128, 'out_features': 64}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 64, 'out_features': get_gym_space_shape(env.action_space)}),
                    (nn.ReLU, {})
                    ]


    critic_layers = [
                    (nn.Linear, {'in_features': get_gym_space_shape(env.observation_space), 'out_features': 128}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 128, 'out_features': 64}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 64, 'out_features': 1})
                    ]

    
    a2c_args = {'gamma': 0.99,
                'env': env,
                'step_lim': 200,
                'policy': GaussianPolicy(policy_layers, env.action_space),
                'policy_optimiser': optim.Adam,
                'policy_lr': 0.00075,
                'critic' : Network(critic_layers),
                'critic_lr': 0.00075,
                'critic_optimiser': optim.Adam,
                'critic_criterion': nn.MSELoss(),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
                'entropy_coef' : 0.05
                }
        
    
    # run experiment
    for i in range(num_agents):
        print("Running training for agent number {}".format(i))
        agent = PPO(ppo_args)
        
        # random.seed(i)
        # np.random.seed(i)
        # torch.manual_seed(i)
        # env.seed(i)
        
        r.append(agent.train(num_epi))

    out = np.array(r).mean(0)

    plt.figure(figsize=(5, 3))
    plt.title('PPO on cartpole')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Reward')
    plt.plot(out, label='rewards')
    plt.legend()

    # plt.savefig('./data/ppo_cartpole.PNG')
    plt.show()
    