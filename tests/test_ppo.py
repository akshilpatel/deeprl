from copy import deepcopy
from typing import List, Tuple, Dict
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym import Space

import torch
from torch import nn, optim
from torch.distributions import Categorical, Normal



import random

from deeprl.common.utils import get_gym_space_shape, discount_cumsum
from deeprl.common.base import Network, CategoricalPolicy, GaussianPolicy
from deeprl.algos.ppo.ppo import PPO

from torch.utils.tensorboard import SummaryWriter


# Requirement: Agent chooses a valid action in the correct shape and outputs valid log_probs and entropies
def test_choose_action():
    envs = ['CartPole-v1', 'Pendulum-v0', 'LunarLanderContinuous-v2']
    for env_name in envs: 
        print(env_name)
        env = gym.make(env_name)
        state_dim = get_gym_space_shape(env.observation_space)
        action_dim = get_gym_space_shape(env.action_space)
        policy_layers = [
                        (nn.Linear, {'in_features': state_dim, 'out_features': 128}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 128, 'out_features': 64}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 64, 'out_features': action_dim}),
                        (nn.ReLU, {})
                        ]


        critic_layers = [
                        (nn.Linear, {'in_features': state_dim , 'out_features': 128}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 128, 'out_features': 64}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 64, 'out_features': 1})
                        ]

    
        ppo_args = {
            'device': 'cpu',
            'gamma': 0.99,
            'env': env,
            'batch_size': 512,
            'mb_size': 64,
            'num_train_passes': 5,
            'pi_loss_clip': 0.1,
            'lam': 0.7,
            'entropy_coef': 0.01,
            'critic': Network(critic_layers),
            'critic_lr': 0.001,
            'critic_optimiser': optim.Adam,
            'critic_criterion': nn.MSELoss(),
            'policy': CategoricalPolicy(policy_layers) if isinstance(env.action_space, Discrete) else GaussianPolicy(policy_layers, env.action_space),
            'policy_lr': 0.003,
            'policy_optimiser': optim.Adam
        }   

        agent = PPO(ppo_args)
        
        # random.seed(1)
        # torch.manual_seed(1)
        # np.random.seed(1)
        # env.manual_seed(1)
        

        state = agent.env.reset()
        num_steps=1000
        
        
        for i in range(num_steps):
            action, log_prob, entropy = agent.choose_action(state)
            

            # Testing # 
            assert agent.env.action_space.contains(action), (action, i)
            
            assert torch.is_floating_point(log_prob), (log_prob, i)
            assert log_prob.item() < 0., (log_prob, i)
            assert 0 <= torch.exp(log_prob).item() <= 1., (log_prob, i)
            assert log_prob.requires_grad, (log_prob, i)
            assert log_prob.shape == (1,), (log_prob, i)
            
            assert torch.is_floating_point(entropy), (entropy, i)
            assert entropy.item() >= 0., (entropy, i)
            assert entropy.shape == (1,), (entropy, i)

            # Iterate
            next_state, _, done, _ = agent.env.step(action)
            if done: 
                state = env.reset()
            else:
                state = next_state
            

# Requirement: Agent simulates in 
def test_generate_experience():
    pass



