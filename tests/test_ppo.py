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

from deeprl.common.utils import net_gym_space_dims, discount_cumsum, get_gym_space_shape
from deeprl.common.base import Network, CategoricalPolicy, GaussianPolicy
from deeprl.algos.ppo.ppo import PPO

from torch.utils.tensorboard import SummaryWriter


# Requirement: Agent chooses a valid action in the correct shape and outputs valid log_probs and entropies
def test_choose_action():
    envs = ['CartPole-v1', 'Pendulum-v0', 'LunarLanderContinuous-v2']
    for env_name in envs: 
        print(env_name)
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
        print(state_dim, action_dim)
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
        num_steps = 300
        
        
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
            

def test_compute_lp_and_ent():
    envs = ['CartPole-v1', 'Pendulum-v0', 'LunarLanderContinuous-v2']
    for env_name in envs: 
        print(env_name)
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
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
        num_steps = 300
        
        
        
        for i in range(num_steps):
            action, log_prob, entropy = agent.choose_action(state)
            
            # Iterate
            next_state, _, done, _ = agent.env.step(action)
            if done: 
                state = env.reset()
            else:
                state = next_state


# Requirement: Agent stores transitions in the buffer correctly and outputs the correct episodic rewards
def test_generate_experience():
    envs = ['CartPole-v1', 'Pendulum-v0', 'LunarLanderContinuous-v2']
    batch_sizes = [10, 100, 512]
    for env_name in envs: 
        print(env_name)
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
        policy_layers = [
                        (nn.Linear, {'in_features': state_dim, 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': action_dim}),
                        (nn.ReLU, {})
                        ]


        critic_layers = [
                        (nn.Linear, {'in_features': state_dim , 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': 1})
                        ]
        
        for b_size in batch_sizes:
            ppo_args = {
                'device': 'cpu',
                'gamma': 0.99,
                'env': env,
                'batch_size': b_size,
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

            epi_rewards = agent.generate_experience()
            assert len(agent.buffer) == agent.batch_size, len(agent.buffer)

            to_torch = lambda x: torch.tensor(x).float().to(agent.device).squeeze()
            buffer_data = zip(*agent.buffer)
            states, actions, log_probs, rewards, dones, next_states = tuple(map(to_torch, buffer_data))


            ### TESTING ###

            # check everything is the right shape and size
            assert len(states) == agent.batch_size
            assert len(actions) == agent.batch_size
            assert log_probs.shape == (agent.batch_size,)
            assert rewards.shape == (agent.batch_size,)
            assert dones.shape == (agent.batch_size,)
            assert next_states.shape == states.shape

            assert states.device == torch.device(agent.device)
            assert rewards.device == torch.device(agent.device)
            assert dones.device == torch.device(agent.device)

            assert torch.is_floating_point(states)
            assert torch.is_floating_point(actions)
            assert torch.is_floating_point(rewards)
            assert torch.is_floating_point(dones)
            assert torch.is_floating_point(next_states)

            for i in range(len(states)):
                assert agent.env.observation_space.contains(states[i].cpu().detach().numpy()), states[i].cpu().detach().numpy()
                assert agent.env.observation_space.contains(next_states[i].cpu().detach().numpy()), next_states[i].cpu().detach().numpy()









