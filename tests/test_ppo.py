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
from deeprl.common.base import *
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
            # TODO: test episodic rewards has length equal to sum(dones)



# Requirement: Accurate computation of gaes over multiple episodes outputting correct shape, no grads, correct dtype stuff. 
def test_compute_gaes():
    env = gym.make('CartPole-v1')
    state_dim = net_gym_space_dims(env.observation_space)
    action_dim = net_gym_space_dims(env.action_space)
    policy_layers = [
                    (nn.Linear, {'in_features': state_dim, 'out_features': 32}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 32, 'out_features': action_dim}),
                    (nn.ReLU, {})
                    ]

    critic_layers = [
                    (nn.Linear, {'in_features': state_dim , 'out_features': 32}),
                    (nn.ReLU, {}),
                    (nn.Linear, {'in_features': 32, 'out_features': 1})
                    ]

    devices = ['cpu', 'cuda']
    
    for dev in devices:
        ppo_args = {
            'device': dev,
            'gamma': 0.99,
            'env': env,
            'batch_size': 15,
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

        r1 = torch.tensor([1 for _ in range (4)]).float().view(-1,).to(dev)
        dones10 = torch.zeros_like(r1).float().to(dev)
        dones_rand1 = torch.tensor([0, 1, 0, 1]).float().view(-1,).to(dev)
        all_values1 = torch.tensor([1.2, -3.4, 0., 4.2, 0.]).float().view(-1,).to(dev)        

        assert len(all_values1) == len(r1) + 1
        assert len(dones_rand1) == len(r1) 

        r2 = torch.tensor([-1, 2, 4.5, 7.9, -12., 200.]).view(-1,).to(dev)        
        dones_rand2 = torch.tensor([0, 0, 1, 0, 1, 0]).float().view(-1,).to(dev)
        dones20 = torch.zeros_like(r2).float().to(dev)
        all_values2 = torch.tensor([1.2, -3.48, 2.01, 0., 0.3, 0., -4.01]).float().view(-1,).to(dev)

        assert len(all_values2) == len(r2) + 1
        assert len(dones_rand2) == len(r2) 


        out_r1_d10 = agent.compute_gaes(r1, all_values1, dones10)
        
        out_r1_d_rand = agent.compute_gaes(r1, all_values1, dones_rand1)
       
        out_r2_d20 = agent.compute_gaes(r2, all_values2, dones20)
        out_r2_d2_rand = agent.compute_gaes(r2, all_values2, dones_rand2)

        print(out_r2_d20)
        assert torch.allclose(out_r1_d10, torch.tensor([ 0.8953,  6.4377,  2.9404, -3.2]).to(dev), 1e-4), out_r1_d10.shape
        assert torch.allclose(out_r1_d_rand, torch.tensor([-0.5168,  4.4,  2.9404, -3.2]).to(dev))
        assert torch.allclose(out_r2_d20, torch.tensor([31.950518, 54.250676, 67.504729, 93.816349, 123.54885, 196.0301]).to(dev))
        assert torch.allclose(out_r2_d2_rand, torch.tensor([0.7272607, 9.19547,  2.49, -0.3269, -12.3, 196.0301]).to(dev))

        assert not out_r1_d10.requires_grad
        

def test_preprocess_buffer():

    envs = ['CartPole-v1', 'LunarLanderContinuous-v2', 'Pendulum-v0']
    
    for i, env_name in enumerate(envs):
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
        policy_layers = [
                        (nn.Linear, {'in_features': state_dim, 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': action_dim}),
                        (nn.ReLU, {})
                        ]

        critic_layers = [
                        (nn.Linear, {'in_features': state_dim , 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': 1})
                        ]
        policies = [CategoricalPolicy(policy_layers),  MultiGaussianPolicy(policy_layers, env.action_space), GaussianPolicy(policy_layers, env.action_space)]
        ppo_args = {
            'device': 'cpu',
            'gamma': 0.99,
            'env': env,
            'batch_size': 300,
            'mb_size': 64,
            'num_train_passes': 5,
            'pi_loss_clip': 0.1,
            'lam': 0.7,
            'entropy_coef': 0.01,
            'critic': Network(critic_layers),
            'critic_lr': 0.001,
            'critic_optimiser': optim.Adam,
            'critic_criterion': nn.MSELoss(),
            'policy': policy_layers[i],
            'policy_lr': 0.003,
            'policy_optimiser': optim.Adam
        }   

        agent = PPO(ppo_args)        
        agent.generate_experience()
        states, actions, log_prob_olds, advs, v_targets = agent.preprocess_buffer()

        assert states.shape == (agent.batch_size, *agent.state_dim)
        assert actions.shape == (agent.batch_size, *agent.action_dim)
        assert log_prob_olds.shape == (agent.batch_size,)
        assert v_targets.shape == (agent.batch_size,)
        assert not v_targets.requires_grad
        assert not log_prob_olds.requires_grad
        assert not advs.requires_grad

def test_compute_policy_loss():
    envs = ['CartPole-v1', 'LunarLanderContinuous-v2', 'Pendulum-v0']
    for env_name in envs:
        print(env_name)
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
        print('State_dim', state_dim)
        print('action_dim', action_dim)
        policy_layers = [
                        (nn.Linear, {'in_features': state_dim, 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': action_dim}),
                        (nn.ReLU, {})
                        ]

        critic_layers = [
                        (nn.Linear, {'in_features': state_dim , 'out_features': 32}),
                        (nn.ReLU, {}),
                        (nn.Linear, {'in_features': 32, 'out_features': 1})
                        ]
        
        ppo_args = {
            'device': 'cpu',
            'gamma': 0.99,
            'env': env,
            'batch_size': 300,
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
        agent.generate_experience()
        states, actions, log_probs_old, advs, _ = agent.preprocess_buffer()
        
        print(agent.compute_policy_loss(states, actions, log_probs_old, advs))