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
from deeprl.common.replay_buffers import Memory
import multiprocessing as mp
from torch.distributions import Categorical, Normal
from deeprl.common.base import CategoricalPolicy, GaussianPolicy


class VPG:
    def __init__(self, params):
        self.device = params['device']
        self.gamma = params['gamma']
        self.lr = params['policy_optimiser_lr']        
        self.env = params['env']
        self.policy = CategoricalPolicy(params['policy_layers'])
        self.policy.to(self.device)
        self.optimiser = params['policy_optimiser'](self.policy.parameters(), self.lr)
        

    #policy net ends with a  softmax
    def choose_action(self, state): # return a numpy action
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        action, log_prob = self.policy.sample(state)# tensors of shape [1, action_dim]
        return action.cpu().squeeze().numpy(), log_prob
    

    def run_episode(self, render=False):
        total_reward = 0
        reward_track = []
        log_prob_track = []
        state = self.env.reset()
        
        while True:
            action, log_prob = self.choose_action(state) # is action the right type and shape to go into all envs?
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            reward_track.append(reward)
            log_prob_track.append(log_prob)
            
            state = next_state # make sure this doesn't screw with mutable states
        
            if done:
                break
        
        self.update((reward_track, log_prob_track))

        return total_reward

    def update(self, experience):
        # Unpack experience and make sure they are all torch tensors
        rewards, log_probs = experience
        log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(-1)
        
        # With no grad: compute gt ()
        gt_vec = self._compute_gt(rewards)
        
        assert log_probs.shape == gt_vec.shape, (log_probs.shape, gt_vec.shape)
        
        
        self.optimiser.zero_grad()
        loss = -torch.sum(log_probs * gt_vec)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1., -1.)
        self.optimiser.step()
        
        # with grads:
        #   check that the log_probs are the same size as the gt and then do a multiply
        #   loss = - reduce sum of the output
        #   backwards, clip grads, step
        return True

    @torch.no_grad()
    def _compute_gt(self, rewards):
        # iterate backwards and add self.gamma * 
        g = 0
        g_vec = torch.zeros_like(rewards)
        for i, r in enumerate(reversed(rewards)):
            g *= self.gamma 
            g += r
            g_vec[i] = g
        return g_vec

    def train(self, num_epi, render=False):
        rewards = [self.run_episode(render) for _ in range(num_epi)]
        return rewards




if __name__ == '__main__':
    
    # setup params
    cartpole_env = gym.make('CartPole-v1')
    input_dim = get_gym_space_shape(cartpole_env.observation_space)
    output_dim = get_gym_space_shape(cartpole_env.action_space)
    policy_layers = [(nn.Linear, {"in_features": input_dim, "out_features": 64}),
                  (nn.ReLU, {}),
                  (nn.Linear, {"in_features": 64, "out_features": 32}),
                  (nn.ReLU, {}),
                  (nn.Linear, {"in_features": 32, "out_features": output_dim})]

    
    vpg_args = {'gamma': 0.99,
                'env': cartpole_env,
                'step_lim': 200,
                'policy_layers': policy_layers,
                'policy_optimiser': optim.Adam,
                'policy_optimiser_lr': 0.001,
                'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                }
    agent = VPG(vpg_args)

    num_agents = 10
    num_epi = 100
    r = []

    # run experiment
    for i in range(num_agents):
        print("Running training for agent number {}".format(i))
        random.seed(i)
        np.random.seed(i)
        agent = VPG(vpg_args)
        r.append(agent.train(num_epi))
    out = np.array(r).mean(0)
    plt.figure(figsize=(16,12))
    plt.plot(out)
    plt.show()
    plt.savefig('./data/vpg_cartpole.PNG')
    plt.close()