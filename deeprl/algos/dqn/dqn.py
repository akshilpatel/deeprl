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


class DQN:
    def __init__(self, kwargs):
        # algo params
        self.gamma = kwargs['gamma']
        self.lr = kwargs['lr']
        self.epsilon = kwargs['epsilon']
        self.eps_decay_rate = kwargs['eps_decay_rate']
        self.mb_size = kwargs['mb_size']
        self.polyak_w = kwargs['polyak_w']
        
        # env params
        self.step_lim = kwargs['step_lim']
        self.env = kwargs['env']
        
        self.device = kwargs['device']
        self.memory = Memory(kwargs['memory_max_len'], self.device)
        self.q = Network(kwargs['net_design'])  # use for action selection (nograd) and learning(grad)
        self.q.to(self.device)
        self.target_q = deepcopy(self.q)
        self.target_q.eval()
        self.target_update_freq = kwargs['target_update_freq']
        self._target_timer = 0
        self.criterion = kwargs['criterion']
        self.optimiser = kwargs['optimiser'](self.q.parameters(), self.lr)

    @torch.no_grad()
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
            # print('random_action')
        else:
            state = torch.tensor([state], dtype=torch.float, device=self.device)  # now passing through network, so should be float really
            
            self.q.eval()
            # convert to numpy so you can do random choice of max vals
            q_vals = self.q(state).cpu().detach().numpy().squeeze()  # (1, num_ac) do i need to detach given that i have already no_grad?
            self.q.train()
    
            greedy_actions = np.where(q_vals == q_vals.max())[0]
            action = np.random.choice(greedy_actions)
            # print(q_vals, greedy_actions, action)
        assert self.env.action_space.contains(action)
        return action
                 
    def _q_update(self):
        states, actions, rewards, dones, next_states = self.memory.sample(self.mb_size)
        rewards = rewards.squeeze(0)
        
        # Online Q net - track this for backprop
        state_q_vals = self.q(states)
        
        # actions are converted to ints and used as indices, then the whole thing is squeezed to match targets shape for loss
        q_preds = torch.gather(state_q_vals, -1, actions.long().unsqueeze(-1))
        q_preds.squeeze_(-1)
        
        assert q_preds.shape == (self.mb_size,)
    
        # Target Q net
        with torch.no_grad():
            nxt_tgt_net_vals = self.target_q(next_states)  # shape:(m_batch, n_actions)
            assert nxt_tgt_net_vals.shape == (self.mb_size, 2)
            
            # Get the max over columns and take only the values - not indices
            max_tgt_net_vals = torch.max(nxt_tgt_net_vals, dim=1)[0]  # shape: (m_batch, 1)
#             max_tgt_net_vals.unsqueeze_(-1)
            
            assert max_tgt_net_vals.shape == (self.mb_size,), max_tgt_net_vals.shape  # does this shape work for subtraction/addition?
            assert max_tgt_net_vals.shape == rewards.shape, rewards.shape
            assert max_tgt_net_vals.shape == dones.shape, dones.shape
            
            q_targets = rewards + self.gamma * (1 - dones) * max_tgt_net_vals  # check
            
            assert q_preds.shape == q_targets.shape
            # print(q_preds.shape)
        
        # learn
        self.optimiser.zero_grad()
        loss = self.criterion(q_preds, q_targets.detach())  # maybe unsqueeze inputs here?
        loss.backward()
        self.optimiser.step()
        
        return True
        
    # Tested
    def _eps_decay(self):
        if self.eps_decay_rate != 1:
            self.epsilon *= self.eps_decay_rate
        return True
    
    # TODO: Test this works and there are no grads?
    @torch.no_grad()
    def _target_update(self):
        self.target_q.load_state_dict(self.q.state_dict())
        return True
    
    # TODO: Check this is complete and test it
    def update(self):
        """ 
        Wrapper function around the component updates
        """
        self._eps_decay()
        
        self._target_timer += 1
        if self._target_timer % self.target_update_freq == 0:
            self._target_update()
        
        # If the agent has collected enough experience
        if len(self.memory) > self.mb_size:
            self._q_update()
        
        return True
    
    # TODO: Comment this and test
    def run_episode(self, render=False):
        state = self.env.reset()
        total_reward = 0
        i = 0
        
        while i < self.step_lim:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            
            if render:
                self.env.render()
            
            self.memory.store((state, action, reward, done, next_state))
            self.update()
                
            if done:
                break
            state = next_state
            i += 1
        return total_reward
    
    def train(self, num_epi, render=False):
        rewards = [self.run_episode(render) for _ in range(num_epi)]
        return rewards


if __name__ == '__main__':
    
    cartpole_env = gym.make('CartPole-v1')
    env2 = gym.make('BipedalWalker-v3')
    input_dim = get_gym_space_shape(cartpole_env.observation_space)
    output_dim = get_gym_space_shape(cartpole_env.action_space)
    net_layers = [(nn.Linear, {"in_features": input_dim, "out_features": 32}),
                  (nn.ReLU, {}),
                  (nn.Linear, {"in_features": 32, "out_features": 16}),
                  (nn.ReLU, {}),
                  (nn.Linear, {"in_features": 16, "out_features": output_dim})]

    
    dqn_args = {'gamma': 0.99,
                'epsilon': 1.,
                'eps_decay_rate': 0.95,
                'env': cartpole_env,
                'step_lim': 200,
                'mb_size': 32,
                'net_design': net_layers,
                'optimiser': optim.Adam,
                'lr': 0.01,
                'polyak_w': 1.,
                'memory_max_len': 10000,
                'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'criterion': nn.MSELoss(),
                'target_update_freq': 32            
                }
    agent = DQN(dqn_args)

    random.seed(1)
    np.random.seed(1)
    r = agent.train(200)
    print(r)
    plt.figure(figsize=(16,12))
    plt.plot(r)
    plt.show()
    print(agent.epsilon)