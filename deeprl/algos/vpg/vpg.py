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
    """
    This class instantiates a REINFORCE agent which can run training with a gym env
    No baseline yet.
    """
    def __init__(self, params):
        self.device = params['device']
        self.gamma = params['gamma']
        self.lr = params['policy_optimiser_lr']        
        self.env = params['env']
        self.policy = CategoricalPolicy(params['policy_layers']).to(self.device)
        self.optimiser = params['policy_optimiser'](self.policy.parameters(), self.lr)
        self.entropy_coef = params['entropy_coef']
        self.step_lim = params['step_lim']
        

    #policy net ends with a  softmax
    def choose_action(self, state): # return a numpy action
        """
        Given a state, this calls the policy net and outputs an action and corresponding log prob
        1) Convert state to float tensor on self.device with shape (1, state_dim)
        Args:
            state (numpy array): singular state taken straight from the gym environment output

        Returns:
            tuple: (action, log_prob) action is chosen action in the form of how gym represents the action, log_prob is tensor of size 1 with gradients tracked and 
        """
        assert type(state) == np.ndarray
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        action, log_prob, entropy = self.policy.sample(state) # tensors of shape [1, action_dim]
        
        action = action.cpu().squeeze().detach().numpy() # tensor of shape (action_dim,)
        
        assert self.env.action_space.contains(action), action

        return action, log_prob, entropy
    
    def _compute_loss(self, log_probs, gt_vec, entropies):
        assert type(log_probs) == torch.Tensor
        assert type(gt_vec) == torch.Tensor
        assert log_probs.shape == gt_vec.shape
        assert log_probs.dtype == gt_vec.dtype
        assert log_probs.device == gt_vec.device
        assert log_probs[3].requires_grad
        # print(log_probs[5])
        gt_vec -= gt_vec.mean()
        gt_vec /= gt_vec.std()

        loss_vec = - log_probs * gt_vec - self.entropy_coef * torch.mean(entropies)
        
        loss = torch.mean(loss_vec)
        assert type(loss) == torch.Tensor
        assert loss.shape == ()
        
        return loss

    def update(self, rewards, log_probs_a, entropies):
        """Batch update of policy.
            1) convert the inputs into tensors of shape (time_steps, 1), float dtype and on self.device
            2) Compute a vector of Gt values indexed by t for the episode
            3) Multiply the stacked log_probs by the rewards and get the mean
            4) Multiply by -1 to get the loss 
            5) Clear grads and optimiser step
        Args:
            (rewards, log_probs) where each element is a list of same length as episode

        Returns:
            bool: Always true.
        """
        assert type(rewards) == list
        assert type(log_probs_a) == list
        assert type(entropies) == list
        assert type(log_probs_a[0]) == torch.Tensor
        assert type(entropies[0]) == torch.Tensor
        assert len(rewards) == len(log_probs_a)
        assert torch.is_floating_point(log_probs_a[0])

        # Convert to tensors
        log_probs_a = torch.stack(log_probs_a).squeeze().to(self.device) # shape= (len(episode), 1)
        entropies = torch.stack(entropies).squeeze().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)

        # With no grad: compute gt ()
        gt_vec = self._compute_gt(rewards)        
        assert log_probs_a.shape == gt_vec.shape, (log_probs_a.shape, gt_vec.shape)
        
        loss = self._compute_loss(log_probs_a, gt_vec, entropies)
        
        # logging
        # self.log_probs.append(log_probs_a.detach().numpy())
        # self.losses.append(loss.cpu().detach().numpy())
        # self.gts.append(gt_vec.detach().mean())
        
        # Update
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        # print(loss)
        return True

    @torch.no_grad()
    def _compute_gt(self, rewards):
        """
        This takes a batch of rewards and computes a vector of: 
        G_t = \sum_{i=0}^{T-t} \gamma^i * r_t indexed by t

        Args:
            rewards ([float tensor, dtype is float, shapeis ]): tensor of an episode of rewards

        Returns:
            [torch float tensor on self.device]: [tensor of G_t computations indexed by t]
        """
        assert type(rewards) == torch.Tensor, rewards
        assert rewards.dtype in [torch.float, torch.float32, torch.float64]
        
        # assert rewards.dim() == 2

        # iterate backwards and add self.gamma * r
        g_t = 0.
        T = len(rewards)
        gt_vec = torch.empty_like(rewards)
        for i in reversed(range(T)):
            g_t = rewards[i] + (self.gamma * g_t)
            gt_vec[i] = g_t

        return gt_vec

    def run_episode(self, render=False):
        state = self.env.reset()
        total_reward = 0.
        i = 0
        action_log = 0
        
        log_prob_track = []
        reward_track = []
        entropy_track = []        

        while i < self.step_lim:
            action, log_prob, entropy = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            action_log += action
            i += 1

            if render:
                self.env.render()
            
            reward_track.append(reward)
            log_prob_track.append(log_prob)
            entropy_track.append(entropy)
            
            if done:
                break
            
            state = next_state
            

        assert len(reward_track) == i
        
        self.update(reward_track, log_prob_track, entropy_track)
        
        # print("Proportion of action 1: ", action_log / len(reward_track))
        # print("\n")
        return total_reward
    
    def train(self, num_epi, render=False):
        rewards = [self.run_episode(render) for _ in range(num_epi)]
        return rewards




if __name__ == '__main__':
    
    # setup params
    env = gym.make('CartPole-v1')
    input_dim = get_gym_space_shape(env.observation_space)
    output_dim = get_gym_space_shape(env.action_space)
    print(input_dim, output_dim)
    policy_layers = [
                    (nn.Linear, {"in_features": input_dim, "out_features": 64}),
                    (nn.ReLU, {}),
                    (nn.Linear, {"in_features": 64, "out_features": 64}),
                    (nn.ReLU, {}),
                    (nn.Linear, {"in_features": 64, "out_features": output_dim})
                    ]

    
    vpg_args = {'gamma': 0.99,
                'env': env,
                'step_lim': 200,
                'policy_layers': policy_layers,
                'policy_optimiser': optim.Adam,
                'policy_optimiser_lr': 0.001,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
                'entropy_coef' : 0.015
                }

    num_agents = 5
    num_epi = 300
    r = []
    losses = []
    gts = []

    # run experiment
    for i in range(num_agents):
        print("Running training for agent number {}".format(i))
        
        # random.seed(i)
        # np.random.seed(i)
        # torch.manual_seed(i)
        # cartpole_env.seed(i)

        agent = VPG(vpg_args)
        r.append(agent.train(num_epi, render=False))
        # losses.append(agent.losses)
        # gts.append(agent.gts)
    
    out = np.array(r).mean(0)
    # loss_out = np.array(losses).mean(0)
    # gt_out = np.array(gts).mean(0)

    plt.figure(figsize=(5, 3))
    plt.title('VPG on cartpole')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Reward')
    plt.plot(out, label='rewards')
    # plt.plot(loss_out, label='loss')
    # plt.plot(gt_out, label='gts')
    plt.legend()
    # plt.savefig('./data/vpg_cartpole.PNG')
    plt.show()
    
    