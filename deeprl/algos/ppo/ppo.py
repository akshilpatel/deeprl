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

from deeprl.common.utils import get_gym_space_shape, discount_cumsum, net_gym_space_dims
from deeprl.common.base import Network
import multiprocessing as mp
from torch.distributions import Categorical, Normal
from deeprl.common.base import CategoricalPolicy, GaussianPolicy
from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, params):
        self.device = params['device']
        self.gamma = params['gamma']       
        self.env = params['env']
        self.action_dim = get_gym_space_shape(self.env.action_space)
        self.batch_size = params['batch_size']
        self.mb_size = params['mb_size']
        self.num_train_passes = params['num_train_passes']
        self.pi_loss_clip = params['pi_loss_clip']
        self.lam = params['lam']
        self.gamma_lam = self.gamma * self.lam
        self.entropy_coef = params['entropy_coef']
        

        # Critic
        self.critic = params['critic'].to(self.device)
        self.critic_lr = params['critic_lr']
        self.critic_optimiser = params['critic_optimiser'](self.critic.parameters(), self.critic_lr)
        self.critic_criterion = params['critic_criterion']

        # Policy
        self.policy = params['policy'].to(self.device)
        self.policy_lr = params['policy_lr'] 
        self.policy_optimiser = params['policy_optimiser'](self.policy.parameters(), self.policy_lr)
        
        self.buffer = deque(maxlen=self.batch_size)     
        
    # TODO: Add logging
    # TODO: Currently this resets env even if the last call of generate_experience didn't terminate the episode - fix this
    def generate_experience(self, render=False):
        """Interact with environment to produce rollout over multiple episodes if necessary.

        Simulates agents interaction with gym env, stores as tuple (s, a, lp, r, d, ns)


        Args:
            render (bool, optional): Whether or not to visualise the interaction. Defaults to False.

        Returns:
            episodic_rewards (list): scalar rewards for each episode that the agent completes.
        """

        interaction_step = 0
        episodic_rewards = []
        state = self.env.reset()
        done = False
        curr_epi_reward = 0.

        # while the current step isn't less than limit
        while interaction_step < self.batch_size:
           
            # Interaction 
            action, lp_old, _ = self.choose_action(state) # TODO: Unpack ent here when you add logging
            next_state, reward, done, _ = self.env.step(action)
            interaction_step += 1
            curr_epi_reward += reward
            
            # Storage
            self.buffer.append([state, action.reshape(self.action_dim), lp_old.detach(), reward, done, next_state])
            
            if render: self.env.render()

            # Dealing with episode end
            if done:
                episodic_rewards.append(curr_epi_reward)
                state = self.env.reset()
                curr_epi_reward = 0.
            
            else:
                state = next_state # TODO: Might have to add copying at some point if this extends to different envs
        
        return episodic_rewards

    def to_torch(self, x):
        """Helper fn for converting things to torch tensors

        Args:
            x (iterable): contains the thing to be converted to tensor (batch_size, data_dims)

        Returns:
            torch.Tensor : [description]
        """
        return torch.tensor(x).float().to(self.device).squeeze()
    
    @torch.no_grad()
    def compute_gaes(self, rewards, all_values, dones):

        # compute deltas = r + gamma * values[:-1] - values[1:]
        deltas = rewards + self.gamma * all_values[:-1] - all_values[1:]
        
        advs = discount_cumsum(deltas, dones, self.gamma_lam)
        advs = torch.from_numpy(advs).to(self.device)
        return advs

    @torch.no_grad()
    def preprocess_buffer(self):
        
        # take the contents out of buffer and zip them to get vectors of s, a, lp, r, d, ns,
        # convert these to tensors
        states, actions, log_prob_olds, rewards, dones, next_states = list(map(self.to_torch, zip(*self.buffer)))
        
        #reshape
        first_state_value = self.critic(states[0])
        next_state_values = self.critic(next_states) * (1-dones)
        all_values = torch.stack([first_state_value, next_state_values])

        # compute adv
        advs = self.compute_gaes(rewards, all_values, dones) # fix this
        
        # using old critic estimation of values for target
        v_targets = advs + all_values[:-1] # use the current state values
        
        # make sure shapes are right       
        return states, actions, log_prob_olds, advs, v_targets

    def clear_buffer(self):
        self.buffer.clear()
    
    def update(self):
        # Call preprocess buffer - states, actions, rewards, log_prob_olds, next_states, 
        states, actions, log_prob_olds, advs, v_targets = self.preprocess_buffer()
        poss_idc = random.shuffle(range(len(states)))
        sample_idc = poss_idc

        for train_i in range(self.num_train_passes):
            # random sample indices to make a minibatch of s, a, r, ... 
            sample_idc = random.sample(poss_idc, self.mb_size)
            s_sample = states[sample_idc]
            a_sample = actions[sample_idc]
            lp_old_sample = log_prob_olds[sample_idc]
            adv_sample = advs[sample_idc]
            v_target_sample = v_targets[sample_idc]

            policy_loss = self.update_policy(s_sample, a_sample, lp_old_sample, adv_sample)
            critic_loss = self.update_critic(s_sample, v_target_sample)

    def update_policy(self, states, actions, log_probs_old, advs):
        """Compute the policy loss and update the policy given the required no_grad computations

        Args:
            states (torch Tensor): shape: (mb_size, state_dims), dtype: float
            actions (torch Tensor): shape: (mb_size, action_dims), dtype: float 
            log_probs_old (torch Tensor): shape: (mb_size,), dtype: float, no_gradients
            advs (torch Tensor): shape: (mb_size,), dtype: float, no_gradients
        """
        # defensive programming

        # standardise advantage
        advs -= advs.mean()
        advs /= advs.std()

        # Get the current log_prob for a|s - with gradients
        log_probs_new, entropies_new = self.policy.sample(states, actions)
        
        assert log_probs_new.shape == (len(actions),)
        assert entropies_new.shape == (len(actions),)
        assert torch.is_floating_point(log_probs_new)
        assert torch.is_floating_point(entropies_new)
        assert log_probs_new.requires_grad
        assert entropies_new.requires_grad

        # Compute ratio = exp(log_prob_curr - log_prob_old) - grads for log_prob_curr
        ratio = torch.exp(log_probs_new - log_probs_old)        
        g = torch.where(advs >= 0, (1 + self.pi_loss_clip) * advs, (1 - self.pi_loss_clip) * advs)
        policy_loss = - torch.min(ratio* advs, g) - entropies_new.mean() * self.entropy_coef

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()
        
        return policy_loss.item()

    def update_critic(self, states, v_targets):
        """Minibatch update for the critic given a sample of states and corresponding targets

        Args:
            states (torch Tensor): shape: (mb_size, state_dim), dtype: float
            v_targets (torch Tensor): shape: (mb_size, 1), dtype: float, gradients: false
        """
        # defensive programming
        # forward pass on states to get preds
        # compute loss(preds, v_targets).mean()
        # clear optim, call backward, clip grads, take a step
        v_preds = self.critic(states)
        critic_loss = self.critic_criterion(v_preds, v_targets)
        
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()
        return critic_loss.item()

    def train(self, num_epochs, render=False):
        """Interface function which is used to run the agent's learning

        Args:
            num_epochs (int): Number of times to iterate through the process of generating and learning
            render (bool, optional): Whether or not to render the agents interactions when generating experience. Used for debugging mostly. Defaults to False. 

        Returns:
            numpy.ndarray : array containing episodic rewards
        """
        # for each epoch
        # call generate - log rewards
        # call update
        # clear buffer
        episodic_rewards = []
        for i_epoch in range(num_epochs):
            print('Starting Epoch {}'.format(i_epoch))
            r = self.generate_experience()
            episodic_rewards += r
            self.update()
            self.clear_buffer()
        return episodic_rewards
    
    def choose_action(self, state):
        """Calls the policy network to sample an action, corresponding log_prob and entropy for a given state. 
        
        Args:
            state (numpy array): current state of the environment

        Returns:
            action (numpy array, (gym action_dim))
            torch float tensor (1,): log probability of policy distribution used to sample the action, gradients:True
            torch float tensor (1,): entropy of policy distribution used to sample the action, gradients:True
        """
        # defensive programming inputs
        assert self.env.observation_space.contains(state), state

        # unsqueeze to make it a batch of size 1
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        
        action, log_prob, entropy = self.policy.sample(state)
        # print(action)

        # Convert to gym action and squeeze to deal with discrete action spaces
        action = action.cpu().detach().numpy().squeeze(0)
        # defensive programming outputs
        assert self.env.action_space.contains(action), (action, action.shape, self.env.action_space)
        assert log_prob.shape == (1,), log_prob
        assert entropy.shape == (1,), entropy

        return action, log_prob, entropy

  
# Run a simulation
if __name__ == '__main__':
    
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
    