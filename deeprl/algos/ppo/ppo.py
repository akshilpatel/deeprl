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

from deeprl.common.utils import get_gym_space_shape, discount_cumsum
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
            print(action.shape)

            # Storage
            self.buffer.append([state, action, lp_old.detach(), reward, done, next_state])
            
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
        return torch.tensor(x).float().to(self.device)
    
    @torch.no_grad()
    def compute_gaes(self):
        
        # call critic to get a forward pass on 
        # compute deltas = r + gamma * values[:-1] - values[1:]
        # call discount_cumsum on deltas, gamma_lam and dones

        return advs

    @torch.no_grad()
    def preprocess_buffer(self):
        
        # take the contents out of buffer and zip them to get vectors of s, a, lp, r, d, ns,
        # convert these to tensors
        states, actions, log_prob_olds, rewards, dones, next_states = list(map(self.to_torch, zip(*self.buffer)))
        
        #reshape
        
        # compute value for first state 
        # compute next_state_values for next_states * (1-dones)
        # concat to get all values
        ## OR##
        # concat states to last state
        # insert a 0 at the beginning of dones
        # call critic to get the values and multpily by (1-dones)
        
        # compute adv
        advs = self.compute_gaes(rewards, values_all, dones) # fix this
        
        # using old critic estimation of values for target
        v_targets = advs + values_all[:-1] # use the current state values
        
        # make sure shapes are right       
        assert states.shape == (self.batch_size,) 

        return states, actions, log_prob_olds, rewards, dones, next_states, advs, v_targets

    def clear_buffer(self):
        self.buffer.clear()
    
    def update(self):
        # Call preprocess buffer - states, actions, rewards, log_prob_olds, next_states, 
        # For num_learning_iterations:
            # random sample indices to make a minibatch of s, a, r, ... 
            # compute current log_prob_currents and entropies: S(float), A(long) -> torch_float_like_rewards, torch_float_like_rewards 
            # Compute policy loss and update: log_probs_currents, log_prob_olds, entropies, advs, 
            # Compute value loss and update: states, targets
            # logging the losses, entropies, values, 
        pass

    def update_policy(self, states, actions, log_prob_olds, advs):
        """Compute the policy loss and update the policy given the required no_grad computations

        Args:
            states (torch Tensor): shape: (mb_size, state_dims), dtype: float
            actions (torch Tensor): shape: (mb_size, action_dims), dtype: float 
            log_prob_olds (torch Tensor): shape: (mb_size,), dtype: float, no_gradients
            advs (torch Tensor): shape: (mb_size,), dtype: float, no_gradients
        """
        # defensive programming
        # Get the current log_prob for a|s - with gradients
            # call sample on the policy with args: states, actions:
        # Get the current entropy - with gradients
        # Compute ratio = exp(log_prob_curr - log_prob_old) - grads for log_prob_curr
        # standardise adv
        # Compute g = torch.where(advs >= 0, (1+self.clip)*adv, (1-self.clip)*adv)
        # Compute loss = - min(ratio*Adv, g) - entropy * entropy_coeff
        # clear optimiser, call backward and 
        # log the policy loss, ration, advantage, entropy and log_prob

    def compute_lp_and_ent(self, states, actions):
        """Gets policy distribution statistics for a given batch of state-action pairs
        Args:
            states (torch.Tensor): shape: (batch_size, *state_shape), dtype: float
            actions (torch.Tensor): shape: (batch_size, *action_shape), dtype: float (gets converted to long if needed by Categorical)

        Returns:
            log_probs_new (torch.Tensor): log_prob for current policy distribution for state-action. shape: (batch_size,), dtype: float
            entropies_new (torch.Tensor): entropy for current policy distribution at state. shape: (batch_size,), dtype: float
        """
        assert torch.is_floating_point(states), states
        assert torch.is_floating_point(actions), actions
        assert len(actions) == len(states), (states.shape, actions.shape)
        assert action.device == self.device

        # call sample on policy with the actions specified
        _, log_probs_new, entropies_new = self.policy.sample(states, actions)

        assert len(log_probs_new) == len(actions)
        assert len(entropies_new) == len(actions)

        return log_probs_new, entropies_new


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
        
        for i_epoch in range(num_epochs):
            print('Starting Epoch {}'.format(i_epoch))
            self.generate_experience()
            self.update()
            self.clear_buffer()
        return episodic_reward
    
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
        action = action.cpu().detach().numpy().view(self.action_dim)

        # defensive programming outputs
        assert self.env.action_space.contains(action), action
        assert log_prob.shape == (1,), log_prob
        assert entropy.shape == (1,), entropy

        return action, log_prob, entropy

  
# Run a simulation
if __name__ == '__main__':
    
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
    