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



class A2C:
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

    @torch.no_grad()
    def update_helper(self, state, next_state, reward, done):
        """
        Computes the td_target and advantage for policy and critic updates. No gradients are tracked.
        
        Args:
            state (torch tensor, float, (1, state_dim)): current state
            next_state (torch tensor, float, (1, state_dim)): next state
            reward (float): one-step reward
            done (boole): true iff next_state of the transition was terminal

        Returns:
            advantage (torch tensor, float, (1, 1)): r + gamma * V(s') * (1-done) - V(s)
            td_target (torch tensor, float, (1, 1)): r + gamma * V(s') * (1-done)
        """
        # defensive programming inputs        
        assert type(self.gamma) == float
        assert torch.is_floating_point(state)
        assert torch.is_floating_point(next_state)
        assert torch.is_floating_point(reward)
        assert reward.shape == (1, 1), reward.shape
        assert type(done) == torch.Tensor

        # r + gamma * (1-done) * V(s') - target for estimated Q(s, a)
        v_next = self.critic(next_state) * (1-done)
        td_target = reward + (self.gamma * v_next)

        # target for Q(s, a) - V(s)
        v_current = self.critic(state)
        advantage = td_target - v_current

        # print(td_target)
        # defensive programming outputs
        assert torch.is_floating_point(advantage)
        assert torch.is_floating_point(td_target)
        assert advantage.shape == (1, 1)
        assert td_target.shape == (1, 1)
        
        return advantage, td_target
    
    def update_policy(self, advantage, log_prob, entropy):
        """
        Computes policy loss: - adv * log_prob 
        Computes gradients
        Updates policy params

        Args:
            advantage (torch tensor float (1,1)): tensor containing estimated r + gamma*V(s') - V(s)
            log_prob (torch tensor float (1, 1)): log prob of taken the action chosen
            entropy (torch tensor float (1, 1)): entropy of prob distribution used to sample the action
        """
        # defensive programming
        assert torch.is_floating_point(advantage)
        assert torch.is_floating_point(log_prob)
        assert torch.is_floating_point(entropy)
        assert log_prob.shape == (1, 1)
        assert advantage.shape == (1, 1)
        assert entropy.shape == (1, 1)
        assert log_prob.requires_grad
        assert entropy.requires_grad

        # compute the loss: - A(s, a) * log(P(a|s)) - Beta * H(s)
        policy_loss = - (log_prob * advantage) - (self.entropy_coef * entropy)
        policy_loss = policy_loss.squeeze()
        assert policy_loss.shape == ()
        

        # update
        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 0.2)
        self.policy_optimiser.step()
        
        

    def update_critic(self, state, td_target):
        """
        Given a state and target, compute the predicted value and fit to the target

        Args:
            state (torch tensor, float, (1, state_shape)): current state
            td_target (torch tensor float): r + gamma * V(next_state), no grads estimate
        """

        # defensive programming
        assert torch.is_floating_point(state)
        assert state.shape == (1, *self.env.observation_space.shape)

        v_pred = self.critic(state)

        assert torch.is_floating_point(td_target)
        assert v_pred.shape == (1, 1)
        assert td_target.shape == (1, 1)
        assert v_pred.requires_grad
        assert not td_target.requires_grad

        critic_loss = self.critic_criterion(v_pred, td_target).squeeze()
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.2)
        self.critic_optimiser.step()

    def update(self, state, log_prob, entropy, reward, next_state, done):
        """
        One step update wrapper for policy and critic updates.
        
        0) Convert state, next_state, done to a tensor
        1) With no_grad: Compute advantage = td_target - value(s) -> advantage, td_target
        2) Policy Update: loss = - log prob * advantage - entropy * entropy_coef
        3) Critic Update(td_target, state, reward)

        Args:
            state (numpy array): current state of the environment
            log_prob (torch float tensor of shape [1,]): log probability of the chosen action
            entropy (torch float tensor of shape [1,]): entropy of the distribution used to sample the chosen action
            reward (float): one step reward from the environment
            next_state (numpy array): next state after taking the action in the current state
            done (bool): true iff next_state is terminal
        """
        # defensive programming
        assert self.env.observation_space.contains(state)
        assert self.env.observation_space.contains(next_state)
        assert torch.is_floating_point(log_prob)
        assert torch.is_floating_point(entropy)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert isinstance(reward, (int, float)), (type(reward), reward)
        assert isinstance(done, bool)
        assert log_prob.requires_grad
        assert entropy.requires_grad
        

        # Convert to tensors of shape 1,1 and on self.device
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.long, device=self.device).view(1, 1)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).view(1, 1)
        log_prob = log_prob.unsqueeze(-1)
        entropy = entropy.unsqueeze(-1)

        # Compute advantage and td_target with no grad
        advantage, td_target = self.update_helper(state, next_state, reward, done)
        
        assert not advantage.requires_grad
        assert not td_target.requires_grad

        self.update_policy(advantage, log_prob, entropy)
        self.update_critic(state, td_target)


    def run_episode(self, render=False):
        """
        The agent collects experience by simulating the environment here. 
        There are two components:
        1) Generating experience
        2) Processing the experience for learning and then calling the update

        Args:
            render (bool, optional): Dictates whether the agent's play is visualised. Defaults to False.

        Returns:
            total_reward (float): Sum of one-step rewards for the environment 
        """

        # Initialise environment
        state = self.env.reset()
        total_reward = 0.
        time_step = 0
        
        while time_step < self.step_lim:
            # Generate experience
            action, log_prob, entropy = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            time_step += 1
            if render: self.env.render()
            
            # One-step update
            self.update(state, log_prob, entropy, reward, next_state, done)
            
            if done: break
            
            state = next_state 
                
        return total_reward
    
    def train(self, num_epi, render=False):
        """Simulates agent's training phase with environment and tracks performance

        Args:
            num_epi (int): number of episodes to simulate
            render (bool, optional): Dictates if we can watch the training in real time. Defaults to False.

        Returns:
            epi_rewards (list(float)): Track of episodic rewards for training
        """
        epi_rewards = [0 for _ in range(num_epi)] 
        
        for i in range(num_epi):
            # Progress indicator
            if i % 100 == 0:
                print('Episode number: {}'.format(i))
            epi_rewards[i] = self.run_episode(render)

        return epi_rewards


# Run a simulation
if __name__ == '__main__':
    
    num_agents = 1
    num_epi = 500
    r = []    
    env = gym.make('LunarLanderContinuous-v2')
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
        agent = A2C(a2c_args)
        
        # random.seed(i)
        # np.random.seed(i)
        # torch.manual_seed(i)
        # env.seed(i)
        
        r.append(agent.train(num_epi))

    out = np.array(r).mean(0)

    plt.figure(figsize=(5, 3))
    plt.title('A2C on cartpole')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Reward')
    plt.plot(out, label='rewards')
    plt.legend()

    # plt.savefig('./data/a2c_cartpole.PNG')
    plt.show()
    