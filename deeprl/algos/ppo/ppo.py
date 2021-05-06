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
        
    def generate_experience(self, render=False):
        # initialise vars and set done
        # while the current step isn't less than limit
            # if done: reset env and vars
            # interact with env and store (state, action, lp_old, reward, done, next_states)
            # increment current step and 
            # log everything as well.
        # What do I do for logging rewards when the agent does not complete the episode?

            
        
        pass
    
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
        states, actions, log_prob_olds, rewards, dones, next_states = list(map(self.to_torch, *zip(self.buffer)))
        
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
        assert states.shape == (self.batch_size, )

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
        assert torch.is_floating_point(states), states
        assert actions.dtype == torch.long, actions
        assert len(actions) == len(states), (states.shape, actions.shape)

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
        action = action.cpu().detach().numpy().squeeze(0)

        # defensive programming outputs
        assert self.env.action_space.contains(action), (action, action.shape, action.dtype, self.env.action_space)
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

    
    ppo_args = {'gamma': 0.99,
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
                'entropy_coef' : 0.05,
                }
        
    
    # # run experiment
    # for i in range(num_agents):
    #     print("Running training for agent number {}".format(i))
    #     agent = PPO(ppo_args)
        
    #     # random.seed(i)
    #     # np.random.seed(i)
    #     # torch.manual_seed(i)
    #     # env.seed(i)
        
    #     r.append(agent.train(num_epi))

    # out = np.array(r).mean(0)

    # plt.figure(figsize=(5, 3))
    # plt.title('PPO on cartpole')
    # plt.xlabel('Episode')
    # plt.ylabel('Episodic Reward')
    # plt.plot(out, label='rewards')
    # plt.legend()

    # # plt.savefig('./data/ppo_cartpole.PNG')
    # plt.show()
    