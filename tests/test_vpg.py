import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import random

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym import Space
from torchtest import assert_vars_change

from copy import deepcopy



from deeprl.common.utils import get_gym_space_shape
from deeprl.common.base import Network, CategoricalPolicy, GaussianPolicy
from deeprl.algos.vpg.vpg import VPG

from torch.distributions import Categorical, Normal

# setup params
cartpole_env = gym.make('CartPole-v1')
input_dim = get_gym_space_shape(cartpole_env.observation_space)
output_dim = get_gym_space_shape(cartpole_env.action_space)
policy_layers = [
                (nn.Linear, {"in_features": input_dim, "out_features": 64}),
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 64, "out_features": 32}),
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 32, "out_features": output_dim})
                ]


vpg_args = {'gamma': 0.99,
            'env': cartpole_env,
            'step_lim': 200,
            'policy_layers': policy_layers,
            'policy_optimiser': optim.Adam,
            'policy_optimiser_lr': 1,
            'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
            }

def test_policy_weights_change():
    cartpole_env = gym.make('CartPole-v1')
    input_dim = get_gym_space_shape(cartpole_env.observation_space)
    output_dim = get_gym_space_shape(cartpole_env.action_space)
    policy_layers = [
                (nn.Linear, {"in_features": input_dim, "out_features": 64}),
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 64, "out_features": 32}),
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 32, "out_features": output_dim})
                ]
    # policy_layers = [
    #             (nn.Linear, {"in_features": input_dim, "out_features": output_dim})
    #             ]


    vpg_args = {'gamma': 0.99,
            'env': cartpole_env,
            'step_lim': 200,
            'policy_layers': policy_layers,
            'policy_optimiser': optim.Adam,
            'policy_optimiser_lr': 1e-2,
            'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
            }
    agent = VPG(vpg_args)
    
    for _ in range(100):
        params = [p for p in agent.policy.named_parameters() if p[1].requires_grad] # tuples of names and weights

        initial_params = [ (name, p.clone()) for (name, p) in params ]

        agent.run_episode()

        for (_, p0), (name, p1) in zip(initial_params, params):
            
            assert not torch.equal(p0.to('cpu'), p1.to('cpu')), name
            
    
def test__compute_gt():
    """Requirement:
        _compute_gt: rewards -> Gt vector
        This function takes a list of floats as
    """

    env = gym.make('CartPole-v1')
    input_dim = get_gym_space_shape(env.observation_space)
    output_dim = get_gym_space_shape(env.action_space)
    policy_layers = [
                    (nn.Linear, {"in_features": input_dim, "out_features": 64}),
                    (nn.ReLU, {}),
                    (nn.Linear, {"in_features": 64, "out_features": output_dim})
                    ]

    
    vpg_args = {'gamma': 0.99,
                'env': env,
                'step_lim': 200,
                'policy_layers': policy_layers,
                'policy_optimiser': optim.Adam,
                'policy_optimiser_lr': 1e-4,
                'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                }
    
    agent = VPG(vpg_args)
    
    rewards = torch.ones((5, 1), dtype=torch.float64, device=agent.device)
    
    out = agent._compute_gt(rewards)
    

    print(out[1])


def test__compute_loss():
    assert False