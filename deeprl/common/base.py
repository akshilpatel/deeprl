import torch
from torch import nn
from typing import List
from abc import ABC
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from deeprl.common.utils import get_gym_space_shape
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, design: List):
        super().__init__()
        self.layers = []

        # Define each layer according to arch.
        for i in range(len(design)):
            layer_type, params = design[i]
            self.layers.append(layer_type(**params))

        self.net = nn.Sequential(*self.layers)
        
    def forward(self, x):
        assert type(x) == torch.Tensor
        assert x.dim() > 1
        assert x.dtype in [float, torch.float32, torch.float64]
        out = self.net(x)
        return out

# TODO: check gradients computed through here properly
class CategoricalPolicy(Network):
    def __init__(self, arch, action_space):
        super().__init__(arch)
    
    def sample(self, state):
        params = self.forward(state) # gives unnormalised logits
        assert type(params) == torch.Tensor
        assert torch.is_floating_point(params)

        prob_dist = Categorical(logits=params) # use logits if unnormalised,       
        
        action = prob_dist.sample()        
        log_prob = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()
        
        return action, log_prob, entropy


# TODO: Add action scaling
class GaussianPolicy(Network):
    def __init__(self, arch, action_space):
        super().__init__(arch)
        
        self.action_dim = get_gym_space_shape(action_space)
        hidden_dim = arch[-2][1]['out_features']
        self.mu_fc = nn.Linear(hidden_dim, self.action_dim)
        self.cov_fc = nn.Linear(hidden_dim, self.action_dim)

        self.dist = MultivariateNormal
        self.action_space = action_space
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = self.action_high - self.action_low
        self.action_mid = (self.action_high + self.action_low)/2.

    def forward(self, x):
        h_out = super().forward(x)

        mu = torch.tanh(self.mu_fc(h_out))
        log_cov = F.softplus(self.cov_fc(h_out))
        cov = torch.exp(log_cov) ** 2

        assert mu.shape == (1, self.action_dim)
        assert cov.shape == (1, self.action_dim)
        assert torch.all(cov >= 0)

        return mu, cov

    # TODO: fix this so it works with MultiNormal
    def sample(self, state):
        params = self.forward(state)
        prob_dist = self.dist(*params)
        
        action = prob_dist.rsample() # rsample to make sure gradient flows through dist

        action = torch.clamp((action * self.action_scale) + self.action_mean, self.action_low, self.action_max)

        log_prob = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()
        
        return action, log_prob, entropy

