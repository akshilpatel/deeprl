import torch
from torch import nn
from typing import List
from abc import ABC
from torch.distributions import Categorical, Normal
import numpy as np


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
    def __init__(self, arch):
        super().__init__(arch)
    
    def sample(self, state):
        params = self.forward(state) # gives unnormalised logits
        assert type(params) == torch.Tensor
        assert params.shape == torch.Size([1, 2])
        assert torch.is_floating_point(params)

        prob_dist = Categorical(logits=params) # use logits if unnormalised,       
        
        action = prob_dist.sample()        
        
        log_prob = prob_dist.log_prob(action)

        entropy = prob_dist.entropy()

        return action, log_prob, entropy


# TODO: Add action scaling
class GaussianPolicy(Network):
    def __init__(self, arch):
        super().__init__(arch['net_design'])
    
    def sample(self, state):
        mu, log_std = self.forward(state) 
        prob_dist = Normal(mu, torch.exp(log_std)**2)
        action = prob_dist.rsample() # rsample to make sure gradient flows through dist
        log_prob = prob_dist.log_prob(action)

        return action, log_prob
