import torch
from torch import nn
from typing import List
from abc import ABC
from torch.distributions import Categorical, Normal

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

class Agent(ABC):
    def __init__(self):
        pass

    def run_episode(self, render):
        pass        

    def train(self, num_epi, render=False):
        pass


# TODO: add action scaling
class CategoricalPolicy(Network):
    def __init__(self, arch):
        super().__init__(arch)
    
    def sample(self, state):
        logits = self.forward(state) # gives log logits right?
        prob_dist = Categorical(logits=logits) # use logits if unnormalised, 
        action = prob_dist.sample()
        log_prob = prob_dist.log_prob(action)

        return action, log_prob

# TODO: add action scaling
class GaussianPolicy(Network):
    def __init__(self, arch):
        super().__init__(arch['net_design'])
    
    def sample(self, state):
        mu, sigma = self.forward(state) 
        prob_dist = Normal(mu, sigma) 
        action = prob_dist.sample()
        log_prob = prob_dist.log_prob(action)

        return action, log_prob
