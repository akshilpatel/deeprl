import torch
from torch import nn
from typing import List, Tuple
from abc import ABC
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from deeprl.common.utils import (
    get_gym_space_shape,
    net_gym_space_dims,
    layer_init,
)
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, design: List):
        super().__init__()
        self.layers_list = []

        # Define each layer according to arch.
        for i in range(len(design)):
            layer_type, params = design[i]
            layer = layer_type(**params)

            self.layers_list.append(layer)

        self.net = nn.Sequential(*self.layers_list)

    def forward(self, x):
        out = self.net(x)
        return out


class Critic(Network):
    def __init__(self, design: List):
        super().__init__(design)
        self.net.apply(layer_init)
        layer_init(self._modules["net"][-1], weight_std=1.0)


class Policy(Network):
    def __init__(self, design: List):
        super().__init__(design)
        self.net.apply(layer_init)
        layer_init(self._modules["net"][-1], weight_std=0.01)

    def get_action(self, state):
        prob_dist = self.forward(state)
        action = prob_dist.sample()
        action = action.cpu().detach().numpy().squeeze()
        return action

    def get_log_prob(self, states, actions):
        prob_dist = self.forward(states)
        log_prob = prob_dist.log_prob(actions)
        log_prob = log_prob.unsqueeze(-1)

        return log_prob

    def get_entropy(self, states):
        prob_dist = self.forward(states)
        entropy = prob_dist.entropy()
        entropy = entropy.unsqueeze(-1)

        return entropy

    def get_log_probs_and_entropies(self, states, actions):

        log_probs = self.get_log_prob(states, actions)
        entropies = self.get_entropy(states)

        return log_probs, entropies

    def get_modal_action(self, states):
        prob_dist = self.forward(states)
        modal_action = prob_dist.mode()
        return modal_action.cpu().detach().numpy().squeeze()


# TODO: change this to take a network as argument and call the net to get params,
#  and include methods for choosing an action, getting log_probs, entropies, params, and having a dist as an attribute
# Policy classes have a network, distribution type and chosoe_action method, and getters for log_prob and entropy and params
class CategoricalPolicy(Policy):
    def forward(self, state):

        params = super().forward(state)

        prob_dist = Categorical(logits=params)  # use logits if unnormalised,
        return prob_dist


class GaussianPolicy(Policy):
    def __init__(self, arch, action_space):
        super().__init__(arch)
        self.action_dim = net_gym_space_dims(action_space)
        hidden_dim = arch[-2][1][
            "out_features"
        ]  # Assuming that the penultimate layer is not an activation.
        self.mu_fc = nn.Linear(hidden_dim, self.action_dim)
        self.cov_fc = nn.Linear(hidden_dim, self.action_dim)

        self.action_space = action_space
        self.action_low = torch.from_numpy(action_space.low)
        self.action_high = torch.from_numpy(action_space.high)
        self.action_scale = self.action_high - self.action_low
        self.action_mid = (self.action_high + self.action_low) / 2.0

    def forward(self, x):
        h_out = super().forward(x)

        mu = torch.tanh(self.mu_fc(h_out))
        log_cov = F.softplus(self.cov_fc(h_out))
        cov = torch.exp(log_cov) ** 2

        # print(cov.shape)
        # turn into square covariance matrix

        assert mu.shape == (len(x), self.action_dim), mu.shape
        # assert cov.shape == (len(x), self.action_dim), cov.shape
        assert torch.all(cov >= 0)

        # turn into covariance matrix
        prob_dist = Normal(mu, cov)
        return prob_dist

    # TODO: fix this so it works with MultiNormalz
    def get_action(self, state):
        prob_dist = self.forward(state)
        action = prob_dist.rsample()  # rsample to make sure gradient flows through dist

        action = action * self.action_scale.expand_as(
            action
        ) + self.action_mid.expand_as(action)
        # Same as clamp but with vector max and min - used to make sure action is within action_space
        action = torch.max(torch.min(action, self.action_high), self.action_low)

        return action


class MultiGaussianPolicy(GaussianPolicy):
    def __init__(self, arch, action_space):
        super().__init__(arch, action_space)

    # TODO: fix this so it works with MultiNormalz
    def sample(self, state, action=None):
        mu, cov = self.forward(state)
        cov = torch.diag(cov) * torch.eye(cov.shape[-1])

        # turn into covariance matrix
        prob_dist = MultivariateNormal(mu, cov)

        action = prob_dist.rsample()  # rsample to make sure gradient flows through dist

        action = action * self.action_scale.expand_as(
            action
        ) + self.action_mid.expand_as(action)
        # Same as clamp but with vector max and min - used to make sure action is within action_space
        action = torch.max(torch.min(action, self.action_high), self.action_low)

        return action


class DeterministicPolicy(GaussianPolicy):
    pass
