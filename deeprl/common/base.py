import torch
from torch import nn
from typing import List
from abc import ABC
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from deeprl.common.utils import get_gym_space_shape, net_gym_space_dims
import torch.nn.functional as F

# TODO: Add squeezing for log_probs and entropies and make that work for code base


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


class Policy(Network):
    def get_action(self, state):
        prob_dist = self.forward(state)
        action = prob_dist.sample()
        return action

    def get_log_prob(self, states, actions):
        prob_dist = self.forward(states)
        log_prob = prob_dist.log_prob(actions)

        return log_prob

    def get_entropy(self, states):
        prob_dist = self.forward(states)
        entropy = prob_dist.entropy()

        return entropy


# TODO: change this to take a network as argument and call the net to get params,
#  and include methods for choosing an action, getting log_probs, entropies, params, and having a dist as an attribute
# Policy classes have a network, distribution type and chosoe_action method, and getters for log_prob and entropy and params
# Updates happen to policies by agents which have policies ==> update is in the agent class
class CategoricalPolicy(Policy):
    def __init__(self, arch):
        super().__init__(arch)

    def forward(self, state):
        # if not torch.is_tensor(state):
        #     state = torch.from_numpy(state.astype(np.float32))
        #     state = state.to(self.net.device)

        params = super().forward(state)

        prob_dist = Categorical(logits=params)  # use logits if unnormalised,
        return prob_dist

    def sample(self, state, action=None):
        params = super().forward(state)  # gives unnormalised logits

        prob_dist = Categorical(logits=params)  # use logits if unnormalised,

        if action is None:
            action = prob_dist.sample()

        log_prob = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()

        return action, log_prob, entropy


class GaussianPolicy(Network):
    def __init__(self, arch, action_space):
        super().__init__(arch)
        self.action_dim = net_gym_space_dims(action_space)
        hidden_dim = arch[-2][1]["out_features"]
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

        return mu, cov

    # TODO: fix this so it works with MultiNormalz
    def sample(self, state, action=None):
        params = self.forward(state)

        # turn into covariance matrix
        prob_dist = Normal(*params)

        if action is None:
            action = (
                prob_dist.rsample()
            )  # rsample to make sure gradient flows through dist

            action = action * self.action_scale.expand_as(
                action
            ) + self.action_mid.expand_as(action)
            # Same as clamp but with vector max and min - used to make sure action is within action_space
            action = torch.max(torch.min(action, self.action_high), self.action_low)

        log_prob = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()

        return action, log_prob, entropy


class MultiGaussianPolicy(GaussianPolicy):
    def __init__(self, arch, action_space):
        super().__init__(arch, action_space)

    # TODO: fix this so it works with MultiNormalz
    def sample(self, state, action=None):
        mu, cov = self.forward(state)
        cov = torch.diag(cov) * torch.eye(cov.shape[-1])

        # turn into covariance matrix
        prob_dist = MultivariateNormal(mu, cov)

        if action is None:
            action = (
                prob_dist.rsample()
            )  # rsample to make sure gradient flows through dist

            action = action * self.action_scale.expand_as(
                action
            ) + self.action_mid.expand_as(action)
            # Same as clamp but with vector max and min - used to make sure action is within action_space
            action = torch.max(torch.min(action, self.action_high), self.action_low)

        else:
            action = action.expand_as(cov)

        log_prob = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()

        return action, log_prob, entropy
