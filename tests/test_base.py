from deeprl.common.base import *
from deeprl.common.utils import net_gym_space_dims, get_gym_space_shape
from torch import nn
import gym
import pytest


ENVNAME = "LunarLander-v2"
env = gym.make(ENVNAME)


DESIGN = [
    (
        nn.Linear,
        {"in_features": net_gym_space_dims(env.observation_space), "out_features": 128},
    ),
    (nn.ReLU, {}),
    (nn.Linear, {"in_features": 128, "out_features": 64}),
    (nn.ReLU, {}),
    (
        nn.Linear,
        {"in_features": 64, "out_features": net_gym_space_dims(env.action_space)},
    ),
]


def test_network():
    # Correct outputs
    # Correct building
    # Correct device
    pass


def test_policy_init():
    pass


def test_policy_get_action():
    pass


def test_policy_get_log_prob():
    pass


def test_policy_get_entropy():
    pass


def test_categorical_policy():
    pass


def test_gaussian_policy():
    pass


def test_multi_gaussian_policy():
    pass
