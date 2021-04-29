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



def test_init():
    assert False

def test_run_episode():
    assert False

def test_update():
    assert False

def test_compute_policy_loss():
    assert False

def test_compute_critic_loss():
    assert False


    