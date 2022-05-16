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
import multiprocessing as mp
from torch.distributions import Categorical, Normal
import random

from deeprl.common.utils import (
    get_gym_space_shape,
    init_envs,
    compute_gae_and_v_targets,
    net_gym_space_dims,
)
from deeprl.common.base import Network
from deeprl.algos.a2c.a2c import A2C
from deeprl.common.base import CategoricalPolicy, GaussianPolicy

torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.tensorboard import SummaryWriter
import wandb


class PPO(A2C):
    def __init__(self, params):
        self.device = params["device"]

        # RL hyperparams
        self.gamma = params["gamma"]
        self.lam = params["lam"]
        self.gamma_lam = self.gamma * self.lam
        self.norm_adv = params["norm_adv"]

        # Environment parameters
        self.multiprocess = params["multiprocess"]
        self.num_workers = params["num_workers"]
        self.env_name = params["env_name"]
        self.envs = init_envs(self.env_name, self.num_workers, self.multiprocess)

        self.state_dim = get_gym_space_shape(self.envs.single_observation_space)
        self.action_dim = get_gym_space_shape(self.envs.single_action_space)

        # Training loop hyperparameters
        self.n_interactions = params["n_interactions"]
        self.minibatch_size = params["minibatch_size"]
        self.num_train_passes = params["num_train_passes"]
        self.grad_clip_coef = params["grad_clip_coef"]
        assert (
            self.num_workers * self.n_interactions
        ) % self.minibatch_size == 0, "Minibatch_size does not divide batch_size"

        # Critic
        self.critic = params["critic"].to(self.device)
        self.critic_lr = params["critic_lr"]
        self.critic_optimiser = params["critic_optimiser"](
            self.critic.parameters(), self.critic_lr
        )
        self.critic_criterion = params["critic_criterion"]

        # Policy
        self.loss_clip_coef = params["loss_clip_coef"]
        self.entropy_coef = params["entropy_coef"]
        self.policy = params["policy"].to(self.device)
        self.policy_lr = params["policy_lr"]
        self.policy_optimiser = params["policy_optimiser"](
            self.policy.parameters(), self.policy_lr
        )

        self.current_epoch = 0

    def update_policy(self, minibatch):
        advantages = minibatch["advantages"]
        states = minibatch["states"]
        actions = minibatch["actions"]
        old_log_probs = minibatch["old_log_probs"]

        new_log_probs, entropies = self.policy.get_log_probs_and_entropies(
            states, actions
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        assert ratio.shape == (len(states), 1)
        assert ratio.shape == advantages.shape

        surr_left = ratio * advantages

        surr_right = advantages * torch.clamp(
            ratio, 1 - self.loss_clip_coef, 1 + self.loss_clip_coef
        )

        assert surr_left.shape == ratio.shape
        assert surr_right.shape == ratio.shape

        policy_loss = -torch.min(surr_left, surr_right) - (
            entropies * self.entropy_coef
        )

        policy_loss_std = policy_loss.detach().cpu().std()

        policy_loss = policy_loss.mean()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_coef)
        self.policy_optimiser.step()

        return policy_loss.item(), policy_loss_std.item()

    def get_log_probs(self, batch):
        return self.policy.get_log_prob(batch["states"], batch["actions"])

    def get_entropies(self, minibatch):
        return self.policy.get_entropies(minibatch["states"])


# Run a simulation
if __name__ == "__main__":

    envs = ["CartPole-v1", "Pendulum-v0", "LunarLanderContinuous-v2"]

    for env_name in envs:
        print(env_name)
        env = gym.make(env_name)
        state_dim = net_gym_space_dims(env.observation_space)
        action_dim = net_gym_space_dims(env.action_space)
        policy_layers = [
            (nn.Linear, {"in_features": state_dim, "out_features": 128}),
            (nn.ReLU, {}),
            (nn.Linear, {"in_features": 128, "out_features": 64}),
            (nn.ReLU, {}),
            (nn.Linear, {"in_features": 64, "out_features": action_dim}),
            (nn.ReLU, {}),
        ]

        critic_layers = [
            (nn.Linear, {"in_features": state_dim, "out_features": 128}),
            (nn.ReLU, {}),
            (nn.Linear, {"in_features": 128, "out_features": 64}),
            (nn.ReLU, {}),
            (nn.Linear, {"in_features": 64, "out_features": 1}),
        ]

        ppo_args = {
            "device": "cpu",
            "gamma": 0.99,
            "env": env,
            "batch_size": 512,
            "mb_size": 64,
            "num_train_passes": 5,
            "pi_loss_clip": 0.1,
            "lam": 0.7,
            "entropy_coef": 0.01,
            "critic": Network(critic_layers),
            "critic_lr": 0.001,
            "critic_optimiser": optim.Adam,
            "critic_criterion": nn.MSELoss(),
            "policy": CategoricalPolicy(policy_layers),
            "policy_lr": 0.003,
            "policy_optimiser": optim.Adam,
        }

        agent = PPO(ppo_args)
        agent.train(10)
