import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
import multiprocessing as mp

from deeprl.common.utils import (
    compute_gae_and_v_targets,
    net_gym_space_dims,
)

from deeprl.algos.a2c.a2c import A2C
from deeprl.common.base import CategoricalPolicy, GaussianPolicy, Critic
import wandb

torch.utils.backcompat.broadcast_warning.enabled = True


class PPO(A2C):
    def __init__(self, params):

        # self.device = params["device"]

        # # RL hyperparams
        # self.gamma = params["gamma"]
        # self.lam = params["lam"]
        # self.gamma_lam = self.gamma * self.lam
        # self.norm_adv = params["norm_adv"]

        # # Environment parameters
        # self.multiprocess = params["multiprocess"]
        # self.num_workers = params["num_workers"]
        # self.env_name = params["env_name"]
        # self.envs = init_envs(self.env_name, self.num_workers, self.multiprocess)

        # # Training loop hyperparameters
        # self.n_interactions = params["n_interactions"]
        # self.minibatch_size = params["minibatch_size"]
        # self.num_train_passes = params["num_train_passes"]
        # self.grad_clip_coef = params["grad_clip_coef"]
        # assert (
        #     self.num_workers * self.n_interactions
        # ) % self.minibatch_size == 0, "Minibatch_size does not divide batch_size"

        # # Critic
        # self.critic = params["critic"].to(self.device)
        # self.critic_lr = params["critic_lr"]
        # self.critic_optimiser = params["critic_optimiser"](
        #     self.critic.parameters(), self.critic_lr
        # )
        # self.critic_criterion = params["critic_criterion"]

        # # Policy
        # self.entropy_coef = params["entropy_coef"]
        # self.policy = params["policy"].to(self.device)
        # self.policy_lr = params["policy_lr"]
        # self.policy_optimiser = params["policy_optimiser"](
        #     self.policy.parameters(), self.policy_lr
        # )

        # self.current_epoch = 0
        # self.training_interaction_step = 0

        super().__init__(params)
        self.loss_clip_coef = params["loss_clip_coef"]

    def update_policy(self, minibatch):
        advantages = minibatch["advantages"]
        states = minibatch["states"]
        actions = minibatch["actions"]
        old_log_probs = minibatch["old_log_probs"]

        new_log_probs, entropies = self.policy.get_log_probs_and_entropies(
            states, actions
        )
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()

        # # See http://joschu.net/blog/kl-approx.html
        # with torch.no_grad():
        #     approx_kl = ratio - 1 - log_ratio

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

        policy_loss = policy_loss.mean()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_coef)
        self.policy_optimiser.step()

        return policy_loss.item(), entropies.mean().item()


# Run a simulation
if __name__ == "__main__":

    env_name = "CartPole-v1"

    cp_env = gym.make(env_name)

    cp_policy_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(cp_env.observation_space),
                "out_features": 32,
            },
        ),
        (nn.Tanh, {}),
        (nn.Linear, {"in_features": 32, "out_features": 32}),
        (nn.Tanh, {}),
        (
            nn.Linear,
            {
                "in_features": 32,
                "out_features": net_gym_space_dims(cp_env.action_space),
            },
        ),
    ]

    cp_critic_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(cp_env.observation_space),
                "out_features": 32,
            },
        ),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 32, "out_features": 32}),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 32, "out_features": 1}),
    ]

    cartpole_ppo_args = {
        "gamma": 0.99,
        "env_name": "CartPole-v1",
        "step_lim": 500,
        "policy": CategoricalPolicy(cp_policy_layers),
        "policy_optimiser": optim.Adam,
        "policy_lr": 0.002,
        "critic": Critic(cp_critic_layers),
        "critic_lr": 0.002,
        "critic_optimiser": optim.Adam,
        "critic_criterion": nn.MSELoss(),
        "device": "cpu",
        "entropy_coef": 0.01,
        "n_interactions": 300,
        "num_train_passes": 4,
        "lam": 0.95,
        "num_eval_episodes": 15,
        "num_workers": 6,
        "minibatch_size": 300,
        "norm_adv": True,
        "multiprocess": False,
        "grad_clip_coef": 0.5,
        "loss_clip_coef": 0.2,
    }

    ll_env = gym.make("LunarLander-v2")

    ll_policy_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(ll_env.observation_space),
                "out_features": 128,
            },
        ),
        (nn.Tanh, {}),
        (nn.Linear, {"in_features": 128, "out_features": 64}),
        (nn.Tanh, {}),
        (
            nn.Linear,
            {
                "in_features": 64,
                "out_features": net_gym_space_dims(ll_env.action_space),
            },
        ),
    ]

    ll_critic_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(ll_env.observation_space),
                "out_features": 128,
            },
        ),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 128, "out_features": 64}),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 64, "out_features": 1}),
    ]

    lunar_lander_ppo_args = {
        "gamma": 0.99,
        "env_name": "LunarLander-v2",
        "step_lim": 500,
        "policy": CategoricalPolicy(ll_policy_layers),
        "policy_optimiser": optim.Adam,
        "policy_lr": 0.0005,
        "critic": Critic(ll_critic_layers),
        "critic_lr": 0.0005,
        "critic_optimiser": optim.Adam,
        "critic_criterion": nn.MSELoss(),
        "device": "cpu",
        "entropy_coef": 0.01,
        "n_interactions": 128,
        "num_train_passes": 10,
        "lam": 0.95,
        "num_eval_episodes": 30,
        "num_workers": 8,
        "minibatch_size": 256,
        "norm_adv": True,
        "multiprocess": False,
        "grad_clip_coef": 0.5,
        "loss_clip_coef": 0.2,
    }
    cp_env.close()
    ll_env.close()

    wandb.init(project="deeprl", name="ppo-lunarlander", entity="akshil")

    conf = lunar_lander_ppo_args
    conf["num_epochs"] = 250
    wandb.config = conf

    agent = PPO(lunar_lander_ppo_args)
    wandb.watch(agent.policy)
    wandb.watch(agent.critic)
    agent.run_training(conf["num_epochs"])

    wandb.finish()
