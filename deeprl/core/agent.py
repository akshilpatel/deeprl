from abc import abstractmethod
import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.multiprocessing as mp

from typing import List, Tuple, Dict

from deeprl.common.utils import (
    net_gym_space_dims,
    discount_cumsum,
    compute_gae_and_v_targets,
    to_torch,
    minibatch_split,
    normalise_adv,
    init_envs,
)
from deeprl.common.base import Network, Critic, CategoricalPolicy, GaussianPolicy
from torch.utils.tensorboard import SummaryWriter
import wandb


class Agent:
    @abstractmethod
    def __init__(self, *args):
        raise NotImplemented("")

    def run_eval_episode(self, step_lim=np.inf, render=False):
        """Functionality for running an episode with updating.

        Args:
            step_lim (int, optional): Maximum number of steps to limit each episode.
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False. Defaults to False.

        Returns:
            float: Reward accumulated during the episode.
        """
        env = gym.make(self.env_name)
        state = env.reset()

        episodic_reward = 0.0
        step_counter = 0
        while step_counter < step_lim:

            action = self.choose_action(np.expand_dims(state, 0))

            next_state, reward, done, _ = env.step(action)

            episodic_reward += reward
            step_counter += 1

            if render:
                env.render()

            if done:
                break

            state = next_state

        env.close()
        info = {"eval_epi_len": step_counter, "eval_epi_reward": episodic_reward}
        return episodic_reward, info

    def run_eval(self, num_episodes: int = 10, step_lim=np.inf, render: bool = False):
        """This is a wrapper method around the `run_episode` method to run several episodes in evaluation.

        Args:
            num_episodes (int): The number of episodes for which to run training.
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False.
            verbose (bool, optional): Used to decide if we should . Defaults to True.

        Returns:
            numpy array: Array of shape (num_episodes,) which contains the episodic rewards returned by the `run_episode` method.
        """
        self.policy.eval()
        self.critic.eval()
        # with mp.Pool(self.num_workers) as pool:
        #     epi_rewards = pool.map(self.run_eval_episode, (step_lim, render))

        epi_rewards, epi_lengths = list(
            zip(*[self.run_eval_episode(step_lim, render) for _ in range(num_episodes)])
        )

        eval_log = {
            "epoch": self.current_epoch,
            "evaluation/mean_epi_reward": np.mean(epi_rewards),
            "evaluation/std_epi_reward": np.std(epi_rewards),
            "evaluation/max_epi_reward": max(epi_rewards),
            "evaluation/min_epi_reward": min(epi_rewards),
            "evaluation/mean_epi_length": np.mean(epi_lengths),
            "evaluation/std_epi_length": np.std(epi_lengths),
            "evaluation/max_epi_length": max(epi_lengths),
            "evaluation/min_epi_length": min(epi_lengths),
        }
        wandb.log(eval_log)

        self.policy.train()
        self.critic.train()
        return np.mean(epi_rewards)

    def update_critic(self, minibatch):
        states = minibatch["states"]
        v_targets = minibatch["v_targets"]

        assert not v_targets.requires_grad

        v_preds = self.critic(states)

        critic_loss = self.critic_criterion(v_preds, v_targets)
        critic_loss_std = critic_loss.detach().std()  # For logging

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_coef)
        self.critic_optimiser.step()
        return critic_loss.item(), critic_loss_std.item()

    def update_policy(self, minibatch):

        advantages = minibatch["advantages"]
        states = minibatch["states"]
        actions = minibatch["actions"]

        log_probs, entropies = self.policy.get_log_probs_and_entropies(states, actions)

        policy_loss = -advantages * log_probs - entropies * self.entropy_coef

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_coef)
        self.policy_optimiser.step()

        return policy_loss.mean().item(), entropies.mean().item()

    @torch.no_grad()
    def choose_action(self, state):
        """Calls the policy network to sample an action for a given state. The log_prob of the action and the entropy of the distribution are also recorded for updates.

        Args:
            state (numpy array): current state of the environment

        Returns:
            numpy array (gym action_dim): action selected
        """

        state = torch.tensor(state, dtype=torch.float, device=self.device)

        if self.num_workers == 1:
            state = state.unsqueeze(0)

        action = self.policy.get_action(state)

        return action

    def generate_rollout(self):
        experience = self.collect_rollout()
        rollout_batch = self.process_rollout(experience)
        return rollout_batch

    def collect_rollout(self):

        batch_components_shape = (self.n_interactions, self.num_workers)

        # Do this in the train function
        states_batch = torch.zeros(
            batch_components_shape + self.envs.single_observation_space.shape,
            dtype=torch.float,
        ).to(self.device)
        actions_batch = torch.zeros(
            batch_components_shape + self.envs.single_action_space.shape,
            dtype=torch.float,
        ).to(self.device)
        rewards_batch = torch.zeros(batch_components_shape).to(self.device)
        dones_batch = torch.zeros(batch_components_shape).to(self.device)
        next_states_batch = torch.zeros(
            batch_components_shape + self.envs.single_observation_space.shape
        ).to(self.device)

        # Put this into the rollout function
        states = deepcopy(self.envs.observations)

        for step in range(self.n_interactions):

            actions = self.choose_action(states)
            next_states, rewards, dones, _ = self.envs.step(actions)

            states_batch[step] = to_torch(states, self.device)
            actions_batch[step] = to_torch(actions, self.device)
            rewards_batch[step] = to_torch(rewards, self.device)
            dones_batch[step] = to_torch(dones, self.device)
            next_states_batch[step] = to_torch(next_states, self.device)

            states = next_states
            self.training_interaction_step += self.num_workers

        return (
            states_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            next_states_batch,
        )

    def process_rollout(self, rollout):
        batches = [None for _ in range(self.num_workers)]
        (
            states_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            next_states_batch,
        ) = rollout

        for j in range(self.num_workers):
            batch = {}
            batch["states"] = states_batch[:, j]
            batch["actions"] = actions_batch[:, j]
            batch["rewards"] = rewards_batch[:, j]
            batch["dones"] = dones_batch[:, j]
            batch["next_states"] = next_states_batch[:, j]
            batch["advantages"], batch["v_targets"] = compute_gae_and_v_targets(
                self.critic, batch, self.device, self.gamma, self.lam
            )

            batches[j] = batch

        concat_batch = {
            k: torch.concat([b[k] for b in batches]) for k in batches[0].keys()
        }

        with torch.no_grad():
            concat_batch["old_log_probs"] = self.policy.get_log_prob(
                concat_batch["states"], concat_batch["actions"]
            )

        self.log_batch_stats(concat_batch)

        return concat_batch

    def update_from_batch(self, batch):
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropies = 0.0

        for _ in range(self.num_train_passes):
            minibatches = minibatch_split(batch, self.minibatch_size)
            for minibatch in minibatches:

                if self.norm_adv:
                    minibatch["advantages"], _ = normalise_adv(minibatch["advantages"])

                policy_loss, entropy = self.update_policy(minibatch)
                critic_loss = self.update_critic(minibatch)

                total_policy_loss += policy_loss
                total_critic_loss += critic_loss
                total_entropies += entropy

        mean_policy_loss = total_policy_loss / self.num_train_passes / len(minibatches)
        mean_critic_loss = total_critic_loss / self.num_train_passes / len(minibatches)
        mean_entropy = total_entropies / self.num_train_passes / len(minibatches)

        wandb.log(
            {
                "epoch": self.current_epoch,
                "mean_policy_loss": mean_policy_loss,
                "mean_critic_loss": mean_critic_loss,
                "mean_entropy": mean_entropy,
            }
        )

        return mean_policy_loss, mean_critic_loss

    def log_batch_stats(self, batch):
        out = {
            "epoch": self.current_epoch,
            "rollout/max_advantage": batch["advantages"].max(),
            "rollout/min_advantage": batch["advantages"].min(),
            "rollout/std_advantage": batch["advantages"].std(),
            "rollout/mean_advantage": batch["advantages"].mean(),
            "rollout/max_reward": batch["rewards"].max(),
            "rollout/min_reward": batch["rewards"].min(),
            "rollout/std_reward": batch["rewards"].std(),
            "rollout/mean_reward": batch["rewards"].mean(),
            "rollout/max_return": batch["v_targets"].max(),
            "rollout/min_return": batch["v_targets"].min(),
            "rollout/std_return": batch["v_targets"].std(),
            "rollout/mean_return": batch["v_targets"].mean(),
        }

        wandb.log(out)

        return out

    def run_training(self, num_epochs: int, verbose=True):
        """This is the main function needed to run."""
        eval_log = [() for _ in range(num_epochs)]
        for e in range(num_epochs):
            self.current_epoch += 1
            batch = self.generate_rollout()
            losses = self.update_from_batch(batch)
            eval_output = self.run_eval()

            eval_log[e] = (eval_output, losses)
            if verbose and e % 10 == 0:
                print("The log for epoch {} is {}".format(e, eval_log[e]))

        return eval_log


if __name__ == "__main__":

    env_name = "CartPole-v1"

    env = gym.make(env_name)

    policy_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(env.observation_space),
                "out_features": 32,
            },
        ),
        (nn.Tanh, {}),
        (nn.Linear, {"in_features": 32, "out_features": 32}),
        (nn.Tanh, {}),
        (
            nn.Linear,
            {"in_features": 32, "out_features": net_gym_space_dims(env.action_space)},
        ),
    ]

    critic_layers = [
        (
            nn.Linear,
            {
                "in_features": net_gym_space_dims(env.observation_space),
                "out_features": 32,
            },
        ),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 32, "out_features": 32}),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 32, "out_features": 1}),
    ]

    a2c_args = {
        "gamma": 0.99,
        "env_name": env_name,
        "step_lim": 200,
        "policy": CategoricalPolicy(policy_layers),
        "policy_optimiser": optim.Adam,
        "policy_lr": 0.002,
        "critic": Critic(critic_layers),
        "critic_lr": 0.002,
        "critic_optimiser": optim.Adam,
        "critic_criterion": nn.MSELoss(),
        "device": "cpu",
        "entropy_coef": 0.01,
        "n_interactions": 300,
        "num_train_passes": 1,
        "lam": 0.95,
        "num_eval_episodes": 15,
        "num_workers": mp.cpu_count() - 1,
    }

    agent = A2C(a2c_args)
    batch = agent.generate_rollout()

    # wandb.init(project="deeprl", name="a2c-cartpole", entity="akshil")
    # wandb.config = {
    #     "environment": env_name,
    #     "policy_layers": policy_layers,
    #     "critic_layers": critic_layers,
    #     "gamma": a2c_args["gamma"],
    #     "entropy_coef": a2c_args["entropy_coef"],
    #     "step_lim": a2c_args["step_lim"],
    #     "policy_lr": a2c_args["policy_lr"],
    #     "critic_lr": a2c_args["critic_lr"],
    #     "num_agents": num_agents,
    #     "num_episodes": num_epi,
    # }
    # num_agents = 5
    # num_epi = 200
    # r = []
    # for i in range(2):
    #     print("Running training for agent number {}".format(i))
    #     agent = A2C(a2c_args)

    #     # random.seed(i)
    #     # np.random.seed(i)
    #     # torch.manual_seed(i)
    #     # env.seed(i)

    #     r.append(agent.train(num_epi))

    # out = np.array(r).mean(0)

    # plt.figure(figsize=(5, 3))
    # plt.title("A2C on cartpole")
    # plt.xlabel("Episode")
    # plt.ylabel("Episodic Reward")
    # plt.plot(out, label="rewards")
    # plt.legend()

    # # plt.savefig('./data/a2c_cartpole.PNG')
    # plt.show()
