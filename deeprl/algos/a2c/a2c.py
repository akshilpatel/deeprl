import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


from typing import List, Tuple, Dict


from deeprl.common.utils import net_gym_space_dims
from deeprl.common.base import Network
import multiprocessing as mp
from deeprl.common.buffers import OnPolicyMemory
from deeprl.common.base import CategoricalPolicy, GaussianPolicy

import wandb


import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


from typing import List, Tuple, Dict


from deeprl.common.utils import net_gym_space_dims, discount_cumsum, to_torch
from deeprl.common.base import Network
import multiprocessing as mp
from deeprl.common.buffers import OnPolicyMemory
from deeprl.common.base import CategoricalPolicy, GaussianPolicy

import wandb


class A2C:
    def __init__(self, args):
        self.device = args["device"]
        self.gamma = args["gamma"]
        self.env = args["env"]
        self.step_lim = args["step_lim"]

        # Critic
        self.critic = args["critic"].to(self.device)
        self.critic_lr = args["critic_lr"]
        self.critic_optimiser = args["critic_optimiser"](
            self.critic.parameters(), self.critic_lr
        )
        self.critic_criterion = args["critic_criterion"]

        # Policy
        self.policy = args["policy"].to(self.device)
        self.policy_lr = args["policy_lr"]
        self.policy_optimiser = args["policy_optimiser"](
            self.policy.parameters(), self.policy_lr
        )
        self.entropy_coef = args["entropy_coef"]

        self.batch_size = args["batch_size"]
        # self.minibatch_size = args["minibatch_size"]
        self.num_train_passes = args["num_train_passes"]
        self.lam = args["lam"]
        self.gamma_lam = self.gamma * self.lam
        self.entropy_coef = args["entropy_coef"]
        self.num_eval_episodes = args["num_eval_episodes"]

        # Memory
        self.buffer = OnPolicyMemory(self.batch_size, self.device)

        self.current_epoch = 0
        self.current_epi = 0
        self.current_epi_reward = 0.0

    def eval_run(self, num_episodes: int, render=False, verbose=True) -> np.array:
        """This is a wrapper method around the `run_episode` method to run several episodes in evaluation.

        Args:
            num_episodes (int): The number of episodes for which to run training.
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False.
            verbose (bool, optional): Used to decide if we should . Defaults to True.

        Returns:
            numpy array: Array of shape (num_episodes,) which contains the episodic rewards returned by the `run_episode` method.
        """

        total_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            total_rewards[i] = self.run_eval_episode(render=render)

            if i % 10 == 0 and verbose:
                print("Episode {}, Reward {}".format(i, total_rewards[i]))
        return total_rewards

    def run_eval_episode(self, step_lim=np.inf, render=False):
        """Functionality for running an episode with updating.

        Args:
            step_lim (int, optional): Maximum number of steps to limit each episode.
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False. Defaults to False.

        Returns:
            float: Reward accumulated during the episode.
        """
        state = self.env.reset()
        episodic_reward = 0.0
        step_counter = 0
        while step_counter < step_lim:

            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            step_counter += 1

            if render:
                self.env.render()

            if done:
                break

            state = next_state

        return episodic_reward

    def run_eval(self, num_episodes: int = 10, step_lim=np.inf, render: bool = False):
        self.policy.eval()
        self.critic.eval()
        epi_rewards = [
            self.run_eval_episode(step_lim, render) for _ in range(num_episodes)
        ]
        self.policy.train()
        self.critic.train()
        return epi_rewards

    def generate_experience(self):
        """Interact with environment to produce rollout over multiple episodes if necessary.

        Simulates agents interaction with gym env, stores as tuple (s, a, lp, r, d, ns)


        Args:
            render (bool, optional): Whether or not to visualise the interaction. Defaults to False.

        Returns:
            episodic_rewards (list): scalar rewards for each episode that the agent completes.
        """
        self.current_epoch += 1
        state = self.env.reset()

        self.current_epi += 1

        self.current_epi_reward = 0.0
        step_counter = 0
        batch_reward = 0.0

        while step_counter < self.batch_size:

            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.buffer.store((state, action, reward, done, next_state))

            batch_reward += reward
            step_counter += 1
            # wandb.log(
            #     {"train_action": action},
            #     step=step_counter,
            # )
            # Storage

            if done:
                state = self.env.reset()
                # log train_episode_length
            else:
                state = next_state

        return batch_reward

    def update_critic(self, minibatch):
        states = minibatch["states"]
        v_targets = minibatch["v_targets"]

        assert not v_targets.requires_grad

        v_preds = self.critic(states)

        critic_loss = self.critic_criterion(v_preds, v_targets)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.critic_optimiser.step()
        return critic_loss.item()

    @torch.no_grad()
    def compute_td_deltas(self, batch):
        states = batch["states"]
        next_states = batch["next_states"]
        rewards = batch["rewards"].unsqueeze(-1)
        dones = batch["dones"].unsqueeze(-1)

        first_state = states[0].unsqueeze(0)
        first_state_value = self.critic(first_state)  # this state is never terminal

        next_values = self.critic(next_states) * (1 - dones)
        assert dones.shape == next_values.shape

        all_values = torch.cat([first_state_value, next_values])
        current_values = all_values[:-1]

        assert rewards.shape == next_values.shape
        assert current_values.shape == next_values.shape

        deltas = rewards + (self.gamma * next_values) - current_values

        assert deltas.dim() == 2, deltas.shape
        assert deltas.shape[1] == 1

        return deltas

    def update_policy(self, minibatch):

        advantages = minibatch["advantages"]
        states = minibatch["states"]
        actions = minibatch["actions"]

        log_probs, entropies = self.get_log_probs_and_entropies(states, actions)

        policy_loss = -advantages * log_probs - entropies * self.entropy_coef
        policy_loss = policy_loss.mean()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimiser.step()

        return policy_loss.item()

    @torch.no_grad()
    def compute_gae_and_v_targets(self, batch):
        """
        Computes the v_targets and advantages for policy and critic updates. No gradients are tracked.
        Advantages are computed using GAE.

        Args:
            states (torch tensor, float, (1, state_dim)): current state
            next_states (torch tensor, float, (1, state_dim)): next state
            rewards (float): one-step reward
            dones (boole): true iff next_state of the transition was terminal

        Returns:
            advantages (torch tensor, float, (batch_size, 1)): Using GAE
            v_targets (torch tensor, float, (batch_size, 1)): adv + v(s)
        """
        td_deltas = self.compute_td_deltas(batch)

        advantages = discount_cumsum(td_deltas, batch["dones"], self.gamma_lam)
        advantages = torch.from_numpy(advantages).to(self.device)

        state_values = self.critic(
            batch["states"]
        )  # States are all those from which an action is taken (hence non-terminal).

        print(advantages.shape, state_values.shape)
        v_targets = advantages + state_values

        assert not td_deltas.requires_grad
        assert not advantages.requires_grad
        assert advantages.shape == (len(batch["states"]), 1)
        assert v_targets.shape == (len(batch["states"]), 1)

        # TODO: Change this to use a standardiser transform.
        advantages -= advantages.mean()
        advantages /= advantages.std()

        return advantages, v_targets

    # @torch.no_grad()
    def choose_action(self, state):
        """Calls the policy network to sample an action for a given state. The log_prob of the action and the entropy of the distribution are also recorded for updates.

        Args:
            state (numpy array): current state of the environment

        Returns:
            numpy array (gym action_dim): action selected
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = self.policy.get_action(state)

        action = action.cpu().detach().numpy().squeeze()

        assert self.env.action_space.contains(action)

        return action

    def get_log_probs_and_entropies(self, states, actions):

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)

        log_probs = self.policy.get_log_prob(states, actions).unsqueeze(-1)
        entropies = self.policy.get_entropy(states).unsqueeze(-1)

        assert log_probs.requires_grad
        assert entropies.requires_grad

        assert log_probs.shape == (len(states), 1)
        assert entropies.shape == (len(states), 1)

        return log_probs, entropies

    def train_one_epoch(self):
        self.generate_experience(self.batch_size)
        batch = self.buffer.sample(batch)
        advantages, v_targets = self.compute_gae_and_v_targets(batch)
        batch["advantages"], batch["v_targets"] = advantages, v_targets

        policy_loss = self.update_policy(batch)
        critic_loss = self.update_critic(batch)

        eval_log = self.run_eval(
            self.num_eval_episodes, step_lim=self.step_lim, render=False
        )
        return policy_loss, critic_loss, eval_log

    def run_experiment(num_epochs):
        for _ in range(num_epochs):
            p_loss, c_loss, eval_log = self.train_one_epoch()


if __name__ == "__main__":

    ENVNAME = "CartPole-v1"

    env = gym.make(ENVNAME)

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
        "env": env,
        "step_lim": 200,
        "policy": CategoricalPolicy(policy_layers),
        "policy_optimiser": optim.Adam,
        "policy_lr": 0.001,
        "critic": Network(critic_layers),
        "critic_lr": 0.001,
        "critic_optimiser": optim.Adam,
        "critic_criterion": nn.MSELoss(),
        "device": "cuda",
        "entropy_coef": 0.01,
        "batch_size": 30,
    }

    num_agents = 5
    num_epi = 200
    r = []

    wandb.init(project="deeprl", name="a2c-cartpole", entity="akshil")
    wandb.config = {
        "environment": ENVNAME,
        "policy_layers": policy_layers,
        "critic_layers": critic_layers,
        "gamma": a2c_args["gamma"],
        "entropy_coef": a2c_args["entropy_coef"],
        "step_lim": a2c_args["step_lim"],
        "policy_lr": a2c_args["policy_lr"],
        "critic_lr": a2c_args["critic_lr"],
        "num_agents": num_agents,
        "num_episodes": num_epi,
    }

    for i in range(num_agents):
        print("Running training for agent number {}".format(i))
        agent = A2C(a2c_args)

        # random.seed(i)
        # np.random.seed(i)
        # torch.manual_seed(i)
        # env.seed(i)

        r.append(agent.train(num_epi))

    out = np.array(r).mean(0)

    plt.figure(figsize=(5, 3))
    plt.title("A2C on cartpole")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.plot(out, label="rewards")
    plt.legend()

    # plt.savefig('./data/a2c_cartpole.PNG')
    plt.show()
