from collections import defaultdict
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
    compute_gae_and_v_targets,
    to_torch,
    minibatch_split,
    normalise_adv,
    init_envs,
    concat_parallel_batches
)
from deeprl.common.base import Network, Critic, CategoricalPolicy, GaussianPolicy
import wandb


class PPO:
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
        self.entropy_coef = params["entropy_coef"]
        self.policy = params["policy"].to(self.device)
        self.policy_lr = params["policy_lr"]
        self.policy_optimiser = params["policy_optimiser"](
            self.policy.parameters(), self.policy_lr
        )

        self.current_epoch = 0
        self.training_interaction_step = 0

        self.envs = init_envs(self.env_name, self.num_workers, self.multiprocess)
        self.loss_clip_coef = params["loss_clip_coef"]

        self.intrinsic_coeff = params["intrinsic_coeff"]
        self.extrinsic_coeff = params["extrinsic_coeff"]
        self.training_states_counts = defaultdict(lambda: 0)

    # Abstract into Agent class
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

        episodic_task_reward = 0.0
        step_counter = 0
        eval_state_counts =  defaultdict(lambda: 0)
        eval_state_counts[state] += 1
        while step_counter < step_lim:

            action = self.choose_action(np.expand_dims(state, 0))
            next_state, task_reward, done, _ = env.step(action)
            eval_state_counts[next_state] += 1
            intrinsic_reward = 1. / np.sqrt(eval_state_counts[next_state])

            episodic_task_reward += reward
            step_counter += 1

            if render:
                env.render()
            if done:
                break

            state = next_state # TODO: Check that this does not increment the wrong state_count, by setting the 

        env.close()
        # info = {"eval_epi_len": step_counter, "eval_epi_reward": episodic_reward}
        return episodic_reward, step_counter, 

    # Abstract into Agent class
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

        eval_stats = {
            "eval/mean_epi_reward": np.mean(epi_rewards),
            "eval/std_epi_reward": np.std(epi_rewards),
            "eval/max_epi_reward": max(epi_rewards),
            "eval/min_epi_reward": min(epi_rewards),
            "eval/mean_epi_length": np.mean(epi_lengths),
            "eval/std_epi_length": np.std(epi_lengths),
            "eval/max_epi_length": max(epi_lengths),
            "eval/min_epi_length": min(epi_lengths),
            "eval/num_interactions": sum(epi_lengths),
        }
        # Add intrinsic rewards and metrics to log here?

        self.policy.train()
        self.critic.train()

        return eval_stats

    def update_critic(self, minibatch):
        states = minibatch["states"]
        v_targets = minibatch["v_targets"]

        assert not v_targets.requires_grad

        v_preds = self.critic(states)

        critic_loss = self.critic_criterion(v_preds, v_targets)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_coef)
        self.critic_optimiser.step()

        return critic_loss.item()

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

    # Abstract into Agent class.
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

    # Abstract to agent class
    def generate_batch(self):
        rollout = self.collect_rollout()
        batches = self.extract_batches(rollout)
        batches = self.process_batches(batches)
        out_batch = concat_parallel_batches(batches)

        return out_batch

    # Abstract to agent class
    def collect_rollout(self):

        batch_components_shape = (self.n_interactions, self.num_workers)
        state_dim = self.envs.single_observation_space.shape
        action_dim = self.envs.single_action_space.shape

        # Buffer
        states_batch = torch.zeros((*batch_components_shape, *state_dim), dtype=torch.float).to(self.device)
        actions_batch = torch.zeros((*batch_components_shape, *action_dim), dtype=torch.float).to(self.device)
        rewards_batch = torch.zeros(batch_components_shape).to(self.device)
        dones_batch = torch.zeros(batch_components_shape).to(self.device)
        next_states_batch = torch.zeros((*batch_components_shape, *state_dim), dtype=torch.float).to(self.device)

        # Put this into the rollout function
        states = deepcopy(self.envs.observations)

        for step in range(self.n_interactions):

            actions = self.choose_action(states)
            next_states, rewards, dones, _ = self.envs.step(actions)

            # self.buffer.store((states, actions, rewards, dones, next_states))

            # buffer
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
   
        
    # Buffer
    def extract_batches(self, rollout):
        batches = [None for _ in range(self.num_workers)]
        states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch = rollout
        

        for j in range(self.num_workers):
            batch = {}
            batch["states"] = states_batch[:, j]
            batch["actions"] = actions_batch[:, j]
            batch["rewards"] = rewards_batch[:, j]
            batch["dones"] = dones_batch[:, j]
            batch["next_states"] = next_states_batch[:, j]
            
            batches[j] = batch
        
        return batches

    # A2C
    def process_batches(self, batches):
        """
        Compute targets etc., for use in updating the agent.
        Args:
            batches (list(dict)): list of dictionary batches where each dictionary corresponds to one worker's experience.
        
        Returns:
            out_batches: The same list of dictionary batches with (key, values) added for everything needed in the update function.
        """
        out_batches = [None for _ in range(self.num_workers)]

        for i, batch in enumerate(batches): 
            adv, v_targets = compute_gae_and_v_targets(self.critic, batch, self.device, self.gamma, self.lam)
            batch["advantages"], batch["v_targets"] = adv, v_targets
            
            # Used for logging if not for updating.
            with torch.no_grad():
                batch["old_log_probs"] = self.policy.get_log_prob(
                    batch["states"], batch["actions"]
                )
            out_batches[i] = batch
        return out_batches
    
    # Abstract to ActorCritic class
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

        with torch.no_grad():
            approx_kl_vec = self.compute_approx_kl(batch)
            

        training_stats = {
            "train/mean_policy_loss": mean_policy_loss,
            "train/mean_critic_loss": mean_critic_loss,
            "train/mean_entropy": mean_entropy,
            "train/mean_kl_div": approx_kl_vec.mean().item(),
        }

        return training_stats

    
    def compute_approx_kl(self, batch):
        assert "old_log_probs" in batch.keys()

        new_log_probs = self.policy.get_log_prob(batch["states"], batch["actions"])
        log_ratio = new_log_probs - batch["old_log_probs"]
        # See http://joschu.net/blog/kl-approx.html
        approx_kl = log_ratio.exp() - 1 - log_ratio
        
        return approx_kl


    def compute_batch_stats(self, batch):
        batch_stats = {
            "rollout/max_advantage": batch["advantages"].max().item(),
            "rollout/min_advantage": batch["advantages"].min().item(),
            "rollout/std_advantage": batch["advantages"].std().item(),
            "rollout/mean_advantage": batch["advantages"].mean().item(),
            "rollout/max_reward": batch["rewards"].max().item(),
            "rollout/min_reward": batch["rewards"].min().item(),
            "rollout/std_reward": batch["rewards"].std().item(),
            "rollout/mean_reward": batch["rewards"].mean().item(),
            "rollout/max_return": batch["v_targets"].max().item(),
            "rollout/min_return": batch["v_targets"].min().item(),
            "rollout/std_return": batch["v_targets"].std().item(),
            "rollout/mean_return": batch["v_targets"].mean().item(),
        }
        return batch_stats

    def run_training(self, num_epochs: int, is_logging: bool = True):
        """This is the main interface for runnning experiments."""
        eval_log = [() for _ in range(num_epochs)]
        for e in range(num_epochs):
            self.current_epoch += 1
            batch = self.generate_batch()

            if is_logging:
                batch_stats = self.compute_batch_stats(batch)

            training_stats = self.update_from_batch(batch)
            eval_stats = self.run_eval()

            eval_log[e] = eval_stats

            epoch_stats = {
                "epoch": self.current_epoch,
                "rollout/interaction_steps": self.training_interaction_step,
            }

            if is_logging:
                wandb.log(
                    {**epoch_stats, **batch_stats, **training_stats, **eval_stats}
                )

            # This is used for debugging only.
            elif e % 5 == 0:
                print("Epoch {}: ".format(e), eval_stats)

        return eval_log


