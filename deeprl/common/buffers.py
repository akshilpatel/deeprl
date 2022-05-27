from collections import deque
import random
import torch
from abc import ABC
from deeprl.common.utils import compute_gae_and_v_targets, to_torch
import numpy as np

# TODO: Turn this into an abstract base class.
class Memory:
    """
    Memory provides storage and access for gym transitions.
    This class will store the data as is and then convert to tensors as needed when sampling.
    """

    def __init__(self, max_len, device):
        self.max_len = max_len
        self.buffer = deque(maxlen=self.max_len)
        self.device = device
        self.keys = ["states", "actions", "rewards", "dones", "next_states"]
        self.batch = {k: None for k in self.keys}

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        """Deletes contents of memory where the buffer lives"""
        self.buffer.clear()

    def samples_to_batch(self, samples):
        states, actions, rewards, dones, next_states = [
            to_torch(x, self.device) for x in zip(*samples)
        ]

        self.batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
        }

        return self.batch

    def sample_batch(self, num_samples):
        buffer_len = len(self.buffer)

        # If there aren't enough samples then take what there is
        if num_samples > buffer_len:
            num_samples = buffer_len

        samples = random.sample(self.buffer, num_samples)

        batch = self.samples_to_batch(samples)

        return batch

    def store(self, transition):
        self.buffer.append(transition)
        return self.buffer

    def minibatch_split(self, mb_size, shuffle=True):
        """Returns a list of minibatch dictionaries."""
        batch_size = len(self.batch["states"])

        if batch_size % mb_size != 0 or mb_size > batch_size:
            raise ValueError("Minibatch size does not divide batch size.")

        batch_idc = np.arange(batch_size)

        if shuffle:
            np.random.shuffle(batch_idc)

        mb_indices = [batch_idc[i : i + mb_size] for i in range(0, batch_size, mb_size)]

        minibatches = []

        for mb_idc in mb_indices:
            minibatch = {k: v[mb_idc] for k, v in self.batch.items()}
            minibatches.append(minibatch)

        return minibatches

    def sample_batch(self):
        return self.batch


class OnPolicyMemory:
    def __init__(self, n_interactions, device, state_dim, action_dim, num_workers=1):
        self.n_workers = num_workers
        self.device = device
        self.keys = ["states", "actions", "rewards", "dones", "next_states"]
        self.n_interactions = n_interactions
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset_batch(self):
        batch_component_shape = self.n_interactions, self.num_workers
        # Do this in the train function
        self.states_buff = torch.zeros((*batch_component_shape, *self.state_dim), dtype=torch.float).to(self.device)

        self.actions_buff = torch.zeros((*batch_component_shape, *self.action_dim), dtype=torch.float).to(self.device)
        
        self.e_rewards_buff = torch.zeros(batch_component_shape).to(self.device)
        # self.i_rewards_buff = torch.zeros(batch_component_shape).to(self.device)
        self.dones_buff = torch.zeros(batch_component_shape).to(self.device)
        self.next_states_buff = torch.zeros((*batch_component_shape, *self.state_dim)).to(self.device)

        return True

    def extract_batch_components(self):
        batches = [None for _ in range(self.num_workers)]

        for j in range(self.num_workers):
            batch = {}
            batch["states"] = self.states_buff[:, j]
            batch["actions"] = self.actions_buff[:, j]
            batch["rewards"] = self.rewards_buff[:, j]
            batch["dones"] = self.dones_buff[:, j]
            batch["next_states"] = self.next_states_buff[:, j]
            
            batches[j] = batch
        
        return batches

    def sample(self):
        batches = self.extract_batch_components
    
    def store(self, transition, curr_step):
        states, actions, rewards, dones, next_states = transition

        self.states_buff[curr_step] = to_torch(states, self.device)
        self.actions_buff[curr_step] = to_torch(actions, self.device)
        self.e_rewards_buff[curr_step] = to_torch(rewards, self.device)
        # self.i_rewards_buff[curr_step] = to_torch(i_rewards, self.device)
        self.dones_buff[curr_step] = to_torch(dones, self.device)
        self.next_states_buff[curr_step] = to_torch(next_states, self.device)

        return True

        # Abstract to AC class
    def process_rollout(self):
        batches = [None for _ in range(self.num_workers)]

        for j in range(self.num_workers):
            batch = {}
            batch["states"] = self.states_buff[:, j]
            batch["actions"] = self.actions_buff[:, j]
            batch["e_rewards"] = self.e_rewards_buff[:, j]
            # batch["i_rewards"] = self.i_rewards_buff[:, j]
            batch["dones"] = self.dones_buff[:, j]
            batch["next_states"] = self.next_states_buff[:, j]
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

        return concat_batch

