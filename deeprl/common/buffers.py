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


class ParallelMemory(Memory):
    def __init__(self, max_len, device, num_workers=1):
        self.max_len = max_len
        self.num_workers = num_workers
        self.buffers = [deque(maxlen=self.max_len) for i in range(self.num_workers)]
        self.device = device
        self.keys = ["states", "actions", "rewards", "dones", "next_states"]

    def store(self, transition):
        if self.num_workers == 1:
            self.buffers[0].append(transition)

        else:
            for i, buffer in enumerate(self.buffers):
                self.buffers.append(transition[i])
        return self.buffer

    def __len__(self):
        return len(self.buffers[0])


class OnPolicyMemory(Memory):
    """Version of memory where order matters."""

    def __init__(self, max_len, device):
        super().__init__(max_len, device)
        self.num_workers = num_workers

    def sample_batch(self):
        """Convert buffer to ordered stacks of different components instead of rows of transitions and convert these to float tensors.

        Returns:
            batch (dict[key: list(data)): Batch of experiences where keys correspond to each component recorded at a time step.
        """
        # separate lists for each part of transition
        states, actions, rewards, dones, next_states = [
            to_torch(data, self.device) for data in zip(*self.buffer)
        ]

        # TODO: Make this generalise to storing anything - maybe with an *args and map(torch.tensor.float().to)
        # Conversion to tensor is here instead of during storage to take advantage of vectorisation for a big batch

        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
        }

        return batch
