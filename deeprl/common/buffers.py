from collections import deque
import random
import torch
from abc import ABC
from deeprl.common.utils import to_torch
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

    def __len__(self):
        return len(self.buffer)

    # def to_torch(self, x):
    #     arr_x = np.array(x).astype(np.float32)
    #     out = torch.from_numpy(arr_x).to(self.device)
    #     return out

    def reset_buffer(self):
        """Deletes contents of memory where the buffer lives"""
        self.buffer.clear()

    def samples_to_batch(self, samples):
        states, actions, rewards, dones, next_states = [
            to_torch(x, self.device) for x in zip(*samples)
        ]

        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
        }

        return batch

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


class OnPolicyMemory(Memory):
    """Version of memory where order matters."""

    def __init__(self, max_len, device):
        super().__init__(max_len, device)

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
