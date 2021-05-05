from collections import deque
import random
import torch


class Memory:
    """
    Memory provides storage and access for gym transitions.
    This class will store the data as is and then convert to tensors as needed basis when sampling.
    """
    def __init__(self, max_len, device):
        self.max_len = max_len
        self.buffer = deque(maxlen=self.max_len)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    
    def to_torch(self, x):
        return torch.tensor(x).to(self.device)


    def reset_buffer(self):
        """Deletes contents of memory where the buffer lives"""
        self.buffer.clear()
        assert len(self.buffer) == 0

    def sample(self, num_samples):
        """
        returns an iterable(states, actions, rewards, dones, next_states)
        """
        buffer_len = len(self.buffer)
        
        # If there aren't enough samples then take what there is
        if num_samples > buffer_len:
            num_samples = buffer_len

        samples = random.sample(self.buffer, num_samples)
        
        # separate lists for each part of transition
        states, actions, rewards, dones, next_states = zip(*samples)
        
        # TODO: Make this generalise to storing anything - maybe with an *args and map(torch.tensor.float().to)
        # Conversion to tensor is here instead of during storage to take advantage of vectorisation for a big batch
        states = torch.tensor(states, dtype=torch.float, device=self.device)            # shape:(mb_size, state_dim) 
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)  # shape: (mb_size, state_dim)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)          # shape:(mb_size, action_dim) used for indexing only 
        rewards = torch.tensor([rewards], dtype=torch.float, device=self.device)        # shape (mb_size,) added to output
        dones = torch.tensor(dones, dtype=torch.int, device=self.device)                # (mb_size, 1)
        
        return states, actions, rewards, dones, next_states

    def store(self, transition):
        self.buffer.append(transition)



class OnPolicyMemory(Memory):
    """Version of memory where order matters

    Args:
        Memory (object): Base memory class which
    """
    def __init__(self, max_len, device):
        self.max_len = max_len
        self.device = device
        self.buffer = deque(maxlen=self.max_len)
    
    def sample(self):
        out = tuple(map(self.to_torch, zip(*self.buffer)))
        return out

    def compute_adv_and_returns(self, rewards, dones):
        pass
