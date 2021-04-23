from collections import deque
import random
import torch


class Memory:
    """
    Memory provides storage and access for gym transitions.
    This class will store the data as is and then convert to tensors on an as needed basis when sampling.
    """
    def __init__(self, max_len, device):
        self.max_len = max_len
        self.buffer = self._reset_buffer()
        self.device = device
#         self.state_dtype = numpy_to_torch_dtype(env.observation_space.dtype)
#         self.action_dtype = numpy_to_torch_dtype(env.action_space.dtype)

    def __len__(self):
        return len(self.buffer)


    def _reset_buffer(self):
        return deque(maxlen=self.max_len)


    def sample(self, num_samples):
        """
        returns an iterable(states, actions, rewards, dones, next_states)
        """
        buffer_len = len(self.buffer)
        # If there aren't enough samples then take the min
        samples = random.sample(self.buffer, min([num_samples, buffer_len]))
        
        # separate lists for each part of transition
        states, actions, rewards, dones, next_states = zip(*samples)
        
        # Convert to tensors and put on device
        # calculate targets with target net 
        states = torch.tensor(states, dtype=torch.float, device=self.device) # shape:(mb_size, state_dim) using torch.float downgrades to float32 which i don't think matters...
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device) #shape: (mb_size, state_dim)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device) # shape:(mb_size, action_dim) used for indexing only 
        
        rewards = torch.tensor([rewards], dtype=torch.float, device=self.device) # mb_size,added to output
        dones = torch.tensor(dones, dtype=torch.int, device=self.device) # (mb_size, 1)
        
        return states, actions, rewards, dones, next_states

    def store(self, transition):
        self.buffer.append(transition)