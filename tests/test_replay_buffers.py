import pytest
import torch
from torch import nn, optim
import gym
from collections import deque

from deeprl.common.utils import get_gym_space_shape
from deeprl.common.replay_buffers import Memory

def test_memory_init():
    m = Memory(100000, 'cpu')
    assert type(m.buffer) == type(deque([1]))
    assert len(m.buffer) == 0
    assert m.device == 'cpu'     


# TODO: finish this test
def test_memory_sample():
    m = Memory(100, 'cuda' if torch.cuda.is_available() else 'cpu')
    ENVNAME = 'CartPole-v1'
    env = gym.make(ENVNAME)
    
    s = env.reset()
    for _ in range(50):
        a = env.action_space.sample()
        s_, r, d, _ = env.step(a)
        t = [s, a, r, d, s_]
        m.store(t)
        if d: 
            s = env.reset()
        
    states, actions, rewards, dones, next_states = m.sample(120)
    assert len(m) == 50
    
    assert states.shape == (50, 4)
    assert next_states.shape == (50, 4)

    assert torch.is_floating_point(state)
    assert actions.shape == (50, 1)
    assert dones.shape == (50, 1)
    


def test_memory_reset():
    m = Memory(100, 'cuda' if torch.cuda.is_available() else 'cpu')
    ENVNAME = 'CartPole-v1'
    env = gym.make(ENVNAME)
    
    s = env.reset()
    for _ in range(50):
        a = env.action_space.sample()
        s_, r, d, _ = env.step(a)
        t = [s, a, r, d, s_]
        m.store(t)
        if d: 
            s = env.reset()



# TODO: use this somehow
def tmp():
    cartpole_env = gym.make('CartPole-v1')
    input_dim = get_gym_space_shape(cartpole_env.observation_space)
    output_dim = get_gym_space_shape(cartpole_env.action_space)
    net_layers = [(nn.Linear, {"in_features": input_dim, "out_features": 16}), 
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 16, "out_features": 4}),
                (nn.ReLU, {}),
                (nn.Linear, {"in_features": 4, "out_features": output_dim})]
    