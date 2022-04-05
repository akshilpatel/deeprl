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
    
#######################
## OnPolicyMemory ##
#######################


def test_sample():
    buffer = OnPolicyMemory()


    ### TESTING ###

    # check everything is the right shape and size
    assert len(states) == agent.batch_size
    assert len(actions) == agent.batch_size
    assert log_probs.shape == (agent.batch_size,)
    assert rewards.shape == (agent.batch_size,)
    assert dones.shape == (agent.batch_size,)
    assert next_states.shape == states.shape

    assert states.device == torch.device(agent.device)
    assert rewards.device == torch.device(agent.device)
    assert dones.device == torch.device(agent.device)

    assert torch.is_floating_point(states)
    assert torch.is_floating_point(actions)
    assert torch.is_floating_point(rewards)
    assert torch.is_floating_point(dones)
    assert torch.is_floating_point(next_states)
    
    assert len(epi_rewards) == torch.sum(dones)
    assert isinstance(epi_rewards)

    for i in range(len(states)):
        assert agent.env.observation_space.contains(states[i].cpu().detach().numpy()), states[i].cpu().detach().numpy()
        assert agent.env.observation_space.contains(next_states[i].cpu().detach().numpy()), next_states[i].cpu().detach().numpy()
