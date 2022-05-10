from gym.spaces import Box, Discrete
import numpy as np
import torch


def net_gym_space_dims(space):
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise TypeError(
            "You haven't input a valid gym space. Here is your input: {}".format(space)
        )


def get_gym_space_shape(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return (1,)
    else:
        raise TypeError(
            "You haven't input a valid gym space. Here is your input: {}".format(space)
        )


def discount_cumsum(x, dones, discount):
    """
    ## Adapted from cleanrl ##
    computing discounted cumulative sums of vectors that resets with dones
    input:
        vector x,  vector dones,
        [x0,       [0,
         x1,        0,
         x2         1,
         x3         0,
         x4]        0]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    assert len(x) > 0
    assert len(dones) == len(x)
    assert type(discount) in [float, int]

    dones = dones.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    out = np.zeros_like(x)
    out[-1] = x[-1]

    # Iterate backward through time
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + discount * out[t + 1] * (1 - dones[t])
    return out


def to_torch(x, device):
    # if type(x) != np.ndarray:
    #     arr_x = np.array(x).astype(np.float32)
    out = torch.tensor(x, device=device, dtype=torch.float).to(device)
    return out


@torch.no_grad()
def compute_td_deltas(critic, batch, gamma):
    states = batch["states"]
    next_states = batch["next_states"]
    rewards = batch["rewards"].unsqueeze(-1)
    dones = batch["dones"].unsqueeze(-1)

    first_state = states[0].unsqueeze(0)
    first_state_value = critic(first_state)  # this state is never terminal

    next_values = critic(next_states) * (1 - dones)
    assert dones.shape == next_values.shape

    all_values = torch.cat([first_state_value, next_values])
    current_values = all_values[:-1]

    assert rewards.shape == next_values.shape
    assert current_values.shape == next_values.shape

    deltas = rewards + (gamma * next_values) - current_values

    assert deltas.dim() == 2, deltas.shape
    assert deltas.shape[1] == 1

    return deltas


@torch.no_grad()
def compute_gae_and_v_targets(critic, batch, device, gamma, lam):
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

    td_deltas = compute_td_deltas(critic, batch, gamma)

    advantages = discount_cumsum(td_deltas, batch["dones"], gamma * lam)
    advantages = torch.tensor(advantages).to(device)

    state_values = critic(batch["states"])  # States are all non-terminal.

    v_targets = advantages + state_values

    assert not td_deltas.requires_grad
    assert not advantages.requires_grad
    assert advantages.shape == (len(batch["states"]), 1)
    assert v_targets.shape == (len(batch["states"]), 1)

    # advantages = normalise_adv(advantages)

    return advantages, v_targets


def normalise_adv(advantages):
    if len(advantages) == 1:
        return advantages

    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages -= adv_mean

    advantages /= adv_std + 1e-8

    return advantages


def process_batch(raw_batch, device):
    """Convert buffer to ordered stacks of different components instead of rows of transitions and convert these to float tensors.
    Args:
        raw_batch (list(tuple(transition))): List of one step transition tuples
        devices : Device on which to put the tensors after conversion.


    Returns:
        batch (dict[key: list(data)): Batch of experiences where keys correspond to each component recorded at a time step.
    """
    # separate lists for each part of transition
    states, actions, rewards, dones, next_states = [
        to_torch(data, device) for data in zip(*raw_batch)
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


def process_parallel_batch(raw_batch, device):

    for (i, b) in raw_batch:
        batch = process_batch(raw_batch, device)

    pass
