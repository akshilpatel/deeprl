from gym.spaces import Box, Discrete
import numpy as np

def net_gym_space_dims(space):
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise TypeError("You haven't input a valid gym space. Here is your input: {}".format(space))


def get_gym_space_shape(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return (1,)
    else:
        raise TypeError("You haven't input a valid gym space. Here is your input: {}".format(space))




def discount_cumsum(x, dones, discount):
    """
    ## from cleanrl ##
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

    dones = dones.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    out = np.zeros_like(x)
    out[-1] = x[-1]

    # Iterate backward through time
    for t in reversed(range(x.shape[0]-1)):
        out[t] = x[t] + discount * out[t+1] * (1-dones[t])
    return out
