from gym.spaces import Box, Discrete


def get_gym_space_shape(space):
    if isinstance(space, Box):
        return space.shape[0]
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise TypeError("You haven't input a valid gym space. Here is your input: {}".format(space))
