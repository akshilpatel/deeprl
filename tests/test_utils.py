from deeprl.common.utils import (
    discount_cumsum,
    net_gym_space_dims,
    get_gym_space_shape,
    to_torch,
)
import pytest
import gym


@pytest.fixture()
def example_envs():
    mountain_car_env = gym.make("MountainCar-v0")

    lunar_lander_env = gym.make("LunarLander-v2")
    lunar_lander_cont_env = gym.make("LunarLander-v2")
    coinrun_env = gym.make("procgen-coinrun-v0")
    minigrid_env = gym.make("MiniGrid-Empty-8x8-v0")

    return (
        mountain_car_env,
        lunar_lander_env,
        lunar_lander_cont_env,
        coinrun_env,
        minigrid_env,
    )


def test_discount_cumsum():
    pass


def test_net_gym_space_dims():
    mountain_car, lunar_lander, lunar_lander_cont, coinrun, minigrid = example_envs()

    mountain_car_dims = (2,)
    lunar_lander_dims = (2,)

    net_gym_space_dims(mountain_car)
