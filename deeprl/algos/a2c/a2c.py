from cv2 import log
import gym
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


from typing import List, Tuple, Dict


from deeprl.common.utils import net_gym_space_dims
from deeprl.common.base import Network
import multiprocessing as mp
from torch.distributions import Categorical, Normal
from deeprl.common.base import CategoricalPolicy, GaussianPolicy

import wandb

class A2C:
    def __init__(self, args):     
        self.device = args["device"]
        self.gamma = args["gamma"]
        self.env = args["env"]
        self.step_lim = args["step_lim"]

        # Critic
        self.critic = args["critic"].to(self.device)
        self.critic_lr = args["critic_lr"]
        self.critic_optimiser = args["critic_optimiser"](
            self.critic.parameters(), self.critic_lr
        )
        self.critic_criterion = args["critic_criterion"]

        # Policy
        self.policy = args["policy"].to(self.device)
        self.policy_lr = args["policy_lr"]
        self.policy_optimiser = args["policy_optimiser"](
            self.policy.parameters(), self.policy_lr
        )
        self.entropy_coef = args["entropy_coef"]
        # self.adv_n = 0.
        # self.adv_mean = 0.


    def train(self, num_episodes:int, render=False, verbose=True) -> np.array:
        """This is a wrapper method around the `run_episode` method. We use `train` mehod for running experiments. 

        Args:
            num_episodes (int): The number of episodes for which to run training.
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False.
            verbose (bool, optional): Used to decide if we should . Defaults to True.

        Returns:
            numpy array: Array of shape (num_episodes,) which contains the episodic rewards returned by the `run_episode` method. 
        """

        total_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            total_rewards[i] = self.run_episode(render=render) 

            if i%10==0 and verbose:
                print("Episode {}, Reward {}".format(i, total_rewards[i]))
        return total_rewards

    def run_episode(self, render=False):
        """Functionality for running an episode with updating.

        Args:
            render (bool, optional): Flag used to render episodes to watch training. Defaults to False. Defaults to False.

        Returns:
            _type_: _description_
        """
        state = self.env.reset()
        episodic_reward = 0.
        step_counter = 0
        while step_counter < self.step_lim:
            
            action, log_prob, entropy = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward 
            step_counter +=1
            
            if render: self.env.render()
            
            losses = self.update(state, reward, next_state, done, log_prob, entropy)
            
            if done: 
                # print(losses)
                break

            state = next_state

        wandb.log({"episodic_reward": episodic_reward})
        return episodic_reward

    def update(self, state, reward, next_state, done:bool, log_prob, entropy):
        """One step update which:
        1) Converts the experience to tensors and puts on the correct device
        2) Calls `.compute_adv_and_td_target` method to compute the advantage and td_target (without gradients) 
        3) Calls the `update_policy` and `update_critic` methods using the advantage and td_target to update both policy and critic.

        Args:
            state (numpy array): State from which the agent starts the transition
            reward (float or int): One step reward for the transition
            next_state (numpy array): Next state after taking the action.
            done (bool): True iff the next_state was terminal. 
            log_prob (torch.FloatTensor): shape=(1,1)
            entropy (torch.FloatTensor): shape=(1,1)

        Returns:
            Tuple(float, float): (Policy Loss, Critic Loss) computed for the transition.
        """

        state = torch.tensor([state], dtype=torch.float, device=self.device)
        next_state = torch.tensor([next_state], dtype=torch.float, device=self.device)
        # done = torch.tensor(done, dtype=torch.int, device=self.device) # are these two lines needed since they don't go through a model.
        # reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        # print(state, next_state, done, reward)

        # These are both targets so no need to track grads
        # with torch.no_grad():
        adv, td_target = self.compute_adv_and_td_target(state, reward, next_state, done)

        assert not adv.requires_grad
        assert not td_target.requires_grad

        policy_loss = self.update_policy(log_prob, entropy, adv)
        critic_loss = self.update_critic(td_target, state)

        return policy_loss, critic_loss

    def update_policy(self, log_prob, entropy, adv):
        """Computes the policy loss and updates the policy network

        Args:
            log_prob (torch.tensor): Log probability of action taken by agent.
            entropy (torch.tensor): Entropy value for transitions.
            adv (torch.tensor): Advantage value for the transitions.

        Returns:
            float: policy loss computed for transition.
        """
        assert not adv.requires_grad
        assert log_prob.requires_grad
        # print(log_prob)
        # print(adv)
        
        policy_loss = -(log_prob * adv) - (entropy * self.entropy_coef)
        # print(policy_loss, policy_loss.shape)
        policy_loss = policy_loss.squeeze()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 0.2)
        self.policy_optimiser.step()
        wandb.log({"policy_loss": policy_loss.item()})
        return policy_loss.item()

    def update_critic(self, td_target, state):
        """Used to compute loss and update for critic

        Args:
            td_target (torch.tensor): r + gamma * V(next_state) * (1 - done)
            state (torch.tensor): State in transition

        Returns:
            float: critic loss
        """
       

        current_state_val = self.critic(state)
        
        critic_loss = self.critic_criterion(current_state_val, td_target)
        
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.2)
        self.critic_optimiser.step()

        wandb.log({"critic_loss": critic_loss.item()})
        
        return critic_loss.item()

    @torch.no_grad()
    def compute_adv_and_td_target(self, state, reward, next_state, done):
        """
        Computes the td_target and advantage for policy and critic updates. No gradients are tracked.
        
        Args:
            state (torch tensor, float, (1, state_dim)): current state
            next_state (torch tensor, float, (1, state_dim)): next state
            reward (float): one-step reward
            done (boole): true iff next_state of the transition was terminal

        Returns:
            advantage (torch tensor, float, (1, 1)): td_target - V(s)
            td_target (torch tensor, float, (1, 1)): r + gamma * V(s') * (1-done)
        """
        
        td_target = reward + self.gamma * self.critic(next_state) * (1-done)
        adv = td_target - self.critic(state)
        
        assert not td_target.requires_grad
        assert not adv.requires_grad
        assert adv.shape == (1,1)
        assert td_target.shape == (1,1)
        
        wandb.log({"advantage": adv.item()})
        wandb.log({"td_target": td_target.item()})

        return adv, td_target

    
    def choose_action(self, state):
        """Calls the policy network to sample an action for a given state. The log_prob of the action and the entropy of the distribution are also recorded for updates.

        Args:
            state (numpy array): current state of the environment

        Returns:
            numpy array (gym action_dim): action selected
            torch float tensor (1, 1): log probability of policy distribution used to sample the action
            torch float tensor (1, 1): entropy of policy distribution used to sample the action
        """ 
        state = torch.tensor([state], dtype=torch.float, device=self.device)
        action, log_prob, entropy = self.policy.sample(state)

        action = action.cpu().detach().numpy().squeeze()
        
        assert self.env.action_space.contains(action)
        wandb.log({"lp": log_prob.item()})
        wandb.log({"entropy": entropy.item()})

        return action, log_prob, entropy



if __name__ == "__main__":

    ENVNAME = "CartPole-v1"

    env = gym.make(ENVNAME)

    policy_layers = [
        (nn.Linear,
            {"in_features": net_gym_space_dims(env.observation_space),
            "out_features": 32}),
        (nn.Tanh, {}),
        (nn.Linear,
            {"in_features": 32,
            "out_features": 32}),
        (nn.Tanh, {}),
        (nn.Linear,{"in_features": 32, "out_features": net_gym_space_dims(env.action_space)}),
    ]

    critic_layers = [
        (nn.Linear, {"in_features": net_gym_space_dims(env.observation_space), "out_features": 32}),
        (nn.ReLU, {}),
        (nn.Linear,
            {"in_features": 32,
            "out_features": 32}),
        (nn.ReLU, {}),
        (nn.Linear, {"in_features": 32, "out_features": 1}),
    ]

    a2c_args = {
        "gamma": 0.99,
        "env": env,
        "step_lim": 200,
        "policy": CategoricalPolicy(policy_layers),
        "policy_optimiser": optim.Adam,
        "policy_lr": 0.001,
        "critic": Network(critic_layers),
        "critic_lr": 0.001,
        "critic_optimiser": optim.Adam,
        "critic_criterion": nn.MSELoss(),
        "device": "cuda",
        "entropy_coef": 0.01,
    }

    num_agents = 5
    num_epi = 200
    r = []

    wandb.init(project="deeprl", name="a2c-cartpole", entity="akshil")
    wandb.config = {
        "environment": ENVNAME,
        "policy_layers": policy_layers,
        "critic_layers": critic_layers,
        "gamma": a2c_args["gamma"],
        "entropy_coef": a2c_args["entropy_coef"],
        "step_lim": a2c_args["step_lim"],
        "policy_lr": a2c_args["policy_lr"],
        "critic_lr": a2c_args["critic_lr"], 
        "num_agents": num_agents,
        "num_episodes": num_epi}
        

    for i in range(num_agents):
        print("Running training for agent number {}".format(i))
        agent = A2C(a2c_args)
            
        # random.seed(i)
        # np.random.seed(i)
        # torch.manual_seed(i)
        # env.seed(i)

        r.append(agent.train(num_epi))

    out = np.array(r).mean(0)

    plt.figure(figsize=(5, 3))
    plt.title('A2C on cartpole')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Reward')
    plt.plot(out, label='rewards')
    plt.legend()

    # plt.savefig('./data/a2c_cartpole.PNG')
    plt.show()