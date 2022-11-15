# -*- coding:utf-8 -*-
import argparse
import os
import random
import time
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.003,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=2000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.98,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=200,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.35,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.03,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """initialize the deep neural network for Q-learning, inherited from nn.Module. In __init__, First call the initializer of father class, because only if we call the initializer of nn.Module can Pytroch track the things that we are adding to the network. Then build our own neural network which is composed of 3 linear layers.  Use Sequential to wrap the all 3 layers."""
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

episode_rewards = []
if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = make_env(args.env_id, args.seed)
    q_network = QNetwork(envs).to(device)
    q_network.load_state_dict(torch.load("lr_0.003++bfSize_2000++gamma_0.98++tnf_200++startE_0.35++endE_0.03.pth"))

    obs = envs.reset()
    envs.render()
    while True:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=0).cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(actions)
        envs.render()
        if dones:
            break
        obs = next_obs
    envs.close()