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

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """schedule the epsilon value linearly. First divide epsilon range to common pieces and provide each time step with epsilon_value = start_e + t * piece_size"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

episode_rewards = []
if __name__ == "__main__":

    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    """we utilize tensorboard to log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """set the seed of random number generator of all backends to a fixed number to guarantee reporductivity"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """set the seed of the random number generator of gym environment to guarantee reproductivity and construct the gym environment"""
    envs = make_env(args.env_id, args.seed)

    """initialize the networks and the optimizer, set the parameters of the two network as the same, move the model to GPU if available"""
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """set the parameters of replay memory according to the parameters given in args"""
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """initialize the environment, get prepared for training."""
    obs = envs.reset()
    for global_step in range(args.total_timesteps):

        """shedule epsilon for each global step using linear_schedule function, which performs a linear decay"""
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        """perform epsilon-greedy policy, act randomly with epsilon probability"""
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

        """perform the chosen action and renew the environment"""
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training

        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            episode_rewards.append(infos['episode']['r'])

        """add current action to the memory pool for later training use"""
        rb.add(obs, next_obs, actions, rewards, dones, infos)

        """renew the environment for the next action or reset the action if the mission is completed"""
        obs = next_obs if not dones else envs.reset()

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:

            """samble #batch_size training samples from the memory pool"""
            data = rb.sample(args.batch_size)

            """clear previous gradient and calculate loss via target net using MSELoss"""
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """record training status every 100 global steps"""
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            """calculate gradient and perform gradient descent algorithm to update parameters for Q-net using Adam optimizer"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """set the parameters of target net to be the same as q-net every #target_network_frequency global steps"""
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())



    torch.save(q_network.state_dict(), f"batchSize_{args.batch_size}++lr_{args.learning_rate}++bfSize_{args.buffer_size}++gamma_{args.gamma}++tnf_{args.target_network_frequency}++startE_{args.start_e}++endE_{args.end_e}.pth")
    # show the result of DQN
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(np.arange(len(episode_rewards)), episode_rewards, color='blue', linestyle='-', linewidth=1.0, alpha=0.8)
    # plt.title('Episode Reward',size=22)
    ax.tick_params(labelsize=18)
    plt.xlabel('Epoch', size=10)
    plt.ylabel('Episode Reward', size=10)

    plt.tight_layout()
    plt.savefig(f"batchSize_{args.batch_size}++lr_{args.learning_rate}++bfSize_{args.buffer_size}++gamma_{args.gamma}++tnf_{args.target_network_frequency}++startE_{args.start_e}++endE_{args.end_e}.png",bbox_inches='tight')
    plt.show()


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
    writer.close()
    """close the env and tensorboard logger"""