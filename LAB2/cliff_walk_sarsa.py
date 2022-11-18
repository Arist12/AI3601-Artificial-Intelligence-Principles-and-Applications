# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import matplotlib.pyplot as plt
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

####### START CODING HERE #######
config = {
    "epsilon_decay": 0.99,
    "alpha": 1,
    "gamma": 1,
    "epsilon": 1,
    "alpha_decay": 0.99,
    "print_result": True,
    "early_stop": True
}

# construct the intelligent agent.
agent = SarsaAgent(all_actions, config["epsilon_decay"], config["alpha"], config["gamma"], config["epsilon"], config["alpha_decay"])

# training data required to keep in track
episode_rewards = []
epsilon_values = []
best_rewards = (float("-inf"), -1)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        action = agent.choose_action(s_)
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, s_, r, action)
        s, a = s_, action
        if isdone:
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)

    agent.decay()
    # early exit if the policy is already converged
    if config["early_stop"]:
        if episode_reward > best_rewards[0]:
            best_rewards = (episode_reward, episode)
        elif episode - best_rewards[1] >= 100:
            print("early stop👍!!")
            break
    episode_rewards.append(episode_reward)
    epsilon_values.append(agent.get_epsilon())

print('\ntraining over\n')
print('Start testing✌✌✌')

agent.start_testing()
s = env.reset()
env.render()
while True:
    a = agent.choose_action(s)
    s_, r, isdone, info = env.step(a)
    time.sleep(0.5)
    env.render()
    s = s_
    if isdone:
        break

# close the render window after training.
env.close()

if config["print_result"]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = np.arange(episode) if config["early_stop"] else np.arange(1000)
    ax.plot(x, episode_rewards, color='blue', linestyle='-', linewidth=1.0, alpha=0.7)
    # plt.title('Episode Reward',size=22)
    ax.tick_params(labelsize=18)
    plt.xlabel('Epoch', size=25)
    plt.ylabel('Episode Reward', size=25)

    plt.tight_layout()
    plt.savefig('sarsa_1.png',bbox_inches='tight')
    plt.show()

    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, epsilon_values, color='blue', linestyle='-', linewidth=1.0, alpha=0.8)
    # plt.title('Episode Reward',size=22)
    ax.tick_params(labelsize=18)
    plt.xlabel('Epoch', size=25)
    plt.ylabel('Epsilon Value', size=25)

    plt.tight_layout()
    plt.savefig('sarsa_2.png',bbox_inches='tight')
    plt.show()
####### END CODING HERE #######


