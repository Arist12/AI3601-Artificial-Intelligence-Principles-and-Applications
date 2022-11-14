# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

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

##### START CODING HERE #####
config = {
    "epsilon_decay": 0.99,
    "alpha": 1,
    "gamma": 0.9,
    "epsilon": 1,
    "alpha_decay": 0.99
}

# construct the intelligent agent.
agent = QLearningAgent(all_actions, config["epsilon_decay"], config["alpha"], config["gamma"], config["epsilon"], config["alpha_decay"])

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
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, s_, r)
        s = s_
        if isdone:
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)

    agent.decay()
    # early exit if the policy is already converged
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

##### END CODING HERE #####


