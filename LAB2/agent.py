# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import random
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon_decay, alpha, gamma, epsilon, alpha_decay):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.Q_value = [[0] * 4 for _ in range(48)]
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.alpha = alpha
        self.gamma = gamma


    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        flag = random.random() > self.epsilon
        if flag:
            max_value = float("-inf")
            for idx, value in enumerate(self.Q_value[observation]):
                if value > max_value:
                    max_value = value
                    action = idx
        else:
            action = np.random.choice(self.all_actions)
        return action

    def learn(self, s, a, s_, r):
        """learn from experience"""
        action = self.choose_action(s_)
        self.Q_value[s][a] += self.alpha * (r + self.gamma * self.Q_value[s_][action] - self.Q_value[s][a])
        return

    def get_epsilon(self):
        """You can add other functions as you wish."""
        return self.epsilon

    def decay(self):
        self.epsilon *= self.epsilon_decay
        self.alpha *= self.alpha_decay

    def start_testing(self):
        self.epsilon = 0

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon_decay, alpha, gamma, epsilon, alpha_decay):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.Q_value = [[0] * 4 for _ in range(48)]
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        flag = random.random() > self.epsilon
        if flag:
            max_value = float("-inf")
            for idx, value in enumerate(self.Q_value[observation]):
                if value > max_value:
                    max_value = value
                    action = idx
        else:
            action = np.random.choice(self.all_actions)
        return action

    def learn(self, s, a, s_, r):
        """learn from experience"""
        self.Q_value[s][a] += self.alpha * (r + self.gamma * max(self.Q_value[s_]) - self.Q_value[s][a])
        return

    def get_epsilon(self):
        """You can add other functions as you wish."""
        return self.epsilon

    def decay(self):
        self.epsilon *= self.epsilon_decay
        self.alpha *= self.alpha_decay

    def start_testing(self):
        self.epsilon = 0


    ##### END CODING HERE #####
