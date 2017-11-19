import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CartPole-v0')
environment.reset()
for _ in range(1000):
    environment.render()
    # random_action = environment.action_space.sample()
    # print(random_action)
    environment.step(1)
