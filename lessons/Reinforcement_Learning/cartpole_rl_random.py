import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CartPole-v0')
environment.reset()
for step in range(1, 1001):
    environment.render()
    random_action = environment.action_space.sample()
    observation, reward, done, info = environment.step(random_action)
    print(observation)
    if done:
        print("This completed in {} steps".format(step))
        break
