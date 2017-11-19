import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CartPole-v0')
environment.reset()
NUM_EPISODES = 10

for episode in range(NUM_EPISODES):
    observation = environment.reset()
    for _ in range(1000):
        environment.render()
        print(observation)
        action = environment.action_space.sample() # this is a 1 or 0
        observation, reward, done, info = environment.step(action)
        pos, veloc, angle, ang_veloc = observation
        if done:
            break
