import gym
import numpy as np

environment = gym.make('CartPole-v0')
environment.reset()

def determine_action(observation, weights):
    """
        Returns 0 (go left) or 1 (go right)
        depending on whether the weighted sum of weights * observations > 0
    """
    weighted_sum = np.dot(observation, weights)
    action = 0 if weighted_sum < 0 else 1
    return action


def run_episode(environment, weights):
    """
    This function runs one episode with a given set of weights
    and returns the total reward with those weights
    """
    observation = environment.reset()
    total_reward = 0
    for step in range(200):
        action = determine_action(observation, weights)
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def find_best_weights(num_episodes):
    """
        This function runs a number of episodes and picks random weights for each one and evaluates
        the reward given by the weights.
        It returns the weights for the best episode.
    """
    best_weights = None
    best_reward = 0
    observation = environment.reset()
    for episode in range(num_episodes):
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(environment, weights)
        if reward > best_reward:
            best_weights = weights
            best_reward = reward
            print("Current Best Weights at episode #{} are {}".format(episode, best_weights))
    return best_weights, best_reward

# Now let's run 300 different weights and pick the best one (the one that gave the highest reward)
best_weights, best_reward = find_best_weights(num_episodes=300)
print("The best weights we've seen are: {}".format(best_weights))

# Now that we've found the best weights we've seen
# let's use our best weights to run our program
observation = environment.reset()
cumulative_reward = 0

for step in range(0, 200):
    environment.render()
    action = determine_action(observation, best_weights)
    observation, reward, done, info = environment.step(action)
    cumulative_reward += reward
    if done:
        print("Reward when done: {}".format(cumulative_reward))
        if cumulative_reward == 200:
            print("Congrats! You successfully solved Cartpole V-0!")
        else:
            print("Unfortunately, even the best weights weren't enough")
        break
