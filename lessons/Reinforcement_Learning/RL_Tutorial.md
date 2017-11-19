# Reinforcement Learning with OpenAI's Gym
Teacher: Carl Shan

School: Nueva School

Acknowledgements: This tutorial borrows heavily from the following resources:
1. [Kevin Frans' wonderful introduction to CartPole and Keras](http://kvfrans.com/simple-algoritms-for-solving-cartpole/) and 
2. [Keon Kim's in-depth tutorial](https://keon.io/deep-q-learning/) for ideas and code.

## Introduction
This tutorial will teach you how to get started with OpenAI's popular `gym` library and to use `Keras` to do Deep Q-Learning.

### Setup
We must first install `gym`. Run the following commands in terminal:

```
sudo pip3 install gym
```

We will also need a code editor for the rest of this tutorial. We will not be using Jupyter Notebooks since they are not yet compatible with rendering the `gym` games we want to train our models to learn how to play.

Instead, please download a code editor such as [Atom](https://www.atom.io) before proceeding.

### Getting Started
We are going to get started with a very basic game called CartPole.

The goal of the game is to balance a stick standing vertically on top of a cart as the cart moves around. 

You can see this [video](https://gym.openai.com/docs/) on OpenAI's gym documentation for an example of what CartPole looks like.

Now open your text editor and paste in the following code:

```python
import gym

environment = gym.make('CartPole-v0')
environment.reset()
for _ in range(1000):
    environment.render()

```

Save it as `cartpole_rl.py`.

To run this code, you need to use Terminal. Open it up and run the above program (make sure you `cd` into the same directory as this file):

```
python3 cartpole_rl.py
```

You should see this the CartPole game pop up on your screen:

![CartPole](https://keon.io/images/deep-q-learning/animation.gif)

*Source: https://keon.io/deep-q-learning*

Now let's get build up an increasingly sophisticated method of playing this game.

Modify the code in your file to the following:

```python
import gym

environment = gym.make('CartPole-v0')
environment.reset()
for _ in range(1000):
    environment.render()
    random_action = environment.action_space.sample()
    print(random_action)
    environment.step(random_action)
```

In the lines of code above the command `environment.action_space.sample()` will take a random action e.g., going either *left* (represented as the value `0`) or *right* (encoded as `1`). 

So if you go to your Terminal you should see a bunch of `0`'s and `1`'s being printed.

Now this is a pretty silly way of playing the game; it doesn't take into account whether the steps we're taking are helping or hurting. Let's do something slightly more intelligent.

However, before we dive deeper, it's important to understand that the `environment.step()` function actually returns some valuable information.

[OpenAI's description](https://gym.openai.com/docs/) of the values that are returned are below:

-----------

> ### **Observations**
>
> If we ever want to do better than take random actions at each step, it'd probably be good to actually know what our actions are doing to the environment.
> The environment's `step` function returns exactly what we need. In fact, `step` returns four values. These are:
>
> 1. `observation` (**object**): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
> 2. `reward` (**float**): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
>  3. `done` (**boolean**): whether it's time to `reset` the environment again.  Most (but not all) tasks are divided up into well-defined episodes, and `done` being `True` indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
> 4. `info` (**dict**): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.
>
> This is just an implementation of the classic "agent-environment loop". Each timestep, the agent chooses an `action`, and the environment returns an `observation` and a `reward`.
>
> ![RL Process](https://gym.openai.com/assets/docs/aeloop-138c89d44114492fd02822303e6b4b07213010bb14ca5856d2d49d6b62d88e53.svg)
>
> The process gets started by calling `reset`, which returns an `initial` observation.
