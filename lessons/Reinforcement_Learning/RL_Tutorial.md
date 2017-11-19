# Reinforcement Learning with OpenAI's Gym
Teacher: Carl Shan

School: Nueva School

Acknowledgements: Nearly all of the code and ideas in tutorial borrows heavily from the following fantastic resources:
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

So now that we have learned the above, we're going to make use of these four variables one-by-one.

First off, let's follow `gym`'s documentation instructions to use the `done` variable:

```python
import gym

environment = gym.make('CartPole-v0')
environment.reset()
for step in range(1, 1001):
    environment.render()
    random_action = environment.action_space.sample()
    observation, reward, done, info = environment.step(random_action)
    if done:
        print("This completed in {} steps".format(step))
        break
```

In Python the `break` command will exit the loop.

Now, the `observation` variable is itself a list of 4 elements long. The elements are the following:

`[position of cart, velocity of cart, angle of pole, rotation rate of pole]`

So that's break apart that variable and use its components to figure out which action to take:

```python
import gym

environment = gym.make('CartPole-v0')
environment.reset()
for step in range(1, 1001):
    environment.render()
    random_action = environment.action_space.sample()
    observation, reward, done, info = environment.step(random_action)
    pos, veloc, angle, ang_veloc = observation
    if done:
        print("This completed in {} steps".format(step))
        break
```

Our goal now is to write a funcion that can take in these four variables and using them in some intelligent manner.

Here's a basic idea that starts to feel more like the machine learning we've been doing: let's find weights for these observations, and if the weighted sum of these observations is positive, let's go right (return `1`) else we'll make the cart go left (return `0`).

I've written the function for you below:

```python
import numpy as np

def determine_action(observations, weights ):
	action = 0 if np.dot(weghts, observation) < 0 else 1 
	return action
```

Cool.

So now we can use this function to pick an action.

But wait a second. How do we pick weights?

Well, just like with Neural Networks let's initialize these weights to random numbers and we'll "learn" the best weights later.

```python
# This will create a vector of length 4 where each value is nitializing chosen randomly between [-1, 1]. 
weights = np.random.rand(4) * 2 - 1
```

Great. 

Now let's figure out how to pick "better" weights. 

In order to pick "better" parameters we need to have an idea of what "better" means. Luckily, CartPole gives us that through the `reward` value that's returned by the `step()` function.

Using that let's write a function that basically tells us how much a set of weights will reward  us:

```python
def run_episode(env, weights):  
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        action = determine_action(observations, weights)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward
```

Okay, so for a given set of weights we can calculate how "good" they are in our CartPole game.

Now let's figure out a smart way to update our weights.

### Randomly Searching
*The below is from [Kevin Fran's post on "Simple Algorithms for Solving Cartpole](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)
*
> One fairly straightforward strategy is to keep trying random weights, and pick the one that performs the best.
> 
> ```python
> bestweights = None  
> bestreward = 0  
> 
> # Let's search through 10000 different random weights
> for _ in xrange(10000):  
>     weights = np.random.rand(4) * 2 - 1
>     reward = run_episode(env, weights)
>     if reward > bestreward:
>        bestreward = reward
>        bestweights = weights
>        # CartPole is considered solved if the agent lasts 200 timesteps
>        if reward == 200:
>            break
> ```
> 
> Since the CartPole environment is relatively simple, with only 4 observations, this basic method works surprisingly well.
> 
> ![Random Search](http://kvfrans.com/content/images/2016/07/cartpole-random-1.png)
> 
> I ran the random search method 1,000 times, keeping track of how many episodes it took until the agent kept the pole up for 200 timesteps. On average, it took 13.53 episodes.

### Now What?
Okay, so that's how we can find the "best weights" when we're just searching randomly through the space of all possible weights? 

Take a look at [Kevin Fran's tutorial](http://kvfrans.com/simple-algoritms-for-solving-cartpole/) and implement some of the other solutions.

For this assignment, Submit your code and a writeup that responds to the following questions:

* Explain in your own words how the Hill-Climbing or Policy Gradient solutions are similar or different from the Random Search solution I show in my tutorial.
* What are some advantages and disadvantages to these other ways of searching for weights?



