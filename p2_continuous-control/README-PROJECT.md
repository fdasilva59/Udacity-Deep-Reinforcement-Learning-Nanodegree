# Deep Reinforcement Learning : Continuous Control 

This project repository contains my work for the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Project 2: Continuous Control.

## Project's goal

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, **the goal of the agent is to maintain its position at the target location for as many time steps as possible.**

![In Project 2, train an agent to maintain its position at the target location for as many time steps as possible.](images/reacher.gif)


### About Deep Reinforcement Learning

> [Reinforcement learning](https://skymind.ai/wiki/deep-reinforcement-learning) refers to goal-oriented algorithms, which learn how to attain a complex objective (goal) or maximize along a particular dimension over many steps; for example, maximize the points won in a game over many moves. They can start from a blank slate, and under the right conditions they achieve superhuman performance. Like a child incentivized by spankings and candy, these algorithms are penalized when they make the wrong decisions and rewarded when they make the right ones – this is reinforcement.

In this project I have chosen to use a Policy Based method called [DDPG (Deep Deterministics Policy Gradient)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

### Environment details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). The project environment provided by Udacity is similar to the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment on the Unity ML-Agents GitHub page.

> The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API. 

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

- Set-up: Double-jointed arm which can move to target locations.
- Goal: The agents must move it's hand to the goal location, and keep it there.
- Agents: The environment contains 10 agent linked to a single Brain.
  - The provided Udacity agent versions are Single Agent or 20-Agents
- Agent Reward Function (independent):
  - +0.1 Each step agent's hand is in goal location.
- Brains: One Brain with the following observation/action space.
  - Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
  - Visual Observations: None.
- Reset Parameters: Two, corresponding to goal size, and goal movement speed.
- Benchmark Mean Reward: 30


For this project, Udacity has provided two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Solving the Environment

Depending on the chosen environment for the implementation, there are 2 possibilities:

- Option 1: Solve the First Version
  - The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes. 

- Option 2: Solve the Second Version
  - The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically:
    - After each episode, the rewards that each agent received (without discounting) are added up , to get a score for each agent. This yields 20 (potentially different) scores. The average of these 20 scores is then used.
    - This yields an average score for each episode (where the average is over all 20 agents).

  - The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.


**In my implementation I have chosen to solve the First version of the environment (Single Agent) using the off-policy DDPG algorithm.** The task is episodic, and **in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.**



## Getting started

### Installation requirements

- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Of course you have to clone this project and have it accessible in your Python environment
- Then you have to install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md) (The Unity ML-agant environment is already configured by Udacity)

Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

- Finally, unzip the environment archive in the 'project's environment' directory and eventually adjust the path to the UnityEnvironment in the code.

Note: A conda environment file is provided with this project (so you can check/install the versions of the libraries I used)


    
### Train a agent
    
Execute the provided notebook within this Nanodegree Udacity Online Workspace for "project #2  Continuous Control" (or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code )

Note :
- Manually playing with the environment has not been implemented as it is not available with Udacity Online Worspace (No Virtual Screen)    
- Watching the trained agent playing in the environment has not been implemented neither, as it is not available with Udacity Online Worspace (No Virtual Screen) and not compatible with my personal setup (see Misc : Configuration used  section)

### Misc : Configuration used 

This agent has been trained on my "Deep Learning Dev Box", which is basically a Linux GPU Server, running Docker containers (using Nvidia Docker 2), serving Jupyter Lab notebooks which are accessed remotely via a web interface (or a ssh connection) : unfortunately this setup does not seem suitable to run Unity ML agent, with the GPU and providing a display for for the agent (See [Unity documentation](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-Docker.md) for more details). Thus the headless / no visualization version of the Unity environment was used.