<h1>Documentation</h1>      

<h1>Table of contents</h1>

- [ppo-scheduler.py](#ppo-schedulerpy)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Command-line Arguments](#command-line-arguments)
  - [Implementation Details](#implementation-details)
    - [Agent Architecture](#agent-architecture)
    - [Training Loop](#training-loop)
    - [Evaluation](#evaluation)
  - [Logging and Visualization](#logging-and-visualization)
    - [References](#references)
- [scheduler.py](#schedulerpy)
  - [Overview](#overview-1)
  - [Installation](#installation-1)
  - [Environment Details](#environment-details)
    - [Constants](#constants)
    - [Classes](#classes)
    - [Methods](#methods)
  - [Running the Environment](#running-the-environment)
  - [Notes](#notes)

# ppo-scheduler.py

## Overview
This project implements a reinforcement learning (RL) training script for our environment using a Proximal Policy Optimization (PPO) algorithm. The script includes features like seeding for reproducibility, logging with TensorBoard, and optional tracking with Weights and Biases (wandb). The agent is built using PyTorch and trained to optimize surgery scheduling.

## Prerequisites
- Python 3.10+
- PyTorch 
- numpy 
- wandb 
- TensorBoard 
- Scheduler environment (see scheduler.py)

## Installation
``` bash
pip install torch numpy wandb tensorboard
```

## Command-line Arguments
The script accepts several command-line arguments to configure the training process:

- --exp-name: Name of the experiment (default: script name). 
- --seed: Seed for random number generators (default: 0). 
- --torch-deterministic: Toggle for deterministic behavior in PyTorch (default: True). 
- --cuda: Toggle for using CUDA if available (default: True). 
- --track: Toggle for tracking the experiment with Weights and Biases (default: False). 
- --wandb-project-name: Name of the Weights and Biases project (default: "cleanRL"). 
- --wandb-entity: Entity (team) for Weights and Biases project (default: None). 
- --capture-video: Toggle for capturing agent performance videos (default: False). 
- --env-id: ID of the environment (default: "sqc_v1"). 
- --total-timesteps: Total timesteps for the experiment (default: 8000). 
- --learning-rate: Learning rate for the optimizer (default: 2.5e-4). 
- --num-envs: Number of parallel environments (default: 1). 
- --num-steps: Number of steps per policy rollout (default: 64). 
- --anneal-lr: Toggle learning rate annealing (default: True). 
- --gamma: Discount factor (default: 0.99). 
- --gae-lambda: Lambda for Generalized Advantage Estimation (default: 0.95). 
- --batch-scale: Coefficient to scale batch size (default: 1). 
- --num-minibatches: Number of minibatches (default: 4). 
- --update-epochs: Number of epochs to update the policy (default: 4). 
- --norm-adv: Toggle advantages normalization (default: True). 
- --clip-coef: Surrogate clipping coefficient (default: 0.2). 
- --clip-vloss: Toggle for using a clipped loss for the value function (default: True). 
- --ent-coef: Entropy coefficient (default: 0.01).
- --vf-coef: Value function coefficient (default: 0.5). 
- --max-grad-norm: Maximum norm for gradient clipping (default: 0.5). 
- --target-kl: Target KL divergence threshold (default: None).

## Implementation Details
### Agent Architecture
The agent is built using a neural network with the following architecture:

- Input layer: Linear layer with 7 input features 
- Hidden layers: Three hidden layers with ReLU activations 
- Output layers: Separate actor and critic heads 
- Actor head: Linear layer with 3 output features (actions)
- Critic head: Linear layer with 1 output feature (state value)

### Training Loop
The training loop follows these steps:

- Initialize the environment and agent.
- Collect data by interacting with the environment.
- Compute advantages and returns using Generalized Advantage Estimation (GAE).
- Update the policy and value networks using PPO with minibatch optimization.
- Log training metrics and optionally save checkpoints.

### Evaluation
After training, the agent is evaluated in the environment with the option to render the environment and observe the agent's performance.

## Logging and Visualization
TensorBoard: Logs training metrics like loss, learning rate, and performance.

Weights and Biases: Optionally tracks the experiment, including hyperparameters and training metrics.

To visualize training metrics using TensorBoard:

``` bash
tensorboard --logdir runs/

```

### References
Schulman, J., et al. "Proximal Policy Optimization Algorithms." (2017). https://arxiv.org/abs/1707.06347

# scheduler.py

## Overview
The Surgery Quota Scheduler is a reinforcement learning environment designed to simulate scheduling surgery quotas over a period of days. The environment allows multiple agents to interact simultaneously, each making decisions on when to schedule surgeries based on various factors. This environment can be used for training reinforcement learning models to optimize scheduling policies.

## Installation
Before running the environment, ensure you have the necessary dependencies installed. You can install them using the following command:

``` bash
pip install -r requirements.txt
```

Make sure you have the following libraries installed:

- gymnasium
- pygame
- pettingzoo
- names-dataset
- numpy

## Environment Details
### Constants
- C: The number of surgeries that can be scheduled per day.
- N: Total number of agents (patients).
- N_DAYS: The number of days in the scheduling period.
- NUM_ITERS: The number of iterations the environment will run, calculated as (N ** 2) / (N_DAYS * C).
- MOVES: A dictionary mapping actions to position changes (0: move forward, 1: move backward, 2: stay).
- b: A reward coefficient.

### Classes
Agent - Represents an agent in the environment.

- name: The agent's name.
- urgency: Urgency level of the surgery (1 to 3).
- completeness: Completeness of the preparation (0 or 1).
- complexity: Complexity of the surgery (0 or 1).
- k: A derived value from urgency, completeness, and complexity.
- position: Current scheduled day.
- mutation_rate: Probability of changes in the agent's attributes.

SurgeryQuotaScheduler - A custom environment for scheduling surgeries.

- render_mode: Determines how the environment is rendered ('human', 'ansi', or 'rgb_array').
- possible_agents: List of all possible agents.
- agent_name_mapping: A mapping of agent objects to indices.
- num_moves: The number of moves taken so far.

### Methods
- __init__(self, render_mode=None) - Initializes the environment with the given render mode.
- observation_space(self, agent) - Returns the observation space for an agent.
- action_space(self, agent) - Returns the action space for an agent.
- reset(self, seed) - Resets the environment to an initial state using the given seed.
- step(self, actions) - Performs a step in the environment based on the actions taken by the agents.
- render(self) - Renders the environment based on the specified render mode.
- close(self) - Closes the environment and releases any resources.

## Running the Environment
To run the environment, use the following code template:

``` python
import time
import torch
from scheduler import SurgeryQuotaScheduler

# Initialize the environment
env = SurgeryQuotaScheduler(render_mode='human')
agent = Agent()  # Assuming an agent class exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reset the environment
next_observations, _ = env.reset(seed=0)
next_observations = torch.Tensor(list(next_observations.values())).to(device)

while True:
    next_actions, _, _, _ = agent.get_action_and_value(next_observations)
    next_observations, next_rewards, next_terminations_list, next_truncations_list, _ = env.step(
        dict(zip(env.possible_agents, next_actions.cpu().numpy()))
    )
    next_observations = torch.Tensor(list(next_observations.values())).to(device)
    next_truncations = max(next_truncations_list.values())
    next_terminations = max(next_terminations_list.values())

    if next_truncations or next_terminations:
        break

# Close the environment
env.close()
```

## Notes
- Ensure the agent's class (Agent) has the appropriate methods to interact with the environment, such as get_action_and_value.
- Customize the constants and logic according to the specific requirements of your scheduling problem.
- Use the provided rendering methods to visualize the scheduling process.

This environment is designed to be flexible and customizable, allowing you to test various reinforcement learning algorithms and scheduling strategies.