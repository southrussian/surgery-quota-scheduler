<div style="display: flex; align-items: center;">
    <img alt="description" src="assets/logo.png" width="10%">
    <h1 style="font-size: 6em; margin-left: 30px; display: flex; align-items: center; vertical-align: baseline;">Surgery Quota Scheduler</h1>
</div>


<div>
  

<img alt="python" src="assets/python.png" height="60">
<img alt="pettingzoo" src="assets/pettingzoo.png" height="60">
<img alt="gym" src="assets/gym.png" height="60">

</div>
The Multi-Agent Reinforcement Learning (MARL) environment's Surgery Quota Scheduler is a strategic game that revolves around optimizing surgery schedules. Agents, representing surgeries, must navigate a limited capacity schedule, balancing urgency, complexity, and data completeness to maximize rewards. They can move through time slots, hold positions, or recall their quotes, all while contending with partial visibility of the schedule. The goal is to efficiently allocate surgeries across time slots without overcrowding, and the game concludes once a predetermined number of steps are taken. This system aims to simulate the decision-making process in surgical scheduling, emphasizing the importance of flexibility and foresight.

<h1 style="text-align: center;width: 120%">
    <img alt="description" src="assets/frame.png" width="90%">
</h1>

<h1>Table of contents</h1>

 - [Environment usage](#Environment-usage)
 - [Environment Parameters](#environment-parameters)
 - [Agents](#agents)
 - [State and Action Spaces](#state-and-action-spaces)
 - [Reward Function](#reward-function)
 - [Termination Rules](#termination-rules)
 - [Environment Dynamics](#environment-dynamics)
 - [State Space and Combinations](#state-space-and-combinations)
 - [Launched application](#launched-application)

## Environment usage
To run the simulation, use a bash script as follows:
```
./run.sh
```
Be careful, this command will do everything for you, but the code execution will take a very long time (up to several days on some PC configurations).

For the command to work you need to use it in the Git Bash terminal. You can find the required terminal in Visual Studio Code, on the terminal bar:

<img alt="gym" src="assets/git.png" width="80%">


In order to quickly reproduce the results of numerical experiments follow these steps:
1) Install Python 3.12
2) Create a virtual environment:
    ```
    python -m venv venv
    ```
3) Activate the virtual environment

    For Windows/Linux, go to the ```env/Scripts``` folder and run the script in the Powershell console
    ```
    ./activate
    ```
    If you use the Windows integrated command console instead of Powershell the activation will look like this:
    ```
    .\activate
    ```
    On a Mac, the procedure will be similar, but the directory you need will be at ``env/bin``.
    
    After that, go back to the directory where the files ```requirements.txt``` and ```ppo-scheduler.py``` are located.

4) Install the necessary dependencies:
    ```
   python -m pip install -r requirements.txt
   ```

## Environment Parameters

1. **Number of Agents (N)**: 28 agents participate in this environment.
2. **Number of Days (N_DAYS)**: The setup includes 14 days during which agents interact.
3. **Number of Iterations (NUM_ITERS)**: Calculated by the formula $\(\frac{N^2}{N_{DAYS} \cdot C}\)$, where $C = 4$. This determines the total number of steps in an episode.
4. **Moves**: 
- Move forward in time: $0$;
- Move back in time: $1$;
- Hold the position: $2$;
- Possible agent movements are encoded as $\( \{0 \rightarrow +1, 1 \rightarrow -1, 2 \rightarrow 0\} \)$.
5. **Base Reward Parameter (b)**: Set to $0.2$.

## Agents

Each agent is characterized by the following parameters:
- **Name**: Randomly generated from a list of popular names and surnames.
- **Urgency**: Takes values from the set $\(\{1, 2, 3\}\)$.
- **Completeness**: Takes values from the set $\(\{0, 1\}\)$.
- **Complexity**: Takes values from the set $\(\{0, 1\}\)$.
- **Position**: Agent's position within the range $\(\{0, 1, ..., 13\}\)$.
- **Coefficient (k)**: Calculated as $\ k = (\text{complexity} + (1 - \text{completeness})) \times \text{urgency} \$.
- **Mutation Rate**: Ranges from 0 to 1.

## State and Action Spaces

- **Observation space**: Discrete space represented by a set of size 7 $(\(\mathbb{O} = Discrete(7)\))$.
- **Action space**: Also discrete, containing 3 possible actions $(\(\mathbb{A} = Discrete(3)\))$.

Actions of each agent are denoted as $\(a_i \in A\)$, where $\(A\)$ is the set of all possible actions.

## Reward Function

An agent's reward is determined by its position and the chosen action according to the following formula:

$reward(a, p, s, k, action) = $

$\- b * k + (s[p] - 1) * b, if  action = 0$

$\- b * k + (s[p] - 2) * b, if  action = 1$

$\- (s[p] - 4) * b, if  action = 2,$

where $(s[p]$ is the number of agents at position $p$, and $b = 0.2$.

## Termination Rules

Environment termination occurs under the following conditions:
- If the number of iterations $NUM MOVES$ reaches $NUM ITERS - 1$ and more than 80% of agents choose action $2$:

$\text{termination} = \text{True}, \quad \text{if } \frac{\sum \text{actions} = 2}{N} \geq 0.8$, $\text{False, otherwise}$


- If the number of iterations reaches $(2 \times NUMITERS - 1\)$:

$\text{truncation} = \text{True}, \text{if } \text{NUM MOVES} = 2 \times NUM ITERS - 1; \text{False}, \text{otherwise}$

## Environment Dynamics

1. **Position Update**: Agent's position changes according to the chosen action. The position is bounded within the range $[0, N_{DAYS}-1]$.
2. **Mutation Level**: If an agent's position exceeds half of the days $N_{DAYS}/2$:
    - Mutation level increases if the action is 0.
    - Mutation level decreases if the action is 1.
3. **Agent Parameter Changes**: Depending on the mutation level, urgency, completeness, and complexity parameters of agents may change.

**Environment Initialization**

Upon environment reset, agent parameters and positions are initialized randomly within specified ranges. Initial observations and information are updated according to the current environment state.

## State Space and Combinations

The environment state space is discrete and defined by the set of parameters of all agents. Each agent can be assigned to one of the 14 days, resulting in a large number of possible system configurations.

Considering possible combinations of 4 agents over 14 days, the number of combinations can be expressed using the binomial coefficient.

$\[ \binom{28}{4} = \frac{28!}{4!(28-4)!} = 20475 \]$

Therefore, in this environment, it is possible to have 20475 different combinations of agents over 4 days out of 14. These combinations create a rich state space, allowing modeling of diverse scenarios and strategies.


## Launched application
The result of launch 'human' render mode is down below:
<h1 style="text-align: center;width: 120%">
    <img alt="pygame gif" src="assets/2024-03-26 15.22.44.gif" width="90%">
</h1>