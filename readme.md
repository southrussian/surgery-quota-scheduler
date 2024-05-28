<div style="display: flex; align-items: center;">
    <img alt="description" src="assets/logo.png" width="10%">
    <span style="font-size: 2em; margin-left: 20px;">Surgery Quota Scheduler</span>
</div>

<div>
  

<img alt="python" src="assets/python.png" width="90">
<img alt="pettingzoo" src="assets/pettingzoo.png" width="113">
<img alt="gym" src="assets/gym.png" width="100">

</div>
The Multi-Agent Reinforcement Learning (MARL) environment's Surgery Quota Scheduler is a strategic game that revolves around optimizing surgery schedules. Agents, representing surgeries, must navigate a limited capacity schedule, balancing urgency, complexity, and data completeness to maximize rewards. They can move through time slots, hold positions, or recall their quotes, all while contending with partial visibility of the schedule. The goal is to efficiently allocate surgeries across time slots without overcrowding, and the game concludes once a predetermined number of steps are taken. This system aims to simulate the decision-making process in surgical scheduling, emphasizing the importance of flexibility and foresight.

<h1 style="text-align: center;width: 120%">
    <img alt="description" src="assets/frame.png" width="90%">
</h1>

<h1>Table of contents</h1>

 - [Environment usage](#Environment-usage)
 - [Hyperparameters (by default)](#hyperparameters-by-default)
 - [Calculated parameters](#calculated-parameters)
 - [Rewards](#rewards)
 - [Actions](#actions)
 - [Observations](#observations)
 - [Termination rule](#termination-rule)

## Environment usage
To run the model training process, use a bash script as follows:
```
./run.sh
```
Be careful, this command will do everything for you, but the code execution will take a very long time (up to several days on some PC configurations).

For the command to work you need to use it in the Git Bash terminal. You can find the required terminal in Visual Studio Code, on the terminal bar:

<img alt="gym" src="assets/git.png" width="100%">


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
    
    After that, go back to the directory where the files ```requirements.txt``` and ```fast_predict.py``` are located.

4) Install the necessary dependencies:
    ```
   python -m pip install -r requirements.txt
   ```
   
The result of launch 'human' render mode is down below:
<h1 style="text-align: center;width: 120%">
    <img alt="pygame gif" src="assets/2024-03-26 15.22.44.gif" width="90%">
</h1>


## Hyperparameters (by default)
- Baseline reward: $b=0.2$;
- Number of agents: $N=12$ (by the number of unique combinations of the following parameters);
  
    Each agent consists of a set of these parameters inherent to it:
    - Complexity: 0 or 1, where 0 - an easy task, 1 - a hard one;
    - Completeness: 0 or 1, where 0 - an insufficient data, 1 - a fully covered case;
    - Urgency: ascending from 1 (lowest priority) to 3 (highest priority);
- Time slot capacity: $C=4$;

## Calculated parameters
- Number of steps: $S=N^2/C$;
- Scaling factor: $k=(Complexity+(1 - Completeness))*Urgency$;

## Rewards
- Reschedule a quote forward 1 time step in future: $r=-b*k$;
- Reschedule a quote back 1 time step in time: $r=+b*k$;
- Hold the position: 
  - if $n \leq C$, $r=(C-n)*b$;
  - if $n>C$, $r=-(n-C)*b$;
  where $n$ - current number of agents in focal time slot
- If more than $C$ agents in one time slot and it is the end of the episode: $r=-10$;
- If every agent placed in time slot with no more than 3 other agents at the end of the episode $r=+10$;

## Actions
- Move forward in time;
- Move back in time;
- Hold the position;
- Recall your quote with chance equals to $P=0.05$ (may randomly occur at any time step);

## Observations
At each step, the agent observes a short period of time that includes the number of bids:
- 3 steps forward in time
- 3 steps backward in time.
  
They can also see 10 steps into the future and past, but not all of it. For the example, we take a date two weeks from now, the agent observes the current number of bids for that day, however, we intentionally introduce random variation by multiplying the true value by a randomly chosen coefficient (uniform probability distribution) in the range [0.5...1.5], rounding to the nearest integer. There is no information after 10 steps forward or backward.

## Termination rule
The episode ends when the numbers of steps reaches $S$.

## Launched application
The result of launch 'human' render mode is down below:
<h1 style="text-align: center;width: 120%">
    <img alt="pygame gif" src="assets/2024-03-26 15.22.44.gif" width="90%">
</h1>