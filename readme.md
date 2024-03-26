# Surgery-quota-scheduler

The Multi-Agent Reinforcement Learning (MARL) environment's Surgery Quota Scheduler is a strategic game that revolves around optimizing surgery schedules. Agents, representing surgeries, must navigate a limited capacity schedule, balancing urgency, complexity, and data completeness to maximize rewards. They can move through time slots, hold positions, or recall their quotes, all while contending with partial visibility of the schedule. The goal is to efficiently allocate surgeries across time slots without overcrowding, and the game concludes once a predetermined number of steps are taken. This system aims to simulate the decision-making process in surgical scheduling, emphasizing the importance of flexibility and foresight.

![Frame 2 (1)](https://github.com/artemisak/Surgery-quota-scheduler/assets/76273674/ff8eb8a2-d7ee-48ee-ac8c-597fd0ba5022)

## Environment usage
The following environment based on ideas of Gymnasium & PettingZoo projects, using their frameworks to create a versatile and interactive platform of surgery quotas scheduling for reinforcement learning and multi-agent systems.

There are two render modes you can use to launch this environment:
- 'ansi' render mode launches the console output of current states in the game
- 'human' render mode launches the pygame window with actual animated states of quotas for all days in calendar

To launch each of these render modes change the following line of code in main.py:
```python
env = parallel_env(render_mode='human') # or 'ansi'
```
The result of launch 'human' render mode is down below:



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