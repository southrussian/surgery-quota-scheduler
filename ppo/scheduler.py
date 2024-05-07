import sys
import time
import random
import functools
import gymnasium
import pygame
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from names_dataset import NameDataset


C = 4
N = 16
N_DAYS = 14
NUM_ITERS = (N ** 2) / C
MOVES = {0: 1,
         1: -1,
         2: 0}
b = 0.2

nd = NameDataset()
popular_first_names = nd.get_top_names(country_alpha2='US', n=100)
popular_surnames = nd.get_top_names(country_alpha2='US', n=100,
                                    use_first_names=False)


def get_random_name():
    if random.randint(0, 1) == 0:
        return (random.choice(popular_first_names['US']['M']) + ' '
                + random.choice(popular_surnames['US']))
    else:
        return (random.choice(popular_first_names['US']['F']) + ' '
                + random.choice(popular_surnames['US']))


def draw_boxes(win, win_width, win_height, calendar):
    win.fill((0, 0, 0))
    box_width = win_width // 16
    for day, n in calendar.items():
        if n < 4:
            color = (0, 255, 0)  # Green
        elif n == 4:
            color = (255, 165, 0)  # Orange
        else:
            color = (255, 0, 0)  # Red
        pygame.draw.rect(win, color, (day * box_width, win_height // 2, box_width, 45))
        font = pygame.font.Font(None, 21)
        text = font.render("Day " + str(day), 1, (255, 255, 255))
        win.blit(text,
                 (day * box_width + box_width // 2 - text.get_width() // 2, win_height // 2 - text.get_height() - 5))
        text = font.render(str(n), 1, (10, 10, 10))
        win.blit(text, (day * box_width + box_width // 2 - text.get_width() // 2, win_height // 2))
    pygame.display.flip()


def game(win, win_width, win_height, observations):
    draw_boxes(win, win_width, win_height, convert_to_calendar(observations))
    time.sleep(0.5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def reward_map(slot_occupancy, position, end_of_episode, k, action, b):
    reward = 0

    if end_of_episode:
        if slot_occupancy[position] >= 4:
            reward = -10
        else:
            reward = 10

    if action.any() == 0:
        reward = b * k
    elif action.any() == 1:
        reward = -b * k
    elif action.any() == 2:
        if slot_occupancy[position] != 0:
            reward = -(slot_occupancy[position] - 4) * b
        else:
            reward = 0

    reward = round(reward, 1)

    if reward == -0.0:
        reward = 0.0

    return reward


class agent():
    def __init__(self, name):
        self.name = name
        self.urgency = 1
        self.comleteness = 1
        self.complexity = 0
        self.k = (self.complexity + (1 - self.comleteness)) * self.urgency
        self.position = 0


class surgery_quota_scheduler(ParallelEnv):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None):
        self.possible_agents = [agent(name=get_random_name()) for _ in range(N)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(7)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn('You are calling render mode without \
                                   specifying any render mode.')
            return

        if self.render_mode == "ansi":
            if len(self.agents) == N:
                string = 'Current state: \n'
                for idx, agent in enumerate(self.agents):
                    string += f'{agent.name}: {self.state[self.agents[idx]]}, '
            else:
                string = 'Game over'
        if self.render_mode == "human":
            pygame.init()
            win_width, win_height = 1000, 600
            win = pygame.display.set_mode((win_width, win_height))
            pygame.display.set_caption("Surgery Quota Scheduler")
            game(win, win_width, win_height, {agent: agent.position for agent in self.agents})

    def close(self):
        pass

    def reset(self, seed):
        np.random.seed(seed)
        self.agents = self.possible_agents[:]
        for agent in self.agents:
            agent.urgency = np.random.randint(1, 4)
            agent.comleteness = np.random.randint(0, 2)
            agent.complexity = np.random.randint(0, 2)
            agent.position = np.random.randint(0, 14)
            agent.k = (agent.comleteness + (1 - agent.complexity)) * agent.urgency
        self.num_moves = 0
        slot_occupancy = convert_to_calendar({agent: agent.position for agent in self.agents})
        observations = {agent: (agent.urgency, agent.comleteness, agent.complexity,
                                agent.position,
                                slot_occupancy.get(agent.position - 1, 0), slot_occupancy.get(agent.position, 0),
                                slot_occupancy.get(agent.position + 1, 0))
                        for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):

        self.num_moves += 1
        end_of_episode = self.num_moves == NUM_ITERS - 1
        if end_of_episode:
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
        else:
            terminations = {agent: False for agent in self.agents}
            truncations = {agent: False for agent in self.agents}

        slot_occupancy = convert_to_calendar({agent: agent.position for agent in self.agents})

        rewards = {agent: reward_map(slot_occupancy, agent.position, end_of_episode, agent.k, actions[agent], b) for
                   agent in self.agents}

        for agent in self.agents:
            agent.position += MOVES[int(actions[agent])]
            if agent.position == N_DAYS:
                agent.position = N_DAYS - 1
            if agent.position < 0:
                agent.position = 0

        observations = {agent: (agent.urgency, agent.comleteness, agent.complexity,
                                agent.position,
                                slot_occupancy.get(agent.position - 1, 0), slot_occupancy.get(agent.position, 0),
                                slot_occupancy.get(agent.position + 1, 0))
                        for agent in self.agents}

        self.state = observations

        infos = {agent: {} for agent in self.agents}

        self.render()

        return observations, rewards, terminations, truncations, infos


def convert_to_calendar(observations):
    dates = {day: 0 for day in range(N_DAYS)}
    for date in observations.values():
        if 0 <= date < N_DAYS:
            dates[date] += 1
    return dates