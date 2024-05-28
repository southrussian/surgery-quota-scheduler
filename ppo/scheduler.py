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
N = 28
N_DAYS = 14
NUM_ITERS = int((N ** 2) / (N_DAYS * C))
MOVES = {0: 1, 1: -1, 2: 0}
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
    box_height = 45
    center_x = win_width // 2
    center_y = win_height // 2
    for day, n in calendar.items():
        if n < 4:
            color = (0, 255, 0)
        elif n == 4:
            color = (255, 165, 0)
        else:
            color = (255, 0, 0)
        x = center_x - box_width * 7 + box_width * day
        y = center_y - box_height * 1.5
        pygame.draw.rect(win, color, (x, y, box_width, box_height))
        font = pygame.font.Font(None, 21)
        text = font.render("Day " + str(day), 1, (255, 255, 255))
        text_x = x + box_width // 2 - text.get_width() // 2
        text_y = y - text.get_height() * 1.5
        win.blit(text, (text_x, text_y))
        text = font.render(str(n), 1, (10, 10, 10))
        text_x = x + box_width // 2 - text.get_width() // 2
        text_y = y + box_height // 2 - text.get_height() // 2
        win.blit(text, (text_x, text_y))
    pygame.display.flip()



def game(win, win_width, win_height, observations):
    draw_boxes(win, win_width, win_height, convert_to_calendar(observations))
    time.sleep(0.5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def convert_to_calendar(observations):
    dates = {day: 0 for day in range(N_DAYS)}
    for date in observations.values():
        if 0 <= date < N_DAYS:
            dates[date] += 1
    return dates


def reward_map(slot_occupancy, position, end_of_episode, k, action):
    reward = 0
    if action == 0:
        reward = b * k - (slot_occupancy[position] - 1) * b
    elif action == 1:
        reward = - b * k - (slot_occupancy[position] - 2) * b
    elif action == 2:
        reward = -(slot_occupancy[position] - 4) * b
    reward = round(reward, 1)
    return reward if reward != -0.0 else 0.0


class Agent:
    def __init__(self, name):
        self.name = name
        self.urgency = 1
        self.completeness = 1
        self.complexity = 0
        self.k = (self.complexity + (1 - self.completeness)) * self.urgency
        self.position = 0
        self.mutation_rate = 0


class SurgeryQuotaScheduler(ParallelEnv):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None):
        self.num_moves = 0
        self.possible_agents = [Agent(name=get_random_name()) for _ in range(N)]
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
        pygame.quit()
        self.win = None
        self.possible_agents = None
        self.agent_name_mapping = None
        self.render_mode = None
        self.agents = None
        self.num_moves = None
        self.state = None

    def reset(self, seed):
        np.random.seed(seed)
        self.agents = self.possible_agents[:]
        for agent in self.agents:
            agent.urgency = np.random.randint(1, 4)
            agent.completeness = np.random.randint(0, 2)
            agent.complexity = np.random.randint(0, 2)
            agent.position = np.random.randint(0, 14)
            agent.k = (agent.completeness + (1 - agent.complexity)) * agent.urgency
        self.num_moves = 0
        slot_occupancy = convert_to_calendar({agent: agent.position for agent in self.agents})
        observations = {agent: (agent.urgency, agent.completeness, agent.complexity,
                                agent.position,
                                slot_occupancy.get(agent.position - 1, 0), slot_occupancy.get(agent.position, 0),
                                slot_occupancy.get(agent.position + 1, 0))
                        for agent in self.agents}
        infos = (self.num_moves, slot_occupancy)
        self.state = observations
        return observations, infos

    def step(self, actions):

        self.num_moves += 1

        if (self.num_moves >= NUM_ITERS - 1) and (
                sum(1 for action in actions.values() if action == 2) / len(self.agents) >= 0.8):
            terminations = {agent: True for agent in self.agents}
        else:
            terminations = {agent: False for agent in self.agents}

        if self.num_moves == NUM_ITERS * 2 - 1:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
            agent.position += MOVES[int(actions[agent])]
            if agent.position >= N_DAYS:
                agent.position = N_DAYS - 1
            if agent.position < 0:
                agent.position = 0
            if agent.position > N_DAYS / 2:
                if int(actions[agent]) == 0:
                    if agent.mutation_rate != 1.0:
                        agent.mutation_rate = min(agent.mutation_rate + 0.025, 1.0)
                elif int(actions[agent]) == 1:
                    if agent.mutation_rate != 0.0:
                        agent.mutation_rate = max(agent.mutation_rate - 0.025, 0.0)
            else:
                agent.mutation_rate = 0.0

            if np.random.choice([1, 0], p=[agent.mutation_rate, 1 - agent.mutation_rate]) == 1:
                if agent.urgency != 3:
                    agent.urgency += 1

            if np.random.choice([1, 0], p=[agent.mutation_rate / 2, 1 - agent.mutation_rate / 2]) == 1:
                if agent.complexity != 1:
                    agent.complexity = 1

            if np.random.choice([1, 0], p=[min(agent.mutation_rate * 2, 1),
                                           1 - min(agent.mutation_rate * 2, 1)]) == 1:
                if agent.completeness != 1:
                    agent.completeness = 1

        slot_occupancy = convert_to_calendar({agent: agent.position for agent in self.agents})

        rewards = {agent: reward_map(slot_occupancy, agent.position,
                                     max(max(terminations.values()), max(truncations.values())),
                                     agent.k, actions[agent]) for agent in self.agents}

        observations = {agent: (agent.urgency, agent.completeness, agent.complexity,
                                agent.position,
                                slot_occupancy.get(agent.position - 1, 0),
                                slot_occupancy.get(agent.position, 0),
                                slot_occupancy.get(agent.position + 1, 0))
                        for agent in self.agents}

        self.state = observations

        infos = (self.num_moves, slot_occupancy)

        self.render()

        return observations, rewards, terminations, truncations, infos
