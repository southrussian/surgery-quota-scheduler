import sys
import time
import random
import functools
import gymnasium
import pygame
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from names_dataset import NameDataset

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


C = 4
N = 12
NUM_ITERS = N**2/C
MOVES = {0: 1,
         1: -1,
         2: 0,
         3: 0}
b = 0.2


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    env = parallel_env(render_mode=render_mode)
    # env = parallel_to_aec(env)
    return env


def reward_map(slot_occupancy, end_of_episode, k, action, b):
    reward = 0

    if end_of_episode:
        if all(n <= 3 for n in slot_occupancy):
            reward = 10
        else:
            reward = -10

    if action == 0:
        reward = b * k
    elif action == 1:
        reward = -b * k
    elif action == 3:
        for n in slot_occupancy:
            if n > 4:
                reward = -(n - 4) * b
            elif n < 4:
                reward = (4 - n) * b
            else:
                reward = 0

    return reward


class agent():
    def __init__(self, name):
        self.name = name
        self.urgency = random.randint(1, 3)
        self.comleteness = random.randint(0, 1)
        self.complexity = random.randint(0, 1)
        self.k = (self.complexity + (1 - self.comleteness))*self.urgency
        self.position = 1


class parallel_env(ParallelEnv):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None):
        self.possible_agents = [agent(name=get_random_name()) for r in range(N)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def obesrvation_spacpe(self, agent):
        return Discrete(3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

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
            print(string)
        if self.render_mode == "human":
            game()


    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: agent.position for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):

        slot_occupancy = [0] * N
        for action in actions.values():
            slot_occupancy[action] += 1

        end_of_episode = self.num_moves >= NUM_ITERS

        rewards = {agent: reward_map(slot_occupancy, end_of_episode, agent.k, actions[agent], b) for agent in self.agents}

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        for agent in self.agents:
          agent.position += MOVES[int(actions[agent])]
          if agent.position > N:
            agent.position = N
          if agent.position < 1:
            agent.position = 1
        observations = {agent: agent.position for agent in self.agents}
        self.state = observations

        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []
            return {}, {}, {}, {}, {}

        if self.render_mode == "ansi" or "human" or "rgb_array":
            self.render()

        return observations, rewards, terminations, truncations, infos


def convert_to_calendar(observations):
    dates = {day+1: 0 for day in range(14)}
    for date in observations.values():
        dates[date] += 1
    return dates

# PARALLEL
env = parallel_env(render_mode='human')
observations, infos = env.reset()
if env.render_mode == 'human':
    pygame.init()
    win_width, win_height = 1000, 600
    win = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption("Surgery Scheduler")


def draw_boxes(win, calendar):
    # pygame.init()
    # win_width, win_height = 1000, 600
    # win = pygame.display.set_mode((win_width, win_height))
    # pygame.display.set_caption("Surgery Scheduler")

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
        win.blit(text, (day * box_width + box_width // 2 - text.get_width() // 2, win_height // 2 - text.get_height() - 5))
        text = font.render(str(n), 1, (10, 10, 10))
        win.blit(text, (day * box_width + box_width // 2 - text.get_width() // 2, win_height // 2))
    pygame.display.flip()


def game():
    draw_boxes(win, convert_to_calendar(observations))
    time.sleep(0.5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()

