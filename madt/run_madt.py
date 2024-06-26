import random
import numpy as np
import torch
from trainer import Trainer
from gpt_model import GPT
import pickle

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.set_num_threads(8)

global_obs_dim = 99
local_obs_dim = 79
action_dim = 10

block_size = 3

model = GPT()
device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)

trainer = Trainer(model)

with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

trainer.train(dataset)
