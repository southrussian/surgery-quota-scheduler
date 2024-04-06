import random
import numpy as np
import torch
import os
from trainer_simplified import Trainer
from gpt_model_simplified import GPT
import pickle

# toy_example_tensor = [
#     (torch.tensor([0.0]), torch.tensor([1]), torch.tensor([1]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.8]), torch.tensor([1]), torch.tensor([0]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.0]), torch.tensor([1]), torch.tensor([2]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([-0.8]), torch.tensor([1]), torch.tensor([1]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.8]), torch.tensor([1]), torch.tensor([0]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.0]), torch.tensor([1]), torch.tensor([1]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.8]), torch.tensor([1]), torch.tensor([3]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.0]), torch.tensor([1]), torch.tensor([1]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.8]), torch.tensor([2]), torch.tensor([2]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.0]), torch.tensor([1]), torch.tensor([3]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.8]), torch.tensor([2]), torch.tensor([0]), torch.tensor(False), torch.tensor(False)),
#     (torch.tensor([0.0]), torch.tensor([3]), torch.tensor([3]), torch.tensor(False), torch.tensor(False))
# ]

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

offline_trainer = Trainer(model)

with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
    print(dataset)

target_rtgs = 20.
print("offline target_rtgs: ", target_rtgs)
for i in range(1):
    offline_actor_loss, _, __, ___ = offline_trainer.train(dataset)
    if True:
        actor_path = '../../offline_model/' + 'easy_trans' + '/actor'
        if not os.path.exists(actor_path):
            os.makedirs(actor_path)
