import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy


class Trainer:

    def __init__(self, model, log_dir='./logs'):
        self.model = model
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.optimizer = self.raw_model.configure_optimizers()
        self.writer = SummaryWriter(log_dir)

    def train(self, dataset):
        model = self.raw_model
        target_model = copy.deepcopy(model)
        target_model.train(False)

        global_step = 0

        def run_epoch():
            nonlocal global_step
            model.train(True)
            loader = DataLoader(dataset, shuffle=True)
            loss_info = 0
            correct_predictions = 0
            total_predictions = 0
            for _, (_, observation, action, _, _) in enumerate(loader):
                # print(action)
                observation = observation.to(self.device)
                action = action.to(self.device)
                with torch.set_grad_enabled(True):
                    logits = model(observation, action)
                    criteria = torch.nn.CrossEntropyLoss()
                    loss = criteria(logits.view(-1, logits.size(-1)), action.view(-1))

                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    confidence, predicted_labels = torch.max(probs, dim=-1)
                    correct_predictions += (predicted_labels == action).sum().item()
                    total_predictions += action.size(0)

                    entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=-1).mean()
                    ratio = correct_predictions / total_predictions
                    confidence = confidence.mean().item()
                    loss_info = loss.item()

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.optimizer.step()

                self.writer.add_scalar('Loss/train', loss_info, global_step)
                self.writer.add_scalar('Entropy/train', entropy.item(), global_step)
                self.writer.add_scalar('Ratio/train', ratio, global_step)
                self.writer.add_scalar('Confidence/train', confidence, global_step)

                global_step += 1

            return loss_info, entropy.item(), ratio, confidence

        actor_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0.
        for _ in range(10):
            actor_loss_ret, entropy, ratio, confidence = run_epoch()

        self.writer.close()
        return actor_loss_ret, entropy, ratio, confidence

# import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter
# import copy
#
#
# class Trainer:
#
#     def __init__(self, model, log_dir='./logs'):
#         self.model = model
#         self.device = 'cpu'
#         if torch.cuda.is_available():
#             self.device = torch.cuda.current_device()
#         self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
#         self.optimizer = self.raw_model.configure_optimizers()
#         self.writer = SummaryWriter(log_dir)  # Initialize the SummaryWriter with a log directory
#
#     def train(self, dataset):
#         model = self.raw_model
#         target_model = copy.deepcopy(model)
#         target_model.train(False)
#
#         global_step = 0  # Initialize a step counter
#
#         def run_epoch():
#             nonlocal global_step  # To modify the global_step within this scope
#             model.train(True)
#             loader = DataLoader(dataset, shuffle=True)
#             loss_info = 0
#             for _, (_, observation, action, _, _) in enumerate(loader):
#                 observation = observation.to(self.device)
#                 action = action.to(self.device)
#                 with torch.set_grad_enabled(True):
#                     logits = model(observation, action)
#                     criteria = torch.nn.CrossEntropyLoss()
#                     loss = criteria(logits.reshape(-1, logits.size(-1)), action.reshape(-1)[0].unsqueeze(0))
#
#                     # probs = torch.nn.functional.softmax(logits, dim=-1)
#                     # log_probs = torch.log(probs)
#                     # target_log_probs = torch.gather(log_probs, dim=-1, index=action.unsqueeze(-1))
#                     # loss = -target_log_probs.mean()
#
#                     entropy_info = 0.
#                     ratio_info = 0.
#                     confidence_info = 0.
#                     loss = loss.mean()
#                     loss_info = loss.item()
#
#                 model.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#                 self.optimizer.step()
#
#                 # Logging the metrics to TensorBoard
#                 self.writer.add_scalar('Loss/train', loss_info, global_step)
#                 self.writer.add_scalar('Entropy/train', entropy_info, global_step)
#                 self.writer.add_scalar('Ratio/train', ratio_info, global_step)
#                 self.writer.add_scalar('Confidence/train', confidence_info, global_step)
#
#                 global_step += 1  # Increment the global step counter after logging
#
#             return loss_info, entropy_info, ratio_info, confidence_info
#
#         actor_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0.
#         for _ in range(10):
#             actor_loss_ret, entropy, ratio, confidence = run_epoch()
#
#         self.writer.close()  # Close the writer after training is done
#         return actor_loss_ret, entropy, ratio, confidence
