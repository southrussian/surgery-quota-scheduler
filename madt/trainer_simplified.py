import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy


class Trainer:

    def __init__(self, model):
        self.model = model
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.optimizer = self.raw_model.configure_optimizers()
        # self.writer = SummaryWriter('./logs')

    def train(self, dataset):
        model = self.raw_model
        target_model = copy.deepcopy(model)
        target_model.train(False)

        global_step = 0

        def run_epoch():
            nonlocal global_step
            model.train(True)
            loader = DataLoader(dataset)
            loss_info = 0
            # correct_predictions = 0
            # total_predictions = 0
            for idx, batch in enumerate(loader):
                states = torch.stack([item[0] for item in batch]).unsqueeze(dim=-1).type(torch.float32)
                actions = torch.stack([item[1] for item in batch]).unsqueeze(dim=-1).type(torch.float32)
                rewards = torch.stack([item[1] for item in batch]).unsqueeze(dim=-1).type(torch.float32)
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                with torch.set_grad_enabled(True):
                    logits = model(states, actions)
                    loss = torch.nn.functional.mse_loss(logits, rewards)
                    loss_info = loss.item()
                    # criteria = torch.nn.CrossEntropyLoss()
                    # loss = criteria(logits.view(-1, logits.size(-1)), rewards.view(-1))
                    # loss_info = loss.item()
                    # probs = torch.nn.functional.softmax(logits, dim=-1)
                    # confidence, predicted_labels = torch.max(probs, dim=-1)
                    # correct_predictions += (predicted_labels == action).sum().item()
                    # total_predictions += action.size(0)

                    # entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=-1).mean()
                    # ratio = correct_predictions / total_predictions
                    # confidence = confidence.mean().item()
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.optimizer.step()

                # self.writer.add_scalar('Loss/train', loss_info, global_step)
                # self.writer.add_scalar('Entropy/train', entropy.item(), global_step)
                # self.writer.add_scalar('Ratio/train', ratio, global_step)
                # self.writer.add_scalar('Confidence/train', confidence, global_step)

                global_step += 1

            return loss_info, entropy, ratio, confidence

        loss, entropy, ratio, confidence = 0., 0., 0., 0.
        for _ in range(10):
            loss, entropy, ratio, confidence = run_epoch()

        # self.writer.close()
        return loss, entropy, ratio, confidence
