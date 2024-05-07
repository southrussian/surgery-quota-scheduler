# pip install torch torchvision torchaudio gym

import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical

# Hyperparameters
gamma = 0.99
clip_param = 0.2
max_grad_norm = 0.5
update_timestep = 100
lr = 0.002
betas = (0.9, 0.999)


# Transformer Model for Critic
class TransformerCritic(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, ff_dim, num_layers):
        super(TransformerCritic, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        value = self.value_head(x[:, -1, :])  # Assuming x is (batch, sequence, feature)
        return value


# PPO Policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.affine = nn.Linear(state_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.affine(x))
        action_probs = self.softmax(self.action_head(x))
        return action_probs


# Main PPO class
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = TransformerCritic(input_dim=state_dim, embed_dim=128, n_heads=4, ff_dim=512, num_layers=2)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, betas=betas)

    def select_action(self, state, memory):
        try:
            state = torch.from_numpy(state[0]).float().unsqueeze(0)
        except TypeError:
            state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def update(self, memory):
        # Calculate discounted rewards and advantages
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards)
        old_states = torch.stack(memory.states).squeeze(1)
        old_actions = torch.stack(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list of states to the right shape for transformer
        states_for_critic = old_states.unsqueeze(1)  # Dummy dimension for sequence

        # Evaluating old actions and values
        action_probs = self.actor(old_states)
        dist = Categorical(action_probs)
        entropy_loss = dist.entropy().mean()
        state_values = self.critic(states_for_critic).squeeze(1)

        # Calculate new log probs and advantages
        new_logprobs = dist.log_prob(old_actions)
        advantages = rewards - state_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(new_logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * (state_values - rewards).pow(2) - 0.01 * entropy_loss

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.optimizer.step()

        # Clear memory
        del memory.actions[:]
        del memory.states[:]
        del memory.logprobs[:]
        del memory.rewards[:]


# Memory class for storing trajectories
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []


def evaluate_model(model, episodes=5):
    env = gym.make("CartPole-v1", render_mode="human")
    model.actor.eval()  # Set the actor model to evaluation mode
    model.critic.eval()  # Set the critic model to evaluation mode
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = model.select_action(state, None)  # Select action without storing to memory
            state, reward, terminated, done, _ = env.step(action)
            env.render()  # Render the environment to visualize the agent
            total_reward += reward
            if terminated or done:
                break
        print(f"Episode finished with total reward: {total_reward}")


# Main function to train and interact with environment
def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    timestep = 0

    not_done = True
    while not_done:
        state = env.reset()
        for _ in range(10000):
            timestep += 1
            action = ppo.select_action(state, memory)
            state, reward, terminated, done, _ = env.step(action)
            memory.rewards.append(torch.tensor([reward]))
            if timestep % update_timestep == 0:
                print('updated', timestep)
                ppo.update(memory)
            if terminated:
                break
            if timestep >= 10000:
                done = True
            if done:
                not_done = False
                break

    # After training, evaluate the model
    evaluate_model(ppo, episodes=10)
    env.close()  # Close the environment


if __name__ == '__main__':
    main()
