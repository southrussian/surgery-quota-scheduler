import torch
import torch.nn as nn
import math
from torch.nn import functional as F


def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.head_dim = n_embd // n_head

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Split into heads
        k = k.view(x.size(0), x.size(1), self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(x.size(0), x.size(1), self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.n_head, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        # Dynamically create the mask based on the actual sequence length
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0).to(att.device)

        att = att.masked_fill(torch.tensor(mask == 0), torch.tensor(float('-inf')))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_head * self.head_dim)
        y = self.resid_drop(self.proj(y))
        print(y)
        return y


class Block(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(32)
        self.ln2 = torch.nn.LayerNorm(32)
        self.attn = CausalSelfAttention(n_embd=32,  # Размер эмбеддингов
                                        n_head=2,  # Количество голов в multihead attn
                                        attn_pdrop=0.1,  # Вероятность dropout для attn
                                        resid_pdrop=0.1,  # Вероятность dropout для остатка
                                        block_size=128)  # Размер блока для маскирования

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 2 * 32),
            torch.nn.GELU(),
            torch.nn.Linear(2 * 32, 32),
            torch.nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, 1 + 1, 32))
        self.global_pos_emb = torch.nn.Parameter(torch.zeros(1, 400 + 1, 32))
        self.drop = torch.nn.Dropout(0.1)
        self.blocks = torch.nn.Sequential(*[Block() for _ in range(2)])
        self.ln_f = torch.nn.LayerNorm(32)
        self.head = torch.nn.Linear(32, 4, bias=False)
        self.apply(init_weights)
        self.state_encoder = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh())
        self.action_encoder = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh())

    def configure_optimizers(self):
        return torch.optim.Adam(params=[p for pn, p in self.named_parameters()])

    def forward(self, states, actions):
        # print(states)
        state_embeddings = self.state_encoder(states)
        # print(state_embeddings)
        action_embeddings = self.action_encoder(actions)
        token_embeddings = state_embeddings + action_embeddings

        seq_len = token_embeddings.size(0)
        position_embeddings = self.pos_emb[:, :seq_len, :]
        token_embeddings = token_embeddings + position_embeddings

        x = self.drop(token_embeddings)
        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
