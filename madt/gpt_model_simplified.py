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
        self.register_buffer("mask", torch.tril(torch.ones(block_size + 1, block_size + 1))
                             .view(1, 1, block_size + 1, block_size + 1))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :32, :32] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v.transpose(-2, -1))
        y = y.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_head * (x.size(-1) // self.n_head))
        y = self.resid_drop(self.proj(y))

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
        self.head = torch.nn.Linear(32, 13, bias=False)
        self.apply(init_weights)
        self.state_encoder = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh())
        self.action_encoder = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh())
        # self.action_embeddings = torch.nn.Sequential(torch.nn.Embedding(2, 32), torch.nn.Tanh())
        # torch.nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def configure_optimizers(self):
        return torch.optim.Adam(params=[p for pn, p in self.named_parameters()])

    def forward(self, states, actions):
        state_embeddings = self.state_encoder(states.unsqueeze(dim=1).type(torch.float32).contiguous())
        action_embeddings = self.action_encoder(actions.unsqueeze(dim=1).type(torch.float32).contiguous())
        token_embeddings = state_embeddings + action_embeddings
        seq_len, embed_dim = token_embeddings.shape[0], token_embeddings.shape[1]
        position_embedding = torch.nn.Embedding(seq_len, embed_dim)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        position_embeddings = position_embedding(position_ids)
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

        # print("Input states shape:", states.shape)
        # state_embeddings = self.state_encoder(states.reshape(-1, 3).type(torch.float32).contiguous())
        # state_embeddings = self.state_encoder(states.unsqueeze(dim=1).type(torch.float32).contiguous())
        # seq_len, embed_dim = state_embeddings.shape[0], state_embeddings.shape[1]
        # position_embedding = torch.nn.Embedding(seq_len, embed_dim)
        # position_ids = torch.arange(seq_len, dtype=torch.long)
        # position_embeddings = position_embedding(position_ids)
        # x = self.drop(state_embeddings + position_embeddings)
        # x = self.blocks(x)
        # x = self.ln_f(x)
        # return self.head(x)
