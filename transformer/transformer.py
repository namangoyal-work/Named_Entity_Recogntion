import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SingleHeadAttention(nn.Module):

    def __init__(self, embed_dim=512, att_dim=64, p=0.2):
        super().__init__()
        self.W_k = nn.Linear(embed_dim, att_dim)
        self.W_q = nn.Linear(embed_dim, att_dim)
        self.W_v = nn.Linear(embed_dim, att_dim)
        self.d   = np.sqrt(embed_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, X):
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)
        kv = torch.bmm(Q,K.transpose(1,2))
        f = torch.bmm(F.softmax(kv/self.d, dim=-1), V)
        return self.dropout(f)

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads=8, embed_dim=512, att_dim=64):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(embed_dim=embed_dim, att_dim=att_dim) for _ in range(n_heads)])
        self.linear = nn.Linear(att_dim*n_heads, embed_dim)

    def forward(self, X):
        att_concat = torch.cat([att(X) for att in self.heads], dim=2)
        return self.linear(att_concat)

class Sublayer(nn.Module):

    def __init__(self, module, layer_size=512, p=0.2):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p)
        self.norm = nn.LayerNorm([layer_size])

    def forward(self, x):
        return self.norm(x+self.dropout(self.module(x)))

class FeedForwardNN(nn.Module):

    def __init__(self, hidden_dim=2048, embed_dim=512, p=0.2):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.d  = nn.Dropout(p)
        self.W2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        return self.W2(self.d(self.W1(x).relu()))

class EncoderLayer(nn.Module):

    def __init__(self, n_heads=8, hidden_dim=2048, embed_dim=512, att_dim=64):
        super().__init__()
        # TODO pass size
        self.attn = Sublayer(MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, att_dim=att_dim))
        self.ff = Sublayer(FeedForwardNN(hidden_dim=hidden_dim, embed_dim=embed_dim))

    def forward(self, x):
        return self.ff(self.attn(x))

class Encoder(nn.Module):

    def __init__(self, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PositionalEmbedding(nn.Module):

    def __init__(self, dim=512, max_len=5000, p=0.2):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * -(np.log(10000.0) / dim)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        # self.register_buffer("pe", self.pe)

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x+self.pe[:,:x.shape[1]].requires_grad_(False))

class Tokenizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab = {chr(i):i for i in range(256)}
        self.rules = {}

    def fit(self, vocab):
        trie = Trie()
        for word in vocab:
            trie.add_str(word)
        # candidates = trie.get_pairs()
        
