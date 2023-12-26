import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        # n_vocab: 단어 집합의 개수
        # n_embd: 임베딩 할 벡터의 차원
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # 학습가능한 parameter
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        # scalar 값들을 더해줘서 위치적인 가중치 값을 가지도록 함.
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)

        # self attention
        self.attention = SelfAttention(n_head, n_embd)

        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4* n_embd, n_embd)

    def forward(self, x):
        residue = x

        ### Self attention ###
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        ### 위 과정이 self attention ###

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x*torch.sigmoid(1.702 * x) # GELU activation function

        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, token: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output















