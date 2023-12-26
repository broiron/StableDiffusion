import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True) -> None:
        super().__init__()

        # 3개의 가중치 matrix : Query, Key, Value
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        # d_embedding이 d_head로 나누어진다..?
        # multi head가 기존 embedding 차원을 늘리는 게 아니라, head개수 만큼 나눠서 병렬적으로 처리하는 것.
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask=False):
        # causal_mask: 목표하는 문장의 일부를 가려서 인위적으로 연속성을 학습하도록 함.
        # x: (Batch_size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 3개의 덩어리로 나눔
        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, dim*3) -> 3 개의 tensor로 나눔 (Batch_size, Seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)


        # (Batch_size, Seq_len, dim) -> (Batch_size, Seq_len, H, Dim / H) -> (Batch_size, H, Seq_len, Dim / h)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (Batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1) # triu: tensor를 대각행렬로 만들어줌
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, Seq_Len, Seq_Len) @ (Batch_size, H, Seq_Len, Dim/H) -> (Batch_size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_size, H, Seq_Len, Dim / H) -> (Batch_size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape) # 처음 shape으로 다시 돌림

        output = self.out_proj(output)

        # (batch_size, Seq_len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # query, key, value를 각각 정의
        # query는 그대로 동일
        # key와 value에 cross할 embedding 입력
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): (batch_size, seq_len_Q, dim_Q)
        # y (context): (batch_size, seq_len_KV, dim_KV) = (batch_Size, 77, 768) -> clip embedding 출력
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # multihead attention을 위해 q의 차원을 dim_heads * n_heads = dim_q 되도록 나누어줌
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # latent를 query에, key랑 value에는 context가 들어감.
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous() # contiguous: 강제로 메모리를 할당해서 output 을 위한 공간을 따로 만들어 줌.
        output = output.view(input_shape)

        output = self.out_proj(output)
        return output