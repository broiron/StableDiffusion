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
        self.d_head = d_embed // n_heads

    def foward(self, x:torch.Tensor, causal_mask=False):
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

        weight = F.softmax(weight, dim = -1)

        # (Batch_size, H, Seq_Len, Seq_Len) @ (Batch_size, H, Seq_Len, Dim/H) -> (Batch_size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_size, H, Seq_Len, Dim / H) -> (Batch_size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape) # 처음 shape으로 다시 돌림

        output = self.out_proj(output)

        # (batch_size, Seq_len, Dim)
        return output

        


