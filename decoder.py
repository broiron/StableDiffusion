import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


'''
GroupNorm이 뭐냥?

norm layer: layer output의 분포를 이전 layer의 분포와 동일하게 맞춰주는 것
평균 : 0, 분산 : 1로 맞춰줌.
가까운 픽셀에서 나온 특성값들을 묶어서 normalize 해 주기 위해 group norm 사
'''

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)




class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):용
        super().__init__()
        # normalization과 conv로 이루어져 있음.
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # skip connection
        # 말 그대로 Input을 layer들을 건너뛰고 그대로 전달해 주는 layer

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()

        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x : (batch, in_channel, h, w)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x) # 사이즈 안 바꿈 -> kernel 3, pad 1
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)




