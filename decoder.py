import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


'''
GroupNorm이 뭐냐?

norm layer: layer output의 분포를 layer input분포와 동일하게 맞춰주는 것
평균 : 0, 분산 : 1로 맞춰줌.
가까운 픽셀에서 나온 특성값들을 묶어서 normalize 해 주기 위해 group norm 사
'''

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, Features(channels), Height, Width)

        residue = x

        n, c, h, w = x.shape

        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        x = x.view(n, c, h*w) # 픽셀 시퀀스를 채널 * 배치 만큼 가지게 됨

        # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        x = x.transpose(-1, -2)
        
        # height * width 크기의 하나의 feature embedding 끼리의 attention 연산 수행한다고 볼 수 있음.
        # (batch_size, height * width, channels) -> (batch_size, height * width, channels)
        x = self.attention(x)

        # attention 연산 이후 원상복구

        # (batch_size, height * width, channels) -> (batch_size, channels, height * width) 
        x = x.transpose(-1, -2)
        
        # (batch_size, channels, height * width) -> (batch_size, channels, height, width)
        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

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

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # 이미지의 해상도를 높이는 과정
            # (Batch_size, 512, 512, Height / 8, Width / 8) -> (Batch_size, 512, 512, Height / 4, Width / 4) 
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, 512, Height / 4, Width / 4) -> (Batch_size, 512, 512, Height / 2, Width / 2) 
            nn.Upsample(scale_factor=2),
            # 각 channel의 해상도는 올리고, channel수는 감소시키는 과정
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, 256, 256, Height / 2, Width / 2) -> (Batch_size, 256, 256, Height, Width) 
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128), # 128개의 channel을 32개씩 묶어서 4개의 group을 만듬

            nn.SiLU(),
            # 원래 이미지로 복기
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, Height / 8, Width / 8)

        x /= 0.18215 

        for module in self:
            x = module(x)

        # (Batch_size, 3, Height, Width)
        return x
