import torch
from torch import nn
from torch.nn import funtional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):

        '''
        왜 이렇게 만들었냥
        이런 식의 모델 설계는 다른 코드와 호완성이 좋음
        기존의 autoencoder의 코드 구조와 유사하기 때문.
        픽셀의 개수는 줄어들지만, 각 픽셀을 표현하는 채널들의 수가 더 많아지는 구조
        '''

        super().__init__(
            # block 들 작성 -> sequential 한 block
            # (batch_size, Channel, height, width) 로 시작
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch_size, 128, height, width) 로 변환 : kernel : 3, padding : 1

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128), # conv와 norm의 조합
            # Residual block은 크기를 바꾸지 않음

            # height, width 변경
            # (batch_size, 128, height, width) -> (batch_size, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)

            # (batch_size, 128, h / 2, w / 2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, h / 2, w / 2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 4, w / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(512, 512),

            # image 해상도를 더 줄임.
            # (batch_size, 512, h / 8, w / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),

            # pixel 간의 self attention 진행
            VAE_AttentionBlock(512),

            # (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            # Activation function ReLU와 유사함
            # (batch_size, 512, h / 8, w / 8)
            nn.SiLU(),

            # pixel수는 그대로, channel은 줄임 -> bottleneck
            # (batch_size, 8, h / 8, w / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, 8, h / 8, w / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (batch size, channel, height, width)
        # noise: (batch size, output_channel, height/8, width/8) -> encoder의 output과 같은 크기
        for module in self:
            if getattr(module, 'stride', None) == (2, 2): # layer의 stride가 2라면...
                # 즉 image resolution을 줄이는 layer라면
                # F.pad(패딩 할 이미지, (Padding_left, Padding_right, Padding_top, Padding_bottom))
                x = F.pad(x, (0, 1, 0, 1)) # right, bottom만 1 만큼 패딩함.
                # stride가 2일 때, 오른쪽이랑 아래쪽 하나씩 짤려서 conv 연산 못 하기 때문에.
            x = module(x)

        # (batch_size, 8, h / 8, w / 8) -> 2개의 tensor로 나눔
        # (batch_size, 4. height/8, width/8) * 2
        # 이게 encoder를 통과한 latent space의 distribution을 의미한다는데...ㅅㅂ?
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20) # -30, 20 의 범위로 조정해 주세용

        varience = log_variance.exp()

        stdev = varience.sqrt()

        # 뭔지는 모르겠지만 지금 gaussian 분포를 따르는 latent space의 평균과 분산을 획득했다.
        # 여기서 latent space의 data를 어떻게 샘플링 할거냐?

        # 공식이 뭔지는 모르겠지만.
        # Z = N(0, 1) -> N(mean, variance = x?) WTF?
        # 이게 sampling 공식이랭 : X = mean + stdev * z
        # 표준 정규분포를 따르는 latent noise에서

        x = mean + stdev * noise








