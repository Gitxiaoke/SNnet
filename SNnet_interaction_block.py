import torch
from torch import nn


"""
 Interaction的输入是F_RA_n,F_RA_s 维度形状是 F_RA：[B, T, F', C]
"""


class Conv_Block(nn.Module):
    def __init__(self, channel=64, kernel=(5, 7), stride=(1, 1), padding=(2, 3)):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.lay(x)


class Interaction_Block(nn.Module):
    def __init__(self, channel=64):
        super(Interaction_Block, self).__init__()
        self.Conv_n2s = Conv_Block(channel*2)  # TODO 这里千万要注意，因为这里是两条路，一个n2s，一个s2n，所以是连个独立的卷积层
        self.Conv_s2n = Conv_Block(channel*2)  # TODO 1*1卷积核是不会学习到任何特征的，它只能起调整通道数的作用

    def forward(self, F_RA_n, F_RA_s):  # F_RA：[B, T, F', C]
        F_cat = torch.cat((F_RA_n, F_RA_s), dim=-1)  # F_cat: [B, T, F', 2*C]

        F_cat = F_cat.permute(0, 3, 1, 2)  # 进入卷积层之前一定要将通道维度 C 调整到第二维度  [B, 2*C, T, F']
        Mask_n = self.Conv_n2s(F_cat)    # [B, C, T, F']
        Mask_s = self.Conv_s2n(F_cat)    # [B, C, T, F']
        Mask_n = Mask_n.permute(0, 2, 3, 1)  # 退出卷积层后也要调整通道维度C所在的位置  [B, T, F', C]
        Mask_s = Mask_s.permute(0, 2, 3, 1)  # 退出卷积层后也要调整通道维度C所在的位置  [B, T, F', C]

        H_n2s = F_RA_n * Mask_n   # [B, T, F', C]
        H_s2n = F_RA_s * Mask_s   # [B, T, F', C]

        F_RA_S = F_RA_s + H_n2s   # [B, T, F', C]
        F_RA_N = F_RA_n + H_s2n   # [B, T, F', C]

        return F_RA_S, F_RA_N


if __name__ == '__main__':
        s = torch.randn(10, 200, 256, 64)
        n = torch.randn(10, 200, 256, 64)
        model = Interaction_Block(64)
        S, N = model(n, s)
        print(S.shape, N.shape)
