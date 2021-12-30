# -*- coding:utf-8 -*-
"""
作者：xiaoke
日期：2021年12月10日
"""
import torch
from torch import nn


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding=(0, 0)):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.lay(x)


class Encoder_Block(nn.Module):
    def __init__(self):
        """
        根据论文的描述，encoder的三层卷积的卷积核大小都是（3，5），而第一层的stride是（1,1），二三层的stride是（1,2）
        并且每一层conv的输入在时间维度上并没有变换，但是encoder的最后一层的输入的频率维度变为了原本的四分子一 F/4
        """
        super(Encoder_Block, self).__init__()
        self.conv1 = Conv_Block(in_channel=1, out_channel=16, kernel=(3, 5), stride=(1, 1), padding=(1, 2))
        self.conv2 = Conv_Block(in_channel=16, out_channel=32, kernel=(3, 5), stride=(1, 2), padding=(1, 2))
        self.conv3 = Conv_Block(in_channel=32, out_channel=64, kernel=(3, 5), stride=(1, 2), padding=(1, 2))

    def forward(self, F_in):   # 最初的输入一定要是【B, C, F, T】
        """
        需要注意的是，STFT的输出是[B, C, F, T], 所以在输入进encoder之前应该将其装换为[B, C=2, T, F]
        conv1的输出为[B, C=16, T, F],
        conv2的输出为[B, C=32, T, F/2],
        conv3的输出为[B, C=64, T, F/4].
        """
        input = F_in.permute(0, 1, 3, 2)
        out1_encoder = self.conv1(input)          # r1 = [B, C=16, T ,F]
        out2_encoder = self.conv2(out1_encoder)   # r2 = [B, C=32, T, F/2]
        out3_encoder = self.conv3(out2_encoder)   # r3 = [B, C=64, T, F/4]
        return out1_encoder, out2_encoder, out3_encoder


if __name__ == '__main__':
    x = torch.randn(1, 1, 257, 100)  # [B, C=1, F=257， T=100]
    encoder = Encoder_Block()
    R1, R2, R3 = encoder(x)   # R1:[1, 16, 10, 257]  R2:[1, 32, 10, 129]  R3:[1, 64, 10, 65]
    print(R1.shape, R2.shape, R3.shape)
