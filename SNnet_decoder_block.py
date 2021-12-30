# -*- coding:utf-8 -*-
"""
作者：xiaoke
日期：2021年12月10日
"""

import torch
from torch import nn


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.lay(x)


class Gated_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding):
        """
        根据论文描述， gated_block 的三个deconv的卷积核都是（3,5）
        """
        super(Gated_Block, self).__init__()
        # TODO: 这里的反卷积可以用 转置卷积 或者 插值法 （参考图像处理中的 U-NET）
        # TODO: 这里的反卷积的目的将是频率维度F还原: F/4 = 65 —> F/2 = 129 -> F = 257
        self.Deconv_lay = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 5), stride=stride, padding=padding)
        self.Conv_lay = Conv_Block(out_channel * 2, out_channel, kernel=(1, 1), stride=(1, 1), padding=0)

    def forward(self, M_in, R_in):
        """
        """
        out_Deconv = self.Deconv_lay(M_in)                     # [B, C=32, T, F/2]
        out1_cat = torch.cat((out_Deconv, R_in), dim=1)        # [B, C=32+32, T, F/2]
        out1_conv = self.Conv_lay(out1_cat)                    # [B, C=32, T, F/2]
        mask_out = R_in * out1_conv                            # [B, C=32, T, F/2]
        out2_cat = torch.cat((mask_out, out_Deconv), dim=1)    # [B, C=32+32, T, F/2]
        out2_conv = self.Conv_lay(out2_cat)                    # [B, C=32, T, F/2]
        return out2_conv + out_Deconv                          # [B, C=32, T, F/2]


class Decoder_Block(nn.Module):
    def __init__(self):
        """
        根据论文描述， gated_block 的三个deconv的卷积核都是（3,5）
        前两个gated_block的deconv的stride是（1, 2），可以推出相应padding是（1, 2）
        最后一个gated_block的deconv的stride是（1, 1），可以推出相应padding是（1, 2）
        """
        super(Decoder_Block, self).__init__()
        self.gated_1 = Gated_Block(in_channel=64, out_channel=32, stride=(1, 2), padding=(1, 2))
        self.gated_2 = Gated_Block(in_channel=32, out_channel=16, stride=(1, 2), padding=(1, 2))
        self.gated_3 = Gated_Block(in_channel=16, out_channel=1, stride=(1, 1), padding=(1, 2))

    def forward(self, f_ra, r2, r1, f_in):
        out_1 = self.gated_1(f_ra, r2)
        out_2 = self.gated_2(out_1, r1)
        out_3 = self.gated_3(out_2, f_in)
        return out_1, out_2, out_3


if __name__ == '__main__':
    f_in = torch.randn(1, 1, 100, 257)
    r_1 = torch.randn(1, 16, 100, 257)
    r_2 = torch.randn(1, 32, 100, 129)
    f_ra = torch.randn(1, 64, 100, 65)
    model = Decoder_Block()
    out1, out2, out3 = model(f_ra, r_2, r_1, f_in)
    print(out1.shape, out2.shape, out3.shape)

