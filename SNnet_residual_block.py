# 根据论文介绍，RA_block中的residual_block就是两个卷积层，非常简单

import torch
from torch import nn

"""
千万要注意，这里是residual_block残差块，并不是普通的卷积神经网络
"""

"""
根据论文的描述, residual block有两个卷积层，卷积核的大小都是（5,7），stride都是（1,1）
而通道数则是和输入数据的通道数保持一致
"""


class Two_Res_Block(nn.Module):
    def __init__(self, channel=64, kernel=(5, 7), stride=(1, 1), padding=(2, 3)):
        super(Two_Res_Block, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel, stride, padding, bias=False),  # W=1 + (W-K+2*P)/S
                                nn.BatchNorm2d(channel),
                                )
        self.r = nn.ReLU(inplace=True)
        self.c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel, stride, padding, bias=False),
                                nn.BatchNorm2d(channel)
                                )

    def forward(self, input):
        # TODO 第一个RA的residual模块接收的是encoder的输出[B, C=64, T, F/4]
        # TODO 然而，RA模块的输出F_RA确实[B, T, F', C]，这是不能直接输入进下一个RA模块的
        # TODO 因此正确的做法是将后面3个RA模块的输入F_RA时将其格式[B, T, F', C]调整为[B, C, T, F']
        """ 第一个residual """
        out = self.c1(input)
        out = self.r(out)
        out = self.c2(out)
        res1_out = self.r(input + out)
        """ 第二个residual  注意这是两个residual的串联 """
        out = self.c1(res1_out)
        out = self.r(out)
        out = self.c2(out)
        res2_out = self.r(res1_out + out)  # [B, C=64, T, F']
        F_res = res2_out.permute(0, 2, 3, 1)  # [B, T, F', C=64]
        return F_res


if __name__ == '__main__':
    x = torch.randn(10, 64, 256, 256)  # [B, C, F, T]
    x1 = x
    model = Two_Res_Block(64)  # 通道数为10，但是实际上应该是64，因为encoder最后的输出的通道数就是64
    y = model(x)
    if x is x1:
        print('not changed')
    else:
        print('change end')
    print(y.shape)

