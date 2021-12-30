# -*- coding:utf-8 -*-
"""
作者：xiaoke
日期：2021年12月16日
"""
import torch
import torch.nn as nn
from model_SNnet.SNnet_residual_block import Two_Res_Block
from model_SNnet.SNnet_attention_block import Attention_block


class Ra_Block(nn.Module):
    def __init__(self):
        super(Ra_Block, self).__init__()
        self.residual_lay = Two_Res_Block()
        self.attention_lay = Attention_block()

    def forward(self, x):  # [B, C=64, T, F']
        F_res = self.residual_lay(x)  # [B, T, F', C=64]
        _, F_ra = self.attention_lay(F_res)  # [B, T, F', C=64]:F_ra
        return F_ra


if __name__ == '__main__':
    x = torch.randn(1, 64, 100, 256//4)
    model = Ra_Block()
    y = model(x)
    print(y.shape)
