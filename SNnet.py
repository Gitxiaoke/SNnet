import torch
import torch.nn as nn
from thop import profile

from model_SNnet.SNnet_encoder_block import Encoder_Block
from model_SNnet.SNnet_ra_block import Ra_Block
from model_SNnet.SNnet_interaction_block import Interaction_Block
from model_SNnet.SNnet_decoder_block import Decoder_Block


class SN_net(nn.Module):
    def __init__(self):
        super(SN_net, self).__init__()
        """ encoder 信号支路和噪声支路 """
        self.encoder_s = Encoder_Block()
        self.encoder_n = Encoder_Block()
        """ attention特征提取模块和交互模块 """
        self.ra_s_1 = Ra_Block()
        self.ra_n_1 = Ra_Block()
        self.interaction_1 = Interaction_Block()
        self.ra_s_2 = Ra_Block()
        self.ra_n_2 = Ra_Block()
        self.interaction_2 = Interaction_Block()
        self.ra_s_3 = Ra_Block()
        self.ra_n_3 = Ra_Block()
        self.interaction_3 = Interaction_Block()
        self.ra_s_4 = Ra_Block()
        self.ra_n_4 = Ra_Block()
        self.interaction_4 = Interaction_Block()
        """ decoder 信号支路和噪声支路 """
        self.decoder_s = Decoder_Block()
        self.decoder_n = Decoder_Block()
        """ 最后一个卷积层输出 """

    def forward(self, x):
        """ SN-net的前向传播过程 """
        """ encoder """
        R_s1, R_s2, R_s3 = self.encoder_s(x)  # R1:[B, C=16, T , F]  R2:[B, C=32, T , F/2]  R3:[B, C=64, T, F/4]
        R_n1, R_n2, R_n3 = self.encoder_n(x)
        """ RA & interaction"""
        F_ra_s_1 = self.ra_s_1(R_s3)
        F_ra_n_1 = self.ra_n_1(R_n3)
        F_ra_s_1, F_ra_n_1 = self.interaction_1(F_ra_n_1, F_ra_s_1)
        F_ra_s_2 = self.ra_s_2(F_ra_s_1.permute(0, 3, 1, 2))
        F_ra_n_2 = self.ra_n_2(F_ra_n_1.permute(0, 3, 1, 2))
        F_ra_s_2, F_ra_n_2 = self.interaction_2(F_ra_n_2, F_ra_s_2)
        F_ra_s_3 = self.ra_s_3(F_ra_s_2.permute(0, 3, 1, 2))
        F_ra_n_3 = self.ra_n_3(F_ra_n_2.permute(0, 3, 1, 2))
        F_ra_s_3, F_ra_n_3 = self.interaction_3(F_ra_n_3, F_ra_s_3)
        F_ra_s_4 = self.ra_s_4(F_ra_s_3.permute(0, 3, 1, 2))
        F_ra_n_4 = self.ra_n_4(F_ra_n_3.permute(0, 3, 1, 2))
        F_ra_s_4, F_ra_n_4 = self.interaction_4(F_ra_n_4, F_ra_s_4)
        """ decoder """
        out_s1, out_s2, out_s3 = self.decoder_s(F_ra_s_4.permute(0, 3, 1, 2), R_s2, R_s1, x.permute(0, 1, 3, 2))  # f_ra, r2, r1, f_in
        out_n1, out_n2, out_n3 = self.decoder_n(F_ra_n_4.permute(0, 3, 1, 2), R_n2, R_n1, x.permute(0, 1, 3, 2))
        """ Conv2d out """

        return out_s3, out_n3


if __name__ == '__main__':
    """ 最开始的输入通道C=2 只是因为原论文中采用输入的是 幅度谱 和 相位谱 """
    # TODO 这里我的实验只采用了幅度谱
    # TODO 最初的输入一定要是【B, C, F, T】
    x = torch.randn(1, 1, 257, 7)
    model = SN_net()
    # out_s, out_n = model(x)
    # print(out_s.shape, out_n.shape)
    macs, params = profile(model, inputs=(x,))
    print('macs:', macs / 1000 / 1000 / 1000, 'params:', params / 1000 / 1000)
