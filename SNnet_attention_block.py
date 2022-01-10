# -*- coding:utf-8 -*-
"""
作者：xiaoke
日期：2021年12月09日
"""
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.lay(x)


class Attention_block(nn.Module):
    def __init__(self, channel=64):
        """ 根据论文中的描述，文中的attention使用的是scaled dot-product，也就是内积"""
        super(Attention_block, self).__init__()  # conv_DownChannel只是为了输入时降通道数量，conv_UpChannel则是为了输出时还原通道数量
        self.conv_DownChannel = Conv_Block(channel, (channel // 2), kernel=(1, 1), stride=(1, 1), padding=0)
        self.conv_UpChannel = Conv_Block((channel // 2), channel, kernel=(1, 1), stride=(1, 1), padding=0)
        self.conv_out = Conv_Block(channel * 3, channel, kernel=(1, 1), stride=(1, 1), padding=0)
        # self.conv_out = Conv_Block(channel * 2, channel, kernel=(1, 1), stride=(1, 1), padding=0)

    def forward(self, F_res):       # F_res: [B, T, F', C=64]  输入语谱F_res维度形状
        num_C = F_res.shape[-1]
        len_F = F_res.shape[-2]
        len_T = F_res.shape[-3]

        conv_Fres = self.conv_DownChannel(F_res.permute(0, 3, 1, 2))    # [B, C/2=32, T, F'] 根据论文中的描述，首先就是对输入的通道降为原来的一半
        conv_Fres_t = conv_Fres.permute(0, 2, 3, 1)                     # [B, T, F', C/2]
        conv_Fres_f = conv_Fres.permute(0, 3, 2, 1)                     # [B, F', T, C/2]
        """ 在语谱图的时间轴上求自相关矩阵：F_temp """
        Ft_Q = Ft_V = Ft_K = conv_Fres_t.reshape(conv_Fres_t.shape[0], conv_Fres_t.shape[1], -1)  # [B, T, (C/2 * F')]  这一步就是论文中的Reshape_t:[B, T, F', C/2]-->[B, T, (C/2 * F')]
        # print(Ft_Q.shape)
        attention_t = F.softmax(torch.matmul(Ft_Q, Ft_K.permute(0, 2, 1)) / sqrt(num_C * len_F / 2), dim=-1)  # [B, T, T]
        # print(attention_t.squeeze(0).shape)
        # plt.imshow(attention_t.squeeze(0).detach().numpy())
        # plt.show()
        SA_t = torch.matmul(attention_t, Ft_V)  # SA_t: [B, T, F*C/2]
        Reshape_SA_t = SA_t.reshape(SA_t.shape[0], SA_t.shape[1], len_F, num_C // 2)  # [B, T, F', C/2]
        F_temp = F_res.permute(0, 3, 1, 2) + self.conv_UpChannel(Reshape_SA_t.permute(0, 3, 1, 2))  # [B, C, T, F']
        F_temp = F_temp.permute(0, 2, 3, 1)  # F_temp: [B, T, F', C]


        """ 在语谱图的频域轴上求自相关矩阵：F_freq """
        
        Ff_Q = Ff_V = Ff_K = conv_Fres_f.reshape(conv_Fres_f.shape[0], conv_Fres_f.shape[1], -1)  # [B, F', (C/2 * T)]  这一步就是论文中的Reshape_f:[B, T, F', C/2]-->[B, F', (C/2 * T)]
        attention_f = F.softmax(torch.matmul(Ff_Q, Ff_K.permute(0, 2, 1)) / sqrt(num_C * len_T / 2), dim=-1)   # [B, F, F]
        # print(attention_f.squeeze(0).shape)
        # plt.imshow(attention_f.squeeze(0).detach().numpy())
        # plt.show()
        SA_f = torch.matmul(attention_f, Ff_V)  # SA_t: [B, F, T*C/2]
        Reshape_SA_f = SA_f.reshape(SA_f.shape[0], SA_f.shape[1], len_T, num_C // 2)  # [B, F', T, C/2]
        F_freq = F_res.permute(0, 3, 2, 1) + self.conv_UpChannel(Reshape_SA_f.permute(0, 3, 1, 2))  # [B, C, F', T]
        F_freq = F_freq.permute(0, 3, 2, 1)  # F_temp: [B, T, F', C]

        """ cat、conv """
        # F_cat = torch.cat((F_res, F_freq, F_temp), dim=-1)    # F_cat: [B, T, F', 3*C]  在通道维度上做cat
        F_cat = torch.cat((F_res, F_freq, F_temp), dim=-1)      # F_cat: [B, T, F', 3*C]  在通道维度上做cat
        F_RA = self.conv_out(F_cat.permute(0, 3, 1, 2))         # [B, 3*C, T, F] -> [B, C, T, F]
        F_RA = F_RA.permute(0, 2, 3, 1)                         # F_RA：[B, T, F', C]

        return (F_res, F_temp, F_freq), F_RA
        # return (F_res, F_freq), F_RA


if __name__ == '__main__':
    Fres = torch.randn(1, 100, 128, 64)  # [B, T, F', C] attention模块的输入是F_res
    a = Fres
    attention = Attention_block()
    (F_res, F_temp, F_freq), F_RA = attention(Fres)
    print(F_res.shape, F_temp.shape, F_freq.shape, F_RA.shape)
    # (F_res, F_freq), F_RA = attention(Fres)
    # print(F_res.shape, F_freq.shape, F_RA.shape)
