import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

# 首先将输入x分成高频、低频2部分
class Disassemble(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(Disassemble, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) # 低频，就像GAP是DCT最低频的特殊情况？长宽缩小一半
        X_h = x
        # 分割后的高频特征[b,c,h,w]->[b,c-c/2,h,w]
        X_h = self.h2h(X_h)
        # 分割后的低频特征[b,c,h,w]->[b,c/2,h/2,w/2]
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class SupplyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(SupplyConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        # 高频特征转化为低频，之后加上l2l
        X_h2l = self.h2g_pool(X_h)
        
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        
        # 连接部分，原始的高频特征h2h与低频转高频特征l2h连接
        X_h = X_l2h + X_h2h
        # 连接部分，原始的低特征l2l与高频转低频特征h2l连接
        X_l = X_h2l + X_l2l

        return X_h, X_l

class FuseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FuseConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        # 这里的叫法h2h和l2h，感觉没太大必要

        # 先将2个分量的通道对齐输出通道
        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道

        # 通过双线性插值法恢复低频组件的h、w
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h  # 本来的设置：高频低频融合输出
        return X_h       #都输出

        # return X_h2h  #只输出高频组
        # return X_l2h    #只输出低频组

        # return X_h, X_h2h, X_l2h

class FIF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(FIF, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = Disassemble(in_channels, out_channels, kernel_size)
        # 第二层，低高频输入，低高频输出
        self.mid1 = SupplyConv(in_channels, in_channels, kernel_size)
        self.mid2 = SupplyConv(in_channels, out_channels, kernel_size)
        # 第三层，将低高频汇合后输出
        self.lst = FuseConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        # 分割后的高频特征x_h:[b,c,h,w]->[b,c-c/2,h,w]
        # 分割后的低频特征x_l:[b,c,h,w]->[b,c/2,h/2,w/2]
        x_h, x_l = self.fir(x)                   # (1,64,64,64) ,(1,64,32,32)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) # (1,64,64,64) ,(1,64,32,32)
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) # (1,32,64,64) ,(1,32,32,32)

        x_ret = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        return x_ret

        # x_l_11 = F.interpolate(x_l_1, (int(x_h_1.size()[2]), int(x_h_1.size()[3])), mode='bilinear')
        # x_ret, x_h_6, x_l_6 = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        # return x0, x_ret,x_hh, x_ll,x_h_1, x_l_1

        # return x0, x_ret, x_hh, x_ll, x_h_6, x_l_6
        # return x0, x_ret
    # fea_name = ['_before','_after', '_beforeH', '_beforeL', '_afterH', '_afterL', '_afterH0', '_afterL0']

