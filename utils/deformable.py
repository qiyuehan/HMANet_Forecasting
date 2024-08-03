import torch.nn as nn
import torch
import numpy as np
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 原卷积

        self.conv_offset = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, 3, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0

        self.conv_mask = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, 3, 3, 3]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))  # 保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset,
                                            weight=self.conv.weight,
                                            mask=mask, padding=(1, 1))
        return out


import torchvision.ops as ops
class Def_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, dilation_rate=1, groups=1):
        super(Def_Conv, self).__init__()
        # p = (k - 1) // 2

        # self.split_size = (2 * in_c * in_c, in_c * in_c)  # (18,9)  18+9=27
        self.split_size = (2 * kernel_size * kernel_size, kernel_size * kernel_size)  # (18,9)  18+9=27
        # self.conv_offset = nn.Conv2d(in_c, 3 * k * k, k, padding=p)  # (3,27,3,1,1)
        # self.conv_offset = nn.Conv2d(in_c, 3 * in_c * in_c, kernel_size, padding=dilation_rate, dilation=dilation_rate, groups=groups)  # (3,27,3,1,1)
        self.conv_offset = nn.Conv2d(in_c, 3 * kernel_size * kernel_size, kernel_size, padding=dilation_rate, dilation=dilation_rate)  # (3,27,3,1,1)
        # self.conv_deform = ops.DeformConv2d(in_c, out_c, k, padding=p)  # group=n
        self.conv_deform = ops.DeformConv2d(in_c, out_c, kernel_size, padding=dilation_rate, dilation=dilation_rate, groups=groups)  # group=n

        # initialize
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.kaiming_normal_(self.conv_deform.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        offset, mask = torch.split(self.conv_offset(x), self.split_size, dim=1)  # ,用于指定如何分割输入张量的参数。它可以是一个整数（表示每个子张量的尺寸），也可以是一个整数列表（表示每个子张量的尺寸），或者是一个元组，包含整数或整数列表。
        mask = torch.sigmoid(mask)
        def_out = self.conv_deform(x, offset, mask)
        return def_out


if __name__ == '__main__':
    input = torch.rand(32,42,7,12)
    net = Def_Conv(42, 70, 3, 5)  # in_c, out_c, kernel_size=3, dilation_rate=1, groups=1
    output = net(input)

#
# if __name__ == '__main__':
#     x = torch.arange(64*3).view(2, 3, 4, 8).float()
#     m = net()
#     y = m(x)
