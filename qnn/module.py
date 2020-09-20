import torch
import torch.nn.functional as F
from . import func as qf


def weight_quant(weight, quantizer):
    if quantizer is not None:
        return quantizer(weight)
    else:
        return weight


class QConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, quantizer: qf.QuantFunc,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        self.w_quant = quantizer

    def forward(self, x):
        w = weight_quant(self.weight, self.w_quant)
        output = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quantizer: qf.QuantFunc,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        self.w_quant = quantizer

    def forward(self, x):
        w = weight_quant(self.weight, self.w_quant)
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QConv3d(torch.nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, quantizer: qf.QuantFunc,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)
        self.w_quant = quantizer

    def forward(self, x):
        w = weight_quant(self.weight, self.w_quant)
        output = F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QLinear(torch.nn.Linear):
    def __init__(self, in_dims, out_dims, quantizer: qf.QuantFunc, bias=True):
        super(QLinear, self).__init__(in_dims, out_dims, bias)
        self.w_quant = quantizer

    def forward(self, x):
        w = weight_quant(self.weight, self.w_quant)
        output = F.linear(x, w, self.bias)
        return output