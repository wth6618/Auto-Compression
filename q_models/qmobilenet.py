'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from qnn import *
from qnn import func as qf

def quantized_activation(a_bits):
    return nn.Sequential(nn.ReLU(), qf.LogQuant(a_bits, unsigned=True))

class QBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes,w_bits, a_bits, stride=1, cfg = None):
        super(QBlock, self).__init__()
        self.a_bits = a_bits
        if cfg:
            self.conv1 = QConv2d(in_planes, cfg, 3,qf.QTanh(w_bits), stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg)
            self.conv2 = QConv2d(cfg, out_planes, 1, qf.QTanh(w_bits), stride = 1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.conv1 = QConv2d(in_planes, in_planes, 3, qf.QTanh(w_bits), stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = QConv2d(in_planes, out_planes, 1, qf.QTanh(w_bits),stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = quantized_activation(self.a_bits)(self.bn1(self.conv1(x)))
        out = quantized_activation(self.a_bits)(self.bn2(self.conv2(out)))
        return out


class QMobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self,w_bits,a_bits, num_classes=10,p_cfg=None):

        super(QMobileNet, self).__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.conv1 = QConv2d(3, 32, 3,qf.QTanh(w_bits) ,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.p_cfg = p_cfg
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        if self.p_cfg:
            idx = 0
            for x in self.cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(QBlock(in_planes, out_planes, self.w_bits,self.a_bits,stride, self.p_cfg[idx]))
                idx += 1
                in_planes = out_planes
        else:
            for x in self.cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(QBlock(in_planes, out_planes,self.w_bits,self.a_bits, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = quantized_activation(self.a_bits)(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



# test()
