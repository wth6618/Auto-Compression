'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from qnn import *
from qnn import func as qf

def quantized_activation(a_bits):
    return nn.Sequential(nn.ReLU(), qf.LogQuant(a_bits, unsigned=True))

class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, w_bits, a_bits,stride=1, cfg = None):
        super(QBasicBlock, self).__init__()
        self.a_bits = a_bits
        if not cfg:
            self.conv1 = QConv2d(planes, planes, 3, qf.QTanh(w_bits), stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = QConv2d(planes, planes, 3, qf.QTanh(w_bits),stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.conv1 = QConv2d(in_planes, cfg, 3, qf.QTanh(w_bits), stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg)
            self.conv2 = QConv2d(cfg, planes, 3, qf.QTanh(w_bits), stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = quantized_activation(self.a_bits)(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = quantized_activation(self.a_bits)(out)
        return out

# change to quant
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QResNet(nn.Module):
    def __init__(self, block, num_blocks,w_bits,a_bits, num_classes=10, cfg = None):
        super(QResNet, self).__init__()
        self.in_planes = 64
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.cfg = cfg

        self.conv1 = QConv2d(3, 64, 3, qf.QTanh(w_bits),stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.cfg:
            offset = 0
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cfg=cfg[offset:offset+num_blocks[0]])
            offset += num_blocks[0]
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cfg=cfg[offset:offset + num_blocks[1]])
            offset += num_blocks[1]
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cfg=cfg[offset: offset + num_blocks[2]])
            offset += num_blocks[2]
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cfg=cfg[offset: offset + num_blocks[3]])
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, cfg):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if cfg:
            for stride, cfg in zip(strides, cfg):
                layers.append(block(self.in_planes, planes, self.w_bits,self.a_bits, stride, cfg))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes,self.w_bits,self.a_bits, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        out = quantized_activation(self.a_bits)(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def QResNet18(w_bits, a_bits, cfg=None):
    return QResNet(QBasicBlock, [2, 2, 2, 2], w_bits, a_bits ,cfg=cfg)


def QResNet34(w_bits, a_bits,cfg = None):
    return QResNet(QBasicBlock, [3, 4, 6, 3], w_bits, a_bits ,cfg=cfg)


def QResNet50(w_bits, a_bits,cfg = None):
    return QResNet(Bottleneck, [3, 4, 6, 3], w_bits, a_bits ,cfg=cfg)


def QResNet101(w_bits, a_bits,cfg = None):
    return QResNet(Bottleneck, [3, 4, 23, 3],w_bits, a_bits ,cfg=cfg)


def QResNet152(w_bits, a_bits,cfg = None):
    return QResNet(Bottleneck, [3, 8, 36, 3],w_bits, a_bits ,cfg=cfg)


def test():
    net = QResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
