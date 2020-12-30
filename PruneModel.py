import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from q_models import *
from utils import *
from prune import *

from collections import OrderedDict
import collections

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ct', '--cifar-type', default='10', type=int, metavar='CT',
                    help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--method', default='l1norm', type=str, metavar='method', help='pruning method')
parser.add_argument('--path', default='./baselines/resNet20.pth', type=str, metavar='PATH', help='path to model')
parser.add_argument('--save', default='./checkpoint/resNet20.pth', type=str, metavar='PATH', help='save path to model')
parser.add_argument('--ratio', default='0.2', type=float, metavar='method', help='pruning method')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epoch', '-e', default=1, type=int, help='epoch')
parser.add_argument('--quant', '-q', action='store_true', help='if input model quant')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
parser.add_argument('--device_name', '-d', default=9, type=int, help='cuda device number')
parser.add_argument('--dataparallel', '-dp', default=False, type=bool, help='saved using nn.parallel')
parser.add_argument('--w', type=int, default=8, help="w_bits")
parser.add_argument('--a', type=int, default=8, help="a_bits")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

trainloader, testloader = load_data(args.ct, args.batch_size)

# Model
print('==> Building model..')

# net = MobileNetV2()

# print(net)

mask, cfg = [], []
if args.quant:
    net = qresnet20_cifar(w_bits=args.w, a_bits=args.a, num_classes=args.ct)
else:
    net = resnet20_cifar(num_classes=args.ct)
checkpoint = torch.load(args.path)
if args.dataparallel:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
else:
    net.load_state_dict(checkpoint['state_dict'], strict=False)

start_epoch = checkpoint['epoch']
print("loaded model with accuracy {}".format(checkpoint['acc']))
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
net = net.to(device)
if (args.method == 'l1norm'):
    pruner = structured_L1(net, pruning_rate=args.ratio)
elif (args.method == 'unstructured'):
    pruner = Unstructured(net, pruning_rate=args.ratio)
elif (args.method == 'vector'):
    pruner = VectorPruning(net, pruning_rate=args.ratio)
elif (args.method == 'sensitivity'):
    pruner = Sensitivity_Pruning(net)
elif (args.method == 'slim'):
    pruner = Network_Slimming(net, pruning_rate=args.ratio)
else:
    print('method not implemented')
    exit(0)

pruner.step()
pruner.zero_params()
if args.method == 'slim' or args.method == 'l1norm':
    keys = sorted(pruner.prune_out + pruner.prune_both + pruner.prune_in)

    assert len(keys) == len(pruner.cfg_mask), 'something wrong'

    p_total, remain = 0, 0
    total = pruner.total
    mask = collections.defaultdict(set)
    for idx in range(len(keys)):
        mask[keys[idx]] = pruner.cfg_mask[idx]
        remain += int(torch.sum(pruner.cfg_mask[idx]))
        p_total += int(pruner.cfg_mask[idx].nelement())

    print("total {}, pruned {}, acutal prune ratio:{:.2f}".format(total, p_total - remain, 1 - (remain / p_total)))
    cfg = pruner.cfg

filename = args.save

# print(net)

# net = nn.DataParallel(net).to(device)
net = net.to(device)

acc = validate(testloader, net, criterion, device)

state = {
    'state_dict': net.state_dict(),
    'acc': acc,
    'epoch': 0,
    'cfg': cfg,
    'mask': mask
}
print("Saving to {}".format(args.save))
torch.save(state, args.save)
best_acc = acc




