'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from q_models import *
from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--method', default='l1norm', type=str,  metavar='method',help='pruning method')
parser.add_argument('--path', default='./baselines/resNet20.pth', type=str,  metavar='PATH',help='path to model')
parser.add_argument('--save', default='./checkpoint/resNet20.pth', type=str,  metavar='PATH',help='save path to model')

parser.add_argument('--epoch', '-e', default=150 ,type=int, help='epoch')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', '-b', default=128 ,type=int, help='batch size')
parser.add_argument('--device_name', '-d', default=9 ,type=int, help='cuda device number')
parser.add_argument('--pruned', '-p', action='store_true', help='if input model pruned')
parser.add_argument('--quant', '-q', action='store_true', help='if input model quant')
parser.add_argument('--dataparallel', '-dp', default=False ,type=bool, help='saved using nn.parallel')
parser.add_argument('--w', type=int, default=8, help="w_bits")
parser.add_argument('--a', type=int, default=8, help="a_bits")
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

trainloader, testloader = load_data(args.ct, args.batch_size)

# Model
print('==> Building model..')

if args.quant:
    net = qresnet20_cifar(w_bits = args.w, a_bits = args.a, num_classes = args.ct)
else:
    net = resnet20_cifar(num_classes = args.ct)
# net = qresnet20_cifar(w_bits = args.w, a_bits = args.a, num_classes = 100)
# net = resnet20_cifar()



cfg = []
mask = []


checkpoint = torch.load(args.path)

if args.pruned:
    cfg = checkpoint['cfg']
    mask = checkpoint['mask']
    net = ApplyCFG(net, cfg).to(device)
    print("loaded pruned model with cfg: {}".format( cfg))

if args.dataparallel:
    print("parallel")
    checkpoint = loadParalData(checkpoint)
net.load_state_dict(checkpoint['state_dict'], strict = False)

print("loaded model with acc {}".format(checkpoint['acc']))

exit(0)

best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']  # start from epoch 0 or last checkpoint epoch



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
net = net.to(device)




ep = args.epoch
end_epoch = start_epoch+ep
# net = nn.DataParallel(net).to(device)
net = net.to(device)
for epoch in range(start_epoch, end_epoch):
    if epoch in [start_epoch + ep * 0.15, start_epoch + ep * 0.3, start_epoch + ep * 0.5, start_epoch + ep * 0.75, start_epoch + ep * 0.9]:
        print('lr decrease')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    train(trainloader, net, criterion, optimizer, epoch, device)
    acc = validate(testloader, net, criterion,device)

    if acc > best_acc:

        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'cfg': cfg,
            'mask': mask
        }
        print("Saving to {}".format(args.save))

        torch.save(state, args.save)
        best_acc = acc



