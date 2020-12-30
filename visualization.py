'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from q_models import *
from utils import *

import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--folder', default='./image', type=str,  metavar='PATH',help='folder name')
parser.add_argument('--path1', default='./baselines/resNet20.pth', type=str,  metavar='PATH',help='path to model1')
parser.add_argument('--path2', default='./checkpoint/resNet20.pth', type=str,  metavar='PATH',help='path to model2')
parser.add_argument('--path3', default='./baselines/resNet20.pth', type=str,  metavar='PATH',help='path to model3')
parser.add_argument('--path4', default='./checkpoint/resNet20.pth', type=str,  metavar='PATH',help='path to model4')
parser.add_argument('--title', default='Prune->Quant, Quant->Prune Comparison', type=str, help='title of the fig')
parser.add_argument('--subtitle', default='resNet20 Prune 0.6, w=8, a=8', type=str, help='subtitle of the fig')
parser.add_argument('--save', default='resNet20_p0.6_q8.png', type=str, help='save fig')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', '-b', default=128 ,type=int, help='batch size')                    
parser.add_argument('--device_name', '-d', default=9 ,type=int, help='cuda device number')
parser.add_argument('--dataparallel', '-dp', default=False ,type=bool, help='saved using nn.parallel')
parser.add_argument('--w', type=int, default=8, help="w_bits")
parser.add_argument('--a', type=int, default=8, help="a_bits")


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# net = resnet20_cifar()


# load model 1
checkpoint = torch.load(args.path1)

cfg1 = checkpoint['cfg']

mask1 = checkpoint['mask']


# print("loaded model1 from {} acc {}\n cfg :{}".format(args.path1, checkpoint['acc'], cfg1))


# load model 2
checkpoint = torch.load(args.path2)

cfg2 = checkpoint['cfg']
mask2 = checkpoint['mask']


assert len(mask1) == len(mask2), 'two model can not be compared'
diff = []
fig, axis = plt.subplots(len(mask1))
fig.suptitle(args.title)
axis[0].set_title(args.subtitle)
count = 0
total = 0
for m1,m2 in zip(mask1.items(), mask2.items()):
    assert m1[0] == m2[0], "unmatched layer"

    temp = np.where(m1[1].cpu() != m2[1].cpu())
    total+= len(temp[0])
    diff.append(temp)
    label = "layer{}".format(m1[0])
    data = [m1[1].cpu().numpy(), m2[1].cpu().numpy()]
    # [m1[1].numpy(), m2[1].numpy()]
    im = axis[count].imshow(data)

    if(len(temp[0] > 1)):
        axis[count].set_ylabel(label, fontsize = 4, color = 'red')
    else:
        axis[count].set_ylabel(label, fontsize=4)

    axis[count].set_xticks(np.arange(len(m1[1])))
    axis[count].set_yticks(np.arange(2))
    # ... and label them with the respective list entries
    axis[count].set_xticklabels(range(len(m1[1])), fontsize = 4)
    axis[count].set_yticklabels(['pq','qp'], fontsize = 6)

    count += 1
axis[count-1].set_xlabel('channels')
print(total)
print("saving image as {}".format(args.save))
plt.savefig(args.save, dpi = 600)
# print(diff)


fig, axis = plt.subplots(1)
fig.suptitle(args.title)
axis.set_title(args.subtitle)
X = []
ratio1, ratio2 = [], []
total = 0
for m1,m2 in zip(mask1.items(), mask2.items()):
    X.append(m1[0])
    data1, data2 = m1[1].cpu().numpy(), m2[1].cpu().numpy()
    ratio1.append(1- (np.sum(data1) / data1.size))
    ratio2.append(1- (np.sum(data2) / data2.size))

print(ratio2)
print(len(list(zip(X,ratio1))))


plt.plot(X, ratio1, label="prune->quant")

plt.plot(X, ratio2, label="quant->prune")
axis.scatter(X,ratio1, s=4)
axis.scatter(X,ratio2, s=4)
plt.xticks(np.arange(min(X), max(X)+1, 1.0))
plt.yticks(np.arange(0.0, 1.0, 0.05))
axis.set_xlabel("Conv2d Layer")
axis.set_ylabel("Prune ratio")
legend = axis.legend(shadow=True)
filename = "prune_ratio_" + args.save
print("saving image as {}".format(filename))
plt.savefig(filename, dpi = 400)

#
#
# fig, axis = plt.subplots(1)
# fig.suptitle(args.title)
# axis.set_title(args.subtitle)
# X = []
# ratio1, ratio2 = [], []
# ratio3, ratio4 = [], []
# total = 0
# for m1,m2,m3,m4 in zip(mask1.items(), mask2.items(), mask3.items(), mask4.items()):
#     X.append(m1[0])
#     data1, data2 = m1[1].cpu().numpy(), m2[1].cpu().numpy()
#     data3, data4 = m3[1].cpu().numpy(), m4[1].cpu().numpy()
#     ratio1.append(1- (np.sum(data1) / data1.size))
#     ratio2.append(1- (np.sum(data2) / data2.size))
#     ratio3.append(1- (np.sum(data3) / data3.size))
#     ratio4.append(1- (np.sum(data4) / data4.size))
#
#
# print(len(list(zip(X,ratio1))))
#
# plt.plot(X, ratio1, label="quant32")
#
# plt.plot(X, ratio2, label="quant16")
# axis.scatter(X,ratio1, s=4)
# axis.scatter(X,ratio2, s=4)
#
# plt.plot(X, ratio3, label="quant8")
#
# plt.plot(X, ratio4, label="quant4")
# axis.scatter(X,ratio3, s=4)
# axis.scatter(X,ratio4, s=4)
# plt.xticks(np.arange(min(X), max(X)+1, 1.0))
# plt.yticks(np.arange(0.0, 1.0, 0.05))
# axis.set_xlabel("Conv2d Layer")
# axis.set_ylabel("Prune ratio")
# legend = axis.legend(shadow=True)
#
# print("saving image as {}".format(args.save))
# plt.savefig(args.save, dpi = 400)
# #
#
plt.show()
#
#
#





