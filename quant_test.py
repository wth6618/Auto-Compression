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
import numpy as np
import pandas as pd
from q_models import *
from utils import *
from prune import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--device_name', '-d', default=9 ,type=int, help='cuda device number')
parser.add_argument('--w', type=int, default=4, help="w_bits")
parser.add_argument('--a', type=int, default=4, help="a_bits")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if device == 'cuda':
    torch.cuda.set_device(0)

# Data
print('==> Preparing data..')



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

print('Prepare data done!')


checkpoint = torch.load('./pruned_model/resNet18.pth', map_location='cuda:0')
cfg = checkpoint['cfg']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print("loaded pruned model with accuracy {}\n cfg: {}".format( best_acc, cfg))
net = QResNet18(args.w, args.a , cfg=cfg)
net.load_state_dict(checkpoint['state_dict'], strict=False)

print("w_bits {}, a_bits {}".format(net.w_bits, net.a_bits))

print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
best_acc = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, acc_list):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    acc_list.append(acc)
    print("accuracy: {},".format(acc))
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'cfg':cfg
        }
        if not os.path.isdir('experiment_results/p_q'):
            os.mkdir('experiment_results/p_q')
        torch.save(state, './experiment_results/p_q/resNet18_p_q.pth')
        best_acc = acc
    return (acc, epoch)


acc_list = []
best = 0
end_epoch = start_epoch+2
net = net.to(device)
for epoch in range(start_epoch, end_epoch):
    if epoch in [end_epoch * 0.5, end_epoch * 0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    best,e = test(epoch, acc_list)
acc_list = np.array(acc_list)
print(acc_list)
pd.DataFrame(acc_list).to_csv("experiment_results/p_q/resnet18_acc.csv")
print('Saving..')
state = {
    'state_dict': net.state_dict(),
    'acc': best,
    'epoch': e,
    'cfg': cfg
}
if not os.path.isdir('experiment_results/p_q'):
    os.mkdir('experiment_results/p_q')
torch.save(state, './experiment_results/p_q/resNet18_p_q.pth')

